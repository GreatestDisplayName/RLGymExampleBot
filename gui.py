import sys
from pathlib import Path

# Add the 'src' directory to sys.path
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from ttkthemes import ThemedTk
from src.launch_training import TrainingLauncher
from src.logger import logger
from src.self_play_league import SelfPlayLeague
from src.utils import SUPPORTED_AGENT_TYPES
import json
import logging
import queue
import threading
from logging.handlers import QueueHandler, QueueListener
from concurrent.futures import ThreadPoolExecutor
from jsonschema import validate, ValidationError
import optuna
import subprocess

THEME_SETTINGS_FILE = Path("theme_settings.json")

# Basic schema for configuration validation
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "agent_type": {"type": "string", "enum": SUPPORTED_AGENT_TYPES},
        "timesteps": {"type": "integer", "minimum": 1},
        "env_settings": {"type": "object"}
    },
    "required": ["agent_type", "timesteps"]
}

class ThreadSafeTextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.after(0, lambda: self._update_text(msg))
    
    def _update_text(self, msg):
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

class TrainingMetricsHandler(logging.Handler):
    def __init__(self, tree_widget, episodes_list, rewards_list, losses_list, ax1, ax2, canvas):
        super().__init__()
        self.tree_widget = tree_widget
        self.episodes_list = episodes_list
        self.rewards_list = rewards_list
        self.losses_list = losses_list
        self.ax1 = ax1
        self.ax2 = ax2
        self.canvas = canvas
        self.setFormatter(logging.Formatter('%(message)s')) # Only interested in the message for parsing

    def emit(self, record):
        msg = self.format(record)
        # Example log format: "Episode: 100, Timesteps: 10000, Avg Reward: 50.2, Loss: 0.123"
        if "Episode:" in msg and "Timesteps:" in msg and "Avg Reward:" in msg:
            try:
                parts = msg.split(", ")
                episode = int(parts[0].split(": ")[1])
                timesteps = int(parts[1].split(": ")[1])
                avg_reward = float(parts[2].split(": ")[1])
                loss = float(parts[3].split(": ")[1]) if len(parts) > 3 and "Loss:" in parts[3] else 0.0 # Default to 0.0 if loss not found
                
                self.tree_widget.after(0, lambda: self.tree_widget.insert("", "end", values=(episode, timesteps, f'{avg_reward:.2f}', f'{loss:.4f}')))
                self.tree_widget.after(0, lambda: self.tree_widget.yview_moveto(1)) # Scroll to bottom

                # Update plot data
                self.episodes_list.append(episode)
                self.rewards_list.append(avg_reward)
                self.losses_list.append(loss)

                self.ax1.clear()
                self.ax2.clear()

                self.ax1.plot(self.episodes_list, self.rewards_list, label='Avg Reward', color='blue')
                self.ax2.plot(self.episodes_list, self.losses_list, label='Loss', color='red')

                self.ax1.set_xlabel("Episode")
                self.ax1.set_ylabel("Avg Reward", color='blue')
                self.ax2.set_ylabel("Loss", color='red')

                self.ax1.tick_params(axis='y', labelcolor='blue')
                self.ax2.tick_params(axis='y', labelcolor='red')

                lines, labels = self.ax1.get_legend_handles_labels()
                lines2, labels2 = self.ax2.get_legend_handles_labels()
                self.ax2.legend(lines + lines2, labels + labels2, loc='upper left')

                self.ax1.grid(True)
                self.canvas.draw()

            except Exception as e:
                # Log parsing error, but don't break the application
                print(f"Error parsing training log: {msg} - {e}")


class RLGymGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("RLGym Workflow Manager")
        master.geometry("1200x800")
        master.minsize(800, 600)
        
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.launcher = TrainingLauncher()
        self.league_manager = SelfPlayLeague()
        self.running_tasks = set()
        self.log_queue = queue.SimpleQueue()
        self.queue_handler = QueueHandler(self.log_queue)

        # For training plot
        self.training_episodes = []
        self.training_avg_rewards = []
        self.training_losses = []

        # For simulation control
        self.running_simulation = False
        self.simulation_thread = None

        # For hyperparameter optimization
        self.optuna_study = None
        self.optimization_thread = None
        self.compare_model_listbox = None # Initialize to None

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.config_frame = ttk.Frame(self.notebook)
        self.training_frame = ttk.Frame(self.notebook)
        self.league_frame = ttk.Frame(self.notebook)
        self.model_management_frame = ttk.Frame(self.notebook)
        self.hyperparam_opt_frame = ttk.Frame(self.notebook)
        self.version_control_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)
        self.logs_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.config_frame, text="Configurations")
        self.notebook.add(self.training_frame, text="Training")
        self.notebook.add(self.league_frame, text="League")
        self.notebook.add(self.model_management_frame, text="Model Management")
        self.notebook.add(self.hyperparam_opt_frame, text="Hyperparameter Optimization")
        self.notebook.add(self.version_control_frame, text="Version Control")
        self.notebook.add(self.settings_frame, text="Settings")
        self.notebook.add(self.logs_frame, text="Logs")

        # Load and apply theme setting before setting up tabs
        self._load_theme_setting()

        self._setup_config_tab()
        self._setup_training_tab()
        self._setup_league_tab()
        self._setup_model_management_tab()
        self._populate_model_list() # Call it here after initialization
        self._setup_hyperparam_opt_tab()
        self._setup_version_control_tab()
        self._setup_settings_tab()
        self._setup_logs_tab()
        self._setup_logging()

        logger.info("RLGym GUI initialized.")
    
    def _setup_logging(self):
        text_handler = ThreadSafeTextHandler(self.logs_text)
        text_handler.setLevel(logging.INFO)
        self.queue_listener = QueueListener(self.log_queue, text_handler)
        self.queue_listener.start()
        logger.logger.addHandler(self.queue_handler)
    
    def _populate_themes_combobox(self):
        # Get available themes from ttkthemes
        available_themes = self.master.get_themes()
        self.theme_combobox['values'] = sorted(available_themes)
        
        # Set current theme
        current_theme = self._load_theme_setting()
        if current_theme and current_theme in available_themes:
            self.theme_combobox.set(current_theme)
            self.master.set_theme(current_theme)
        else:
            # Fallback to default if no setting or invalid theme
            self.theme_combobox.set("arc")
            self.master.set_theme("arc")

    def _apply_theme(self, event=None):
        selected_theme = self.theme_combobox.get()
        if selected_theme:
            try:
                self.master.set_theme(selected_theme)
                self._save_theme_setting(selected_theme)
                logger.info(f"Applied theme: {selected_theme}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply theme: {e}")
                logger.exception(f"Failed to apply theme: {selected_theme}")

    def _save_theme_setting(self, theme_name):
        try:
            with open(THEME_SETTINGS_FILE, 'w') as f:
                json.dump({"theme": theme_name}, f)
        except Exception as e:
            logger.error(f"Failed to save theme setting: {e}")

    def _load_theme_setting(self):
        if THEME_SETTINGS_FILE.exists():
            try:
                with open(THEME_SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                    return settings.get("theme")
            except Exception as e:
                logger.error(f"Failed to load theme setting: {e}")
        return None

    def __del__(self):
        if hasattr(self, 'queue_listener'):
            self.queue_listener.stop()
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)

    def _populate_config_list(self):
        self.config_listbox.delete(0, tk.END)
        default_configs = self.launcher.default_configs.keys()
        custom_configs = [f.stem for f in self.launcher.config_dir.glob("*.json") if f.stem not in default_configs]
        all_configs = sorted(list(default_configs) + list(custom_configs))
        for config_name in all_configs:
            self.config_listbox.insert(tk.END, config_name)
        logger.info("Configuration list populated.")

    def _on_config_select(self, event: tk.Event):
        selected_indices = self.config_listbox.curselection()
        if not selected_indices:
            return

        selected_config_name = self.config_listbox.get(selected_indices[0])
        try:
            config_data = self.launcher.load_config(selected_config_name)
            self.config_details_text.config(state="normal")
            self.config_details_text.delete("1.0", tk.END)
            self.config_details_text.insert(tk.END, json.dumps(config_data, indent=2))
            logger.info(f"Displayed details for config: {selected_config_name}")

            # Validate loaded config
            try:
                validate(instance=config_data, schema=CONFIG_SCHEMA)
                logger.info(f"Config '{selected_config_name}' validated successfully against schema.")
            except ValidationError as e:
                messagebox.showwarning("Validation Warning", f"Configuration '{selected_config_name}' does not conform to schema:\n{e.message}")
                logger.warning(f"Config '{selected_config_name}' validation warning: {e.message}")
        except FileNotFoundError:
            messagebox.showerror("Error", f"Configuration '{selected_config_name}' not found.")
            logger.error(f"Config file not found: {selected_config_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config '{selected_config_name}': {e}")
            logger.exception(f"Failed to load config: {selected_config_name}")

    def _save_config(self):
        selected_indices = self.config_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "No configuration selected to save.")
            return

        selected_config_name = self.config_listbox.get(selected_indices[0])
        config_content = self.config_details_text.get("1.0", tk.END)

        try:
            config_data = json.loads(config_content)
            # Validate config before saving
            try:
                validate(instance=config_data, schema=CONFIG_SCHEMA)
            except ValidationError as e:
                messagebox.showerror("Validation Error", f"Configuration does not conform to schema and cannot be saved:\n{e.message}")
                logger.error(f"Config validation error on save: {e.message}")
                return

            config_path = self.launcher.config_dir / f"{selected_config_name}.json"
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            messagebox.showinfo("Success", f"Configuration '{selected_config_name}' saved successfully.")
            logger.info(f"Configuration '{selected_config_name}' saved.")
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON format. Please check the content.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
            logger.exception(f"Failed to save config: {selected_config_name}")

    def _save_config_as(self):
        config_content = self.config_details_text.get("1.0", tk.END)
        
        try:
            config_data = json.loads(config_content)
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON format. Please check the content before saving.")
            return

        # Validate config before saving
        try:
            validate(instance=config_data, schema=CONFIG_SCHEMA)
        except ValidationError as e:
            messagebox.showerror("Validation Error", f"Configuration does not conform to schema and cannot be saved:\n{e.message}")
            logger.error(f"Config validation error on save as: {e.message}")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Configuration As",
            initialdir=self.launcher.config_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.* אמיתי") ]
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(config_data, f, indent=2)
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
                logger.info(f"Configuration saved as {filename}")
                self._populate_config_list()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")
                logger.exception(f"Failed to save config to {filename}")

    def _create_new_config_dialog(self):
        dialog = tk.Toplevel(self.master)
        dialog.title("Create New Configuration")
        dialog.geometry("400x400")

        ttk.Label(dialog, text="Config Name:").pack(pady=5)
        name_entry = ttk.Entry(dialog)
        name_entry.pack(pady=5)

        ttk.Label(dialog, text=f"Agent Type ({', '.join(SUPPORTED_AGENT_TYPES)}):").pack(pady=5)
        agent_type_entry = ttk.Entry(dialog)
        agent_type_entry.pack(pady=5)
        if SUPPORTED_AGENT_TYPES:
            agent_type_entry.insert(0, SUPPORTED_AGENT_TYPES[0])

        ttk.Label(dialog, text="Timesteps:").pack(pady=5)
        timesteps_entry = ttk.Entry(dialog)
        timesteps_entry.pack(pady=5)
        timesteps_entry.insert(0, "10000")

        def save_new_config():
            config_name = name_entry.get().strip()
            agent_type = agent_type_entry.get().strip()
            timesteps_str = timesteps_entry.get().strip()

            if not config_name or not agent_type or not timesteps_str:
                messagebox.showerror("Error", "All fields are required.")
                return

            try:
                timesteps_int = int(timesteps_str)
                self.launcher.create_config(name=config_name, agent_type=agent_type, timesteps=timesteps_int)
                messagebox.showinfo("Success", f"Config '{config_name}' created successfully.")
                self._populate_config_list()
                # Select the newly created config
                self.config_listbox.selection_clear(0, tk.END)
                index = self.config_listbox.get(0, tk.END).index(config_name)
                self.config_listbox.selection_set(index)
                self.config_listbox.see(index)
                self._on_config_select(None) # Manually trigger selection event
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Timesteps must be an integer.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create config: {e}")
                logger.exception("Failed to create new config.")

        ttk.Button(dialog, text="Save Config", command=save_new_config).pack(pady=10)

    def _populate_training_config_combobox(self):
        all_configs = sorted(list(self.launcher.default_configs.keys()) + [f.stem for f in self.launcher.config_dir.glob("*.json") if f.stem not in self.launcher.default_configs.keys()])
        self.training_config_combobox['values'] = all_configs
        if all_configs:
            self.training_config_combobox.set(all_configs[0])

    def _on_training_config_select(self, event: tk.Event):
        selected_config_name = self.training_config_combobox.get()
        logger.info(f"Selected training config: {selected_config_name}")

    def _launch_training_session(self):
        config_name = self.training_config_combobox.get()
        resume_from = self.resume_from_entry.get().strip()
        test_after = self.test_after_var.get()
        render_test = self.render_test_var.get()

        if not config_name:
            messagebox.showerror("Error", "Please select a training configuration.")
            return

        if resume_from == "":
            resume_from = None

        logger.info(f"Attempting to launch training with config: {config_name}")
        logger.info(f"Resume from: {resume_from}, Test after: {test_after}, Render test: {render_test}")

        # Redirect logger to the training log text widget
        self.training_log_text.config(state="normal")
        self.training_log_text.delete("1.0", tk.END)
        self.training_log_text.insert(tk.END, "Starting training session...\n")
        self.training_log_text.see(tk.END)

        # Clear previous plot data and plot
        self.training_episodes.clear()
        self.training_avg_rewards.clear()
        self.training_losses.clear()
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_xlabel("Episode")
        self.ax1.set_ylabel("Avg Reward", color='blue')
        self.ax2.set_ylabel("Loss", color='red')
        # Combine legends from both axes
        lines, labels = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        self.ax1.grid(True)
        self.canvas.draw()

        self.training_launch_button.config(state="disabled") # Disable button during training

        training_log_handler = ThreadSafeTextHandler(self.training_log_text)
        training_log_handler.setLevel(logging.INFO)
        
        # We need a separate queue and listener for the training log
        training_log_queue = queue.SimpleQueue()
        training_queue_handler = QueueHandler(training_log_queue)
        training_queue_listener = QueueListener(training_log_queue, training_log_handler)
        training_queue_listener.start()

        # Add metrics handler
        metrics_handler = TrainingMetricsHandler(self.training_metrics_tree, self.training_episodes, self.training_avg_rewards, self.training_losses, self.ax1, self.ax2, self.canvas)
        metrics_handler.setLevel(logging.INFO)
        logger.logger.addHandler(metrics_handler)
        
        # Add the handler to the logger and remove it when done
        logger.logger.addHandler(training_queue_handler)

        def training_task():
            try:
                self.launcher.launch_training(config_name=config_name, resume_from=resume_from, test_after=test_after, render_test=render_test)
                self.master.after(0, lambda: messagebox.showinfo("Training Status", "Training session finished."))
            except Exception as e:
                logger.exception("Training session failed.")
                self.master.after(0, lambda: messagebox.showerror("Training Status", f"Training session failed: {e}"))
            finally:
                logger.logger.removeHandler(training_queue_handler)
                training_queue_listener.stop()
                logger.logger.removeHandler(metrics_handler) # Remove metrics handler
                self.master.after(0, lambda: self.training_launch_button.config(state="normal")) # Re-enable button

        # Run training in a separate thread to avoid blocking the GUI
        training_thread = threading.Thread(target=training_task)
        training_thread.start()

    def _populate_player_comboboxes(self):
        try:
            player_names = sorted(list(self.league_manager.players.keys()))
            self.player1_combobox['values'] = player_names
            self.player2_combobox['values'] = player_names
            self.sim_agent1_combobox['values'] = player_names
            self.sim_agent2_combobox['values'] = player_names
            if player_names:
                self.player1_combobox.set(player_names[0])
                self.sim_agent1_combobox.set(player_names[0])
                if len(player_names) > 1:
                    self.player2_combobox.set(player_names[1])
                    self.sim_agent2_combobox.set(player_names[1])
        except Exception as e:
            logger.error(f"Failed to populate player comboboxes: {e}")
            messagebox.showerror("Error", f"Failed to load players for comboboxes: {e}")

    def _update_leaderboard_view(self):
        for i in self.leaderboard_tree.get_children():
            self.leaderboard_tree.delete(i)
        
        leaderboard = self.league_manager.get_leaderboard()
        for i, player in enumerate(leaderboard):
            self.leaderboard_tree.insert("", "end", values=(i + 1, player.name, player.rating, player.wins, player.losses, player.draws))

    def _play_league_match(self):
        player1_name = self.player1_combobox.get()
        player2_name = self.player2_combobox.get()
        num_games_str = self.num_games_entry.get().strip()

        if not player1_name or not player2_name:
            messagebox.showerror("Error", "Please select both players.")
            return
        if player1_name == player2_name:
            messagebox.showerror("Error", "Cannot play a match against yourself.")
            return

        try:
            num_games = int(num_games_str)
            if num_games < 1:
                messagebox.showerror("Error", "Number of games must be at least 1.")
                return
        except ValueError:
            messagebox.showerror("Error", "Number of games must be an integer.")
            return

        self._set_match_ui_state(False)
        self._create_progress_dialog("Playing Match", f"Playing {num_games} match(es) between {player1_name} and {player2_name}...")

        future = self.thread_pool.submit(self._run_match, player1_name, player2_name, num_games)
        self.running_tasks.add(future)
        self.master.after(100, lambda: self._check_match_completion(future))
    
    def _run_match(self, player1_name, player2_name, num_games):
        try:
            logger.info(f"Attempting to play {num_games} match(es) between {player1_name} and {player2_name}")
            match_results = self.league_manager.play_match(player1_name, player2_name, n_games=num_games)
            return {"success": True, "results": match_results}
        except Exception as e:
            logger.error(f"Error playing match: {e}")
            return {"success": False, "error": str(e)}
    
    def _check_match_completion(self, future):
        if future.done():
            self.running_tasks.discard(future)
            self._close_progress_dialog()
            self._set_match_ui_state(True)
            
            try:
                result = future.result()
                if result["success"]:
                    messagebox.showinfo("Match Results", "Matches completed! Check logs for details.")
                    self._update_leaderboard_view()
                else:
                    messagebox.showerror("Error", f"Failed to play match: {result['error']}")
            except Exception as e:
                messagebox.showerror("Error", f"Unexpected error: {str(e)}")
        else:
            self.master.after(100, lambda: self._check_match_completion(future))

    def _start_simulation(self):
        agent1_name = self.sim_agent1_combobox.get()
        agent2_name = self.sim_agent2_combobox.get()
        num_games_str = self.sim_num_games_entry.get().strip() # New: Get number of games

        if not agent1_name or not agent2_name:
            messagebox.showerror("Error", "Please select both agents for simulation.")
            return
        if agent1_name == agent2_name:
            messagebox.showerror("Error", "Cannot simulate an agent against itself.")
            return
        
        try:
            num_games = int(num_games_str)
            if num_games < 1:
                messagebox.showerror("Error", "Number of games must be at least 1.")
                return
        except ValueError:
            messagebox.showerror("Error", "Number of games must be an integer.")
            return

        self.start_sim_button.config(state="disabled")
        self.stop_sim_button.config(state="normal")
        self.sim_agent1_combobox.config(state="disabled")
        self.sim_agent2_combobox.config(state="disabled")
        self.sim_num_games_entry.config(state="disabled") # Disable num games entry

        self.simulation_log_text.config(state="normal")
        self.simulation_log_text.delete("1.0", tk.END)
        self.simulation_log_text.insert(tk.END, f"Starting simulation: {agent1_name} vs {agent2_name} for {num_games} games...\n")
        self.simulation_log_text.config(state="disabled")

        self._create_progress_dialog("Running Simulation", f"Simulating {agent1_name} vs {agent2_name}...")

        self.running_simulation = True
        self.simulation_thread = threading.Thread(target=self._run_simulation_task, args=(agent1_name, agent2_name, num_games))
        self.simulation_thread.start()

    def _run_simulation_task(self, agent1_name, agent2_name, num_games):
        try:
            logger.info(f"Simulation task started: {agent1_name} vs {agent2_name} for {num_games} games.")
            match_results = []
            for i in range(num_games):
                if not self.running_simulation:
                    logger.info("Simulation stopped by user.")
                    break
                
                self.master.after(0, lambda: self._update_progress_dialog(f"Playing game {i+1}/{num_games}..."))
                
                # Call play_match from league_manager
                # Note: play_match returns a list of MatchResult objects
                game_results = self.league_manager.play_match(agent1_name, agent2_name, n_games=1)
                match_results.extend(game_results)

                # Log game result
                if game_results:
                    result = game_results[0]
                    self.master.after(0, lambda r=result: self._log_simulation_game_result(r))
                
            if self.running_simulation: # Only show success if not stopped by user
                self.master.after(0, lambda: self._display_simulation_results(match_results))
            
        except Exception as e:
            logger.exception("Simulation failed.")
            self.master.after(0, lambda: messagebox.showerror("Simulation", f"Simulation failed: {e}"))
        finally:
            self.master.after(0, self._reset_simulation_ui)
            self.master.after(0, self._close_progress_dialog)

    def _log_simulation_game_result(self, result):
        self.simulation_log_text.config(state="normal")
        self.simulation_log_text.insert(tk.END, f"Game {len(self.simulation_log_text.get('1.0', tk.END).splitlines())-1}: {result.winner} wins ({result.reason})\n")
        self.simulation_log_text.see(tk.END)
        self.simulation_log_text.config(state="disabled")

    def _display_simulation_results(self, match_results):
        total_wins1 = sum(1 for m in match_results if m.winner == self.sim_agent1_combobox.get())
        total_wins2 = sum(1 for m in match_results if m.winner == self.sim_agent2_combobox.get())
        total_draws = sum(1 for m in match_results if m.winner == "draw")
        
        message = f"Simulation Completed!\n\n" \
                  f"{self.sim_agent1_combobox.get()}: {total_wins1} wins\n" \
                  f"{self.sim_agent2_combobox.get()}: {total_wins2} wins\n" \
                  f"Draws: {total_draws}\n" \
                  f"Total Games: {len(match_results)}"
        messagebox.showinfo("Simulation Results", message)

    def _update_progress_dialog(self, message):
        if hasattr(self, 'progress_window') and self.progress_window:
            # Find the label and update its text
            for widget in self.progress_window.winfo_children():
                if isinstance(widget, ttk.Label):
                    widget.config(text=message)
                    break
            self.progress_window.update()

    def _stop_simulation(self):
        self.running_simulation = False
        self._close_progress_dialog()
        self._reset_simulation_ui()
        logger.info("Simulation stop requested.")

    def _reset_simulation_ui(self):
        self.start_sim_button.config(state="normal")
        self.stop_sim_button.config(state="disabled")
        self.sim_agent1_combobox.config(state="normal")
        self.sim_agent2_combobox.config(state="normal")
    
    def _set_match_ui_state(self, enabled):
        state = "normal" if enabled else "disabled"
        self.play_match_button.config(state=state)
        self.player1_combobox.config(state=state)
        self.player2_combobox.config(state=state)
        self.num_games_entry.config(state=state)
    
    def _create_progress_dialog(self, title, message):
        self.progress_window = tk.Toplevel(self.master)
        self.progress_window.title(title)
        self.progress_window.geometry("300x100")
        self.progress_window.resizable(False, False)
        self.progress_window.transient(self.master)
        self.progress_window.grab_set()
        self.progress_window.geometry("+%d+%d" % (self.master.winfo_rootx() + 450, self.master.winfo_rooty() + 300))
        ttk.Label(self.progress_window, text=message, wraplength=280).pack(pady=20)
        progress_bar = ttk.Progressbar(self.progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, padx=20, fill='x')
        progress_bar.start()
        self.progress_window.update()
    
    def _close_progress_dialog(self):
        if hasattr(self, 'progress_window') and self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None

    def _browse_model_file(self):
        filetypes = [("Model files", "*.pth *.zip *.pkl"), ("PyTorch files", "*.pth"), ("Zip files", "*.zip"), ("Pickle files", "*.pkl"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(title="Select Model File", filetypes=filetypes, initialdir=Path("models") if Path("models").exists() else Path.cwd())
        if filename:
            self.add_player_model_path_entry.delete(0, tk.END)
            self.add_player_model_path_entry.insert(0, filename)

    def _add_league_player(self):
        name = self.add_player_name_entry.get().strip()
        agent_type = self.add_player_agent_type_combobox.get().strip()
        model_path = self.add_player_model_path_entry.get().strip()

        if not name or not agent_type:
            messagebox.showerror("Error", "Player name and agent type are required.")
            return
        if name in self.league_manager.players:
            messagebox.showerror("Error", f"Player '{name}' already exists.")
            return
        if model_path and not Path(model_path).exists():
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return
        if model_path == "":
            model_path = None

        logger.info(f"Attempting to add player: {name} (Type: {agent_type}, Model: {model_path})")

        try:
            self.league_manager.add_player(name, agent_type, model_path)
            messagebox.showinfo("Success", f"Player '{name}' added successfully!")
            self._populate_player_comboboxes()
            self._populate_train_agent_combobox()
            self.add_player_name_entry.delete(0, tk.END)
            self.add_player_model_path_entry.delete(0, tk.END)
            self.add_player_agent_type_combobox.set("")
        except ValueError as e:
            messagebox.showerror("Error", f"Failed to add player: {e}")
            logger.error(f"Failed to add player {name}: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            logger.exception(f"Unexpected error adding player {name}")

    def _create_league_agent(self):
        name = self.create_agent_name_entry.get().strip()
        agent_type = self.create_agent_agent_type_combobox.get().strip()
        training_steps_str = self.create_agent_training_steps_entry.get().strip()

        if not name or not agent_type:
            messagebox.showerror("Error", "Agent name and type are required.")
            return

        try:
            training_steps = int(training_steps_str)
            if training_steps <= 0:
                messagebox.showerror("Error", "Training steps must be positive.")
                return
        except ValueError:
            messagebox.showerror("Error", "Training steps must be an integer.")
            return

        logger.info(f"Attempting to create initial agent: {name} (Type: {agent_type}, Steps: {training_steps})")

        try:
            model_path = self.league_manager.create_initial_agent(name, agent_type, training_steps)
            messagebox.showinfo("Success", f"Agent '{name}' created and trained successfully! Model saved to: {model_path}")
            self._populate_player_comboboxes()
            self.create_agent_name_entry.delete(0, tk.END)
            self.create_agent_training_steps_entry.delete(0, tk.END)
        except ValueError as e:
            messagebox.showerror("Error", f"Failed to create agent: {e}")
            logger.error(f"Failed to create agent {name}: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            logger.exception(f"Unexpected error creating agent {name}")

    def _on_train_agent_select(self, event: tk.Event):
        selected_player_name = self.train_agent_name_combobox.get()
        if selected_player_name in self.league_manager.players:
            player = self.league_manager.players[selected_player_name]
            self.train_agent_agent_type_label.config(text=player.agent_type)
        else:
            self.train_agent_agent_type_label.config(text="")

    def _train_league_agent(self):
        name = self.train_agent_name_combobox.get().strip()
        timesteps_str = self.train_agent_timesteps_entry.get().strip()
        agent_type = self.train_agent_agent_type_label.cget("text")

        if not name or not agent_type:
            messagebox.showerror("Error", "Agent name and type are required.")
            return

        try:
            timesteps = int(timesteps_str)
            if timesteps <= 0:
                messagebox.showerror("Error", "Training timesteps must be positive.")
                return
        except ValueError:
            messagebox.showerror("Error", "Training timesteps must be an integer.")
            return

        logger.info(f"Attempting to train agent: {name} (Type: {agent_type}, Timesteps: {timesteps})")

        try:
            model_path = self.league_manager.train_agent(name, agent_type, timesteps)
            messagebox.showinfo("Success", f"Agent '{name}' trained successfully! Model saved to: {model_path}")
            self._populate_player_comboboxes()
        except ValueError as e:
            messagebox.showerror("Error", f"Failed to train agent: {e}")
            logger.error(f"Failed to train agent {name}: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            logger.exception(f"Unexpected error training agent {name}")

    def _populate_train_agent_combobox(self):
        try:
            player_names = sorted(list(self.league_manager.players.keys()))
            self.train_agent_name_combobox['values'] = player_names
            if player_names:
                self.train_agent_name_combobox.set(player_names[0])
                self._on_train_agent_select(None)
        except Exception as e:
            logger.error(f"Failed to populate train agent combobox: {e}")
            messagebox.showerror("Error", f"Failed to load players for combobox: {e}")

    def _populate_model_list(self):
        self.model_listbox.delete(0, tk.END)
        model_dir = Path("models") # Assuming models are in a 'models' directory
        if not model_dir.exists():
            model_dir.mkdir()

        # List all files in the models directory and its subdirectories
        all_models = []
        for f in model_dir.rglob("*"):
            if f.is_file() and f.suffix in [".pth", ".zip", ".pkl"]:
                all_models.append(str(f.relative_to(model_dir)))
        
        for model_name in sorted(all_models):
            self.model_listbox.insert(tk.END, model_name)
            self.compare_model_listbox.insert(tk.END, model_name)
        logger.info("Model list populated.")

    def _on_model_select(self, event: tk.Event):
        selected_indices = self.model_listbox.curselection()
        if not selected_indices:
            return

        selected_model_path_str = self.model_listbox.get(selected_indices[0])
        selected_model_path = Path("models") / selected_model_path_str

        self.model_details_text.config(state="normal")
        self.model_details_text.delete("1.0", tk.END)
        
        if selected_model_path.exists():
            # Try to load metadata if available (e.g., training_metadata.json)
            metadata_path = selected_model_path.parent / "training_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    self.model_details_text.insert(tk.END, "Metadata:\n")
                    self.model_details_text.insert(tk.END, json.dumps(metadata, indent=2))
                    self.model_details_text.insert(tk.END, "\n\n")
                except Exception as e:
                    self.model_details_text.insert(tk.END, f"Error loading metadata: {e}\n\n")
            
            self.model_details_text.insert(tk.END, f"File Path: {selected_model_path}\n")
            self.model_details_text.insert(tk.END, f"File Size: {selected_model_path.stat().st_size / (1024*1024):.2f} MB\n")
            self.model_details_text.insert(tk.END, f"Last Modified: {selected_model_path.stat().st_mtime}\n")

        else:
            self.model_details_text.insert(tk.END, "Model file not found.\n")
        
        self.model_details_text.config(state="disabled")
        logger.info(f"Displayed details for model: {selected_model_path_str}")

    def _load_model(self):
        selected_indices = self.model_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "No model selected to load.")
            return

        selected_model_path_str = self.model_listbox.get(selected_indices[0])
        full_model_path = Path("models") / selected_model_path_str

        if not full_model_path.exists():
            messagebox.showerror("Error", f"Model file not found: {full_model_path}")
            return

        # This is a placeholder. Actual model loading depends on the framework (e.g., PyTorch, TensorFlow)
        # For now, we'll just simulate loading and inform the user.
        try:
            # In a real scenario, you would load the model here.
            # e.g., model = torch.load(full_model_path)
            messagebox.showinfo("Load Model", f"Model loaded successfully (simulated): {full_model_path}")
            logger.info(f"Model loaded: {full_model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            logger.exception(f"Failed to load model: {full_model_path}")

    def _delete_model(self):
        selected_indices = self.model_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "No model selected to delete.")
            return

        selected_model_path_str = self.model_listbox.get(selected_indices[0])
        full_model_path = Path("models") / selected_model_path_str

        if not full_model_path.exists():
            messagebox.showerror("Error", f"Model file not found: {full_model_path}")
            return

        if messagebox.askyesno("Delete Model", f"Are you sure you want to delete {selected_model_path_str}?"):
            try:
                # Delete the model file
                full_model_path.unlink()
                # Also delete associated metadata if it exists
                metadata_path = full_model_path.parent / "training_metadata.json"
                if metadata_path.exists():
                    metadata_path.unlink()
                
                messagebox.showinfo("Delete Model", f"Model {selected_model_path_str} deleted successfully.")
                logger.info(f"Model deleted: {selected_model_path_str}")
                self._populate_model_list() # Refresh the list
                self.model_details_text.config(state="normal")
                self.model_details_text.delete("1.0", tk.END)
                self.model_details_text.config(state="disabled")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete model: {e}")
                logger.exception(f"Failed to delete model: {selected_model_path_str}")

    def _compare_models(self):
        selected_indices = self.compare_model_listbox.curselection()
        if len(selected_indices) < 2:
            messagebox.showwarning("Model Comparison", "Please select at least two models for comparison.")
            return

        models_to_compare = [self.compare_model_listbox.get(i) for i in selected_indices]
        comparison_data = []

        for model_name in models_to_compare:
            model_path = Path("models") / model_name
            metadata_path = model_path.parent / "training_metadata.json"
            
            model_info = {"Model": model_name}
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    model_info["Avg Reward"] = f"{metadata.get('avg_reward', 'N/A'):.2f}"
                    model_info["Timesteps"] = metadata.get("timesteps", "N/A")
                    model_info["Episodes"] = metadata.get("episodes", "N/A")
                    # Add more metrics as needed
                except Exception as e:
                    logger.error(f"Error loading metadata for {model_name}: {e}")
                    model_info["Metadata Error"] = "Yes"
            else:
                model_info["Metadata Error"] = "No Metadata"
            comparison_data.append(model_info)

        # Clear previous comparison results
        for item in self.model_comparison_tree.get_children():
            self.model_comparison_tree.delete(item)

        if not comparison_data:
            return

        # Determine columns dynamically from all available keys
        all_keys = set()
        for data in comparison_data:
            all_keys.update(data.keys())
        
        columns = sorted(list(all_keys))
        self.model_comparison_tree["columns"] = columns
        for col in columns:
            self.model_comparison_tree.heading(col, text=col)
            self.model_comparison_tree.column(col, width=100, anchor="center")

        # Insert data
        for data in comparison_data:
            values = [data.get(col, "-") for col in columns]
            self.model_comparison_tree.insert("", "end", values=values)

    def _populate_opt_config_combobox(self):
        all_configs = sorted(list(self.launcher.default_configs.keys()) + [f.stem for f in self.launcher.config_dir.glob("*.json") if f.stem not in self.launcher.default_configs.keys()])
        self.opt_config_combobox['values'] = all_configs
        if all_configs:
            self.opt_config_combobox.set(all_configs[0])

    def _start_hyperparam_optimization(self):
        config_name = self.opt_config_combobox.get()
        num_trials_str = self.num_trials_entry.get().strip()
        hp_text_content = self.hp_text.get("1.0", tk.END).strip()

        if not config_name:
            messagebox.showerror("Error", "Please select a configuration for optimization.")
            return
        try:
            num_trials = int(num_trials_str)
            if num_trials <= 0:
                messagebox.showerror("Error", "Number of trials must be positive.")
                return
        except ValueError:
            messagebox.showerror("Error", "Number of trials must be an integer.")
            return
        
        if not hp_text_content or hp_text_content == "# Example: \nlearning_rate: [loguniform, 1e-5, 1e-1]\nbatch_size: [int, 32, 128]\n":
            messagebox.showerror("Error", "Please define hyperparameters to optimize.")
            return

        # Parse hyperparameters
        hyperparams_to_optimize = {}
        try:
            for line in hp_text_content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, value_str = line.split(": ", 1)
                value_parts = value_str.strip("[]").split(", ")
                param_type = value_parts[0]
                if param_type == "int":
                    hyperparams_to_optimize[key] = {"type": "int", "low": int(value_parts[1]), "high": int(value_parts[2])}
                elif param_type == "float":
                    hyperparams_to_optimize[key] = {"type": "float", "low": float(value_parts[1]), "high": float(value_parts[2])}
                elif param_type == "loguniform":
                    hyperparams_to_optimize[key] = {"type": "loguniform", "low": float(value_parts[1]), "high": float(value_parts[2])}
                elif param_type == "categorical":
                    hyperparams_to_optimize[key] = {"type": "categorical", "choices": [x.strip() for x in value_parts[1:]]}
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
        except Exception as e:
            messagebox.showerror("Error", f"Error parsing hyperparameters: {e}\nExpected format: param_name: [type, min, max] or [type, choice1, choice2]")
            return

        self.start_opt_button.config(state="disabled")
        self.stop_opt_button.config(state="normal")
        self.opt_config_combobox.config(state="disabled")
        self.num_trials_entry.config(state="disabled")
        self.hp_text.config(state="disabled")

        # Clear previous results
        for item in self.opt_results_tree.get_children():
            self.opt_results_tree.delete(item)

        self.optuna_study = optuna.create_study(direction="maximize") # Assuming we want to maximize reward
        self.optimization_thread = threading.Thread(target=self._run_optimization_task, args=(config_name, num_trials, hyperparams_to_optimize))
        self.optimization_thread.start()

    def _run_optimization_task(self, config_name, num_trials, hyperparams_to_optimize):
        def objective(trial):
            # Load the base config
            config_data = self.launcher.load_config(config_name)

            # Suggest hyperparameters from Optuna trial
            trial_params = {}
            for hp_name, hp_info in hyperparams_to_optimize.items():
                if hp_info["type"] == "int":
                    trial_params[hp_name] = trial.suggest_int(hp_name, hp_info["low"], hp_info["high"])
                elif hp_info["type"] == "float":
                    trial_params[hp_name] = trial.suggest_float(hp_name, hp_info["low"], hp_info["high"])
                elif hp_info["type"] == "loguniform":
                    trial_params[hp_name] = trial.suggest_loguniform(hp_name, hp_info["low"], hp_info["high"])
                elif hp_info["type"] == "categorical":
                    trial_params[hp_name] = trial.suggest_categorical(hp_name, hp_info["choices"])
            
            # Apply trial parameters to config_data
            # This part needs to be adapted based on how your config handles nested parameters
            # For simplicity, assuming top-level parameters for now
            updated_config_data = config_data.copy()
            updated_config_data.update(trial_params)

            # Launch training with updated config (simulated for now)
            # In a real scenario, you would save this config to a temp file and launch training
            # and then get the final reward/metric from the training run.
            logger.info(f"Running trial {trial.number} with params: {trial_params}")
            # Simulate training and get a metric (e.g., average reward)
            import time
            time.sleep(1) # Simulate training time
            # This is where you would call your training function and get a metric
            # For demonstration, return a random value
            metric_value = (trial.number % 5) + (sum(trial_params.values()) if trial_params else 0) # Example metric
            
            self.master.after(0, lambda: self._update_optimization_results(trial.number, metric_value, trial_params))
            return metric_value

        try:
            self.optuna_study.optimize(objective, n_trials=num_trials, callbacks=[self._optuna_callback])
            logger.info("Optimization completed.")
            self.master.after(0, lambda: messagebox.showinfo("Optimization", "Hyperparameter optimization finished!"))
        except Exception as e:
            logger.exception("Optimization failed.")
            self.master.after(0, lambda: messagebox.showerror("Optimization", f"Optimization failed: {e}"))
        finally:
            self.master.after(0, self._reset_optimization_ui)

    def _optuna_callback(self, study, trial):
        # This callback is called after each trial completes
        self.master.after(0, lambda: self._update_optimization_results(trial.number, trial.value, trial.params))

    def _update_optimization_results(self, trial_number, value, params):
        self.opt_results_tree.insert("", "end", values=(trial_number, f'{value:.2f}', str(params)))
        self.opt_results_tree.yview_moveto(1) # Scroll to bottom

    def _stop_hyperparam_optimization(self):
        if self.optimization_thread and self.optimization_thread.is_alive():
            # Optuna doesn't have a direct stop method for optimize().
            # You would typically use a custom callback and raise optuna.exceptions.TrialPruned
            # or manage a flag. For simplicity, we'll just inform the user.
            messagebox.showinfo("Stop Optimization", "Optimization stop requested. It will finish the current trial.")
            # In a real scenario, you might set a flag that the objective function checks
            # to gracefully exit.
            logger.info("Optimization stop requested.")
        else:
            messagebox.showinfo("Stop Optimization", "No active optimization to stop.")
        self._reset_optimization_ui()

    def _apply_best_trial_config(self):
        if not hasattr(self, 'optuna_study') or not self.optuna_study.best_trial:
            messagebox.showwarning("Apply Config", "No best trial found from optimization.")
            return
        
        best_trial = self.optuna_study.best_trial
        best_params = best_trial.params

        selected_config_name = self.opt_config_combobox.get()
        if not selected_config_name:
            messagebox.showerror("Error", "Please select a configuration to apply parameters to.")
            return

        # Load the base config
        config_data = self.launcher.load_config(selected_config_name)
        
        # Apply best parameters
        updated_config_data = config_data.copy()
        updated_config_data.update(best_params) # Assuming top-level parameters

        # Save the updated config (or open in editor)
        try:
            config_path = self.launcher.config_dir / f"{selected_config_name}_optimized.json"
            with open(config_path, 'w') as f:
                json.dump(updated_config_data, f, indent=2)
            messagebox.showinfo("Apply Config", f"Best trial parameters applied and saved to {config_path}")
            logger.info(f"Best trial parameters applied to {config_path}")
            self._populate_config_list() # Refresh config list
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply best trial config: {e}")
            logger.exception("Failed to apply best trial config.")

    def _reset_optimization_ui(self):
        self.start_opt_button.config(state="normal")
        self.stop_opt_button.config(state="disabled")
        self.opt_config_combobox.config(state="normal")
        self.num_trials_entry.config(state="normal")
        self.hp_text.config(state="normal")

    def _run_git_command(self, command, message="Executing Git command..."):
        self.git_output_text.config(state="normal")
        self.git_output_text.delete("1.0", tk.END)
        self.git_output_text.insert(tk.END, f"{message}\n")
        self.git_output_text.see(tk.END)
        self.git_output_text.config(state="disabled")

        def run_command():
            try:
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=Path(__file__).parent)
                stdout, stderr = process.communicate()

                self.master.after(0, lambda: self.git_output_text.config(state="normal"))
                self.master.after(0, lambda: self.git_output_text.insert(tk.END, stdout))
                self.master.after(0, lambda: self.git_output_text.insert(tk.END, stderr))
                self.master.after(0, lambda: self.git_output_text.see(tk.END))
                self.master.after(0, lambda: self.git_output_text.config(state="disabled"))

                if process.returncode != 0:
                    self.master.after(0, lambda: messagebox.showerror("Git Error", f"Git command failed with error:\n{stderr}"))
            except Exception as e:
                self.master.after(0, lambda: messagebox.showerror("Error", f"Failed to execute Git command: {e}"))
                logger.exception("Failed to execute Git command.")

        threading.Thread(target=run_command).start()

    def _git_status(self):
        self._run_git_command(["git", "status"], "Getting Git status...")

    def _git_add_all(self):
        self._run_git_command(["git", "add", "."], "Adding all changes...")

    def _git_commit(self):
        commit_message = self.commit_message_entry.get().strip()
        if not commit_message:
            messagebox.showwarning("Git Commit", "Please enter a commit message.")
            return
        self._run_git_command(["git", "commit", "-m", commit_message], f"Committing with message: '{commit_message}'...")
        self.commit_message_entry.delete(0, tk.END)

    def _git_push(self):
        self._run_git_command(["git", "push"], "Pushing changes...")

    def _git_pull(self):
        self._run_git_command(["git", "pull"], "Pulling changes...")

    

    def _setup_config_tab(self):
        config_pane = ttk.PanedWindow(self.config_frame, orient=tk.HORIZONTAL)
        config_pane.pack(expand=True, fill='both', padx=10, pady=10)

        list_frame = ttk.LabelFrame(config_pane, text="Available Configurations")
        config_pane.add(list_frame, weight=1)

        self.config_listbox = tk.Listbox(list_frame, height=20)
        self.config_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.config_listbox.bind("<<ListboxSelect>>", self._on_config_select)

        config_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.config_listbox.yview)
        config_scrollbar.pack(side="right", fill="y")
        self.config_listbox.config(yscrollcommand=config_scrollbar.set)

        details_frame = ttk.LabelFrame(config_pane, text="Configuration Editor")
        config_pane.add(details_frame, weight=2)

        self.config_details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, width=50, height=20)
        self.config_details_text.pack(padx=5, pady=5, fill="both", expand=True)
        self.config_details_text.config(state="normal")

        button_frame = ttk.Frame(details_frame)
        button_frame.pack(side="bottom", fill="x", padx=5, pady=5)

        ttk.Button(button_frame, text="↻ Refresh List", command=self._populate_config_list).pack(side="left", padx=5)
        ttk.Button(button_frame, text="➕ Create New", command=self._create_new_config_dialog).pack(side="left", padx=5)
        ttk.Button(button_frame, text="✔ Save", command=self._save_config).pack(side="right", padx=5)
        ttk.Button(button_frame, text="💾 Save As...", command=self._save_config_as).pack(side="right", padx=5)

        self._populate_config_list()

    def _setup_training_tab(self):
        training_controls_frame = ttk.LabelFrame(self.training_frame, text="Launch Training")
        training_controls_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(training_controls_frame, text="Select Configuration:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.training_config_combobox = ttk.Combobox(training_controls_frame, state="readonly")
        self.training_config_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.training_config_combobox.bind("<<ComboboxSelected>>", self._on_training_config_select)
        self._populate_training_config_combobox()

        ttk.Label(training_controls_frame, text="Resume From (model path):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.resume_from_entry = ttk.Entry(training_controls_frame)
        self.resume_from_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.test_after_var = tk.BooleanVar()
        self.test_after_checkbox = ttk.Checkbutton(training_controls_frame, text="Test After Training", variable=self.test_after_var)
        self.test_after_checkbox.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.render_test_var = tk.BooleanVar()
        self.render_test_checkbox = ttk.Checkbutton(training_controls_frame, text="Render Test", variable=self.render_test_var)
        self.render_test_checkbox.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        self.training_launch_button = ttk.Button(training_controls_frame, text="▶ Launch Training", command=self._launch_training_session)
        self.training_launch_button.grid(row=3, column=0, columnspan=2, pady=10)

        training_controls_frame.grid_columnconfigure(1, weight=1)

        self.training_log_text = scrolledtext.ScrolledText(self.training_frame, wrap=tk.WORD, width=100, height=15)
        self.training_log_text.pack(padx=10, pady=10, fill="both", expand=True)
        self.training_log_text.config(state="disabled")

        # Training Metrics Display
        metrics_frame = ttk.LabelFrame(self.training_frame, text="Training Metrics")
        metrics_frame.pack(padx=10, pady=10, fill="both", expand=True)

        metrics_columns = ("episode", "timesteps", "avg_reward", "loss")
        self.training_metrics_tree = ttk.Treeview(metrics_frame, columns=metrics_columns, show="headings")
        for col in metrics_columns:
            self.training_metrics_tree.heading(col, text=col.replace("_", " ").title())
            self.training_metrics_tree.column(col, width=100, anchor="center")
        self.training_metrics_tree.pack(side="left", fill="both", expand=True)

        metrics_scrollbar = ttk.Scrollbar(metrics_frame, orient="vertical", command=self.training_metrics_tree.yview)
        metrics_scrollbar.pack(side="right", fill="y")
        self.training_metrics_tree.config(yscrollcommand=metrics_scrollbar.set)

        # Matplotlib Plot for Training Metrics
        plot_frame = ttk.LabelFrame(self.training_frame, text="Training Progress Plot")
        plot_frame.pack(padx=10, pady=10, fill="both", expand=True)

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx() # Create a twin Axes for the second y-axis

        self.line1, = self.ax1.plot([], [], label='Avg Reward', color='blue')
        self.line2, = self.ax2.plot([], [], label='Loss', color='red')

        self.ax1.set_xlabel("Episode")
        self.ax1.set_ylabel("Avg Reward", color='blue')
        self.ax2.set_ylabel("Loss", color='red')

        # Combine legends from both axes
        lines, labels = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        self.ax1.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def _setup_league_tab(self):
        # Main layout container
        main_pane = ttk.PanedWindow(self.league_frame, orient=tk.VERTICAL)
        main_pane.pack(expand=True, fill='both', padx=10, pady=10)

        # Top frame for leaderboard and match playing
        top_frame = ttk.Frame(main_pane)
        main_pane.add(top_frame, weight=1)

        # Leaderboard
        leaderboard_frame = ttk.LabelFrame(top_frame, text="Leaderboard")
        leaderboard_frame.pack(padx=10, pady=10, fill='both', expand=True)

        columns = ("rank", "name", "rating", "wins", "losses", "draws")
        self.leaderboard_tree = ttk.Treeview(leaderboard_frame, columns=columns, show="headings")
        for col in columns:
            self.leaderboard_tree.heading(col, text=col.capitalize())
        self.leaderboard_tree.pack(fill="both", expand=True, side='left')
        
        leaderboard_scrollbar = ttk.Scrollbar(leaderboard_frame, orient="vertical", command=self.leaderboard_tree.yview)
        leaderboard_scrollbar.pack(side='right', fill='y')
        self.leaderboard_tree.config(yscrollcommand=leaderboard_scrollbar.set)

        self._update_leaderboard_view()

        # Play Match
        play_match_frame = ttk.LabelFrame(top_frame, text="Play Match")
        play_match_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(play_match_frame, text="Player 1:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.player1_combobox = ttk.Combobox(play_match_frame, state="readonly")
        self.player1_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(play_match_frame, text="Player 2:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.player2_combobox = ttk.Combobox(play_match_frame, state="readonly")
        self.player2_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(play_match_frame, text="Number of Games:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.num_games_entry = ttk.Entry(play_match_frame)
        self.num_games_entry.insert(0, "1")
        self.num_games_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.play_match_button = ttk.Button(play_match_frame, text="▶ Play Match", command=self._play_league_match)
        self.play_match_button.grid(row=3, column=0, columnspan=2, pady=10)
        play_match_frame.grid_columnconfigure(1, weight=1)

        # Agent vs. Agent Simulation
        simulation_frame = ttk.LabelFrame(top_frame, text="Agent vs. Agent Simulation")
        simulation_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(simulation_frame, text="Agent 1:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.sim_agent1_combobox = ttk.Combobox(simulation_frame, state="readonly")
        self.sim_agent1_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(simulation_frame, text="Agent 2:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.sim_agent2_combobox = ttk.Combobox(simulation_frame, state="readonly")
        self.sim_agent2_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(simulation_frame, text="Number of Games:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.sim_num_games_entry = ttk.Entry(simulation_frame)
        self.sim_num_games_entry.insert(0, "1")
        self.sim_num_games_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.start_sim_button = ttk.Button(simulation_frame, text="▶ Start Simulation", command=self._start_simulation)
        self.start_sim_button.grid(row=3, column=0, padx=5, pady=10)
        self.stop_sim_button = ttk.Button(simulation_frame, text="■ Stop Simulation", command=self._stop_simulation, state="disabled")
        self.stop_sim_button.grid(row=3, column=1, padx=5, pady=10)

        simulation_frame.grid_columnconfigure(1, weight=1)

        self.simulation_log_text = scrolledtext.ScrolledText(simulation_frame, wrap=tk.WORD, width=100, height=10)
        self.simulation_log_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.simulation_log_text.config(state="disabled")

        # Bottom frame for agent management
        bottom_frame = ttk.Frame(main_pane)
        main_pane.add(bottom_frame, weight=1)

        # Agent management notebook
        agent_notebook = ttk.Notebook(bottom_frame)
        agent_notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # Add Player Tab
        add_player_frame = ttk.Frame(agent_notebook)
        agent_notebook.add(add_player_frame, text="Add Player")

        ttk.Label(add_player_frame, text="Player Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.add_player_name_entry = ttk.Entry(add_player_frame)
        self.add_player_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(add_player_frame, text="Agent Type:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.add_player_agent_type_combobox = ttk.Combobox(add_player_frame, values=SUPPORTED_AGENT_TYPES, state="readonly")
        self.add_player_agent_type_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        if SUPPORTED_AGENT_TYPES:
            self.add_player_agent_type_combobox.set(SUPPORTED_AGENT_TYPES[0])

        ttk.Label(add_player_frame, text="Model Path (optional):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        model_path_frame = ttk.Frame(add_player_frame)
        model_path_frame.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        model_path_frame.grid_columnconfigure(0, weight=1)
        
        self.add_player_model_path_entry = ttk.Entry(model_path_frame)
        self.add_player_model_path_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        ttk.Button(model_path_frame, text="Browse...", command=self._browse_model_file).grid(row=0, column=1)

        ttk.Button(add_player_frame, text="➕ Add Player", command=self._add_league_player).grid(row=3, column=0, columnspan=2, pady=10)
        add_player_frame.grid_columnconfigure(1, weight=1)

        # Create Initial Agent Tab
        create_agent_frame = ttk.Frame(agent_notebook)
        agent_notebook.add(create_agent_frame, text="Create Initial Agent")

        ttk.Label(create_agent_frame, text="Agent Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.create_agent_name_entry = ttk.Entry(create_agent_frame)
        self.create_agent_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(create_agent_frame, text="Agent Type:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.create_agent_agent_type_combobox = ttk.Combobox(create_agent_frame, values=SUPPORTED_AGENT_TYPES, state="readonly")
        self.create_agent_agent_type_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        if SUPPORTED_AGENT_TYPES:
            self.create_agent_agent_type_combobox.set(SUPPORTED_AGENT_TYPES[0])

        ttk.Label(create_agent_frame, text="Training Steps:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.create_agent_training_steps_entry = ttk.Entry(create_agent_frame)
        self.create_agent_training_steps_entry.insert(0, "10000")
        self.create_agent_training_steps_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        ttk.Button(create_agent_frame, text="✨ Create Agent", command=self._create_league_agent).grid(row=3, column=0, columnspan=2, pady=10)
        create_agent_frame.grid_columnconfigure(1, weight=1)

        # Train Existing Agent Tab
        train_agent_frame = ttk.Frame(agent_notebook)
        agent_notebook.add(train_agent_frame, text="Train Existing Agent")

        ttk.Label(train_agent_frame, text="Agent Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.train_agent_name_combobox = ttk.Combobox(train_agent_frame, state="readonly")
        self.train_agent_name_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.train_agent_name_combobox.bind("<<ComboboxSelected>>", self._on_train_agent_select)

        ttk.Label(train_agent_frame, text="Agent Type:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.train_agent_agent_type_label = ttk.Label(train_agent_frame, text="")
        self.train_agent_agent_type_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(train_agent_frame, text="Training Timesteps:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.train_agent_timesteps_entry = ttk.Entry(train_agent_frame)
        self.train_agent_timesteps_entry.insert(0, "100000")
        self.train_agent_timesteps_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        ttk.Button(train_agent_frame, text="⚒️ Train Agent", command=self._train_league_agent).grid(row=3, column=0, columnspan=2, pady=10)
        train_agent_frame.grid_columnconfigure(1, weight=1)

        self._populate_player_comboboxes()
        self._populate_train_agent_combobox()

    def _setup_model_management_tab(self):
        model_pane = ttk.PanedWindow(self.model_management_frame, orient=tk.HORIZONTAL)
        model_pane.pack(expand=True, fill='both', padx=10, pady=10)

        list_frame = ttk.LabelFrame(model_pane, text="Available Models")
        model_pane.add(list_frame, weight=1)

        self.model_listbox = tk.Listbox(list_frame, height=20)
        self.model_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.model_listbox.bind("<<ListboxSelect>>", self._on_model_select)

        model_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.model_listbox.yview)
        model_scrollbar.pack(side="right", fill="y")
        self.model_listbox.config(yscrollcommand=model_scrollbar.set)

        details_frame = ttk.LabelFrame(model_pane, text="Model Details")
        model_pane.add(details_frame, weight=2)

        self.model_details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, width=50, height=20)
        self.model_details_text.pack(padx=5, pady=5, fill="both", expand=True)
        self.model_details_text.config(state="disabled")

        button_frame = ttk.Frame(details_frame)
        button_frame.pack(side="bottom", fill="x", padx=5, pady=5)

        ttk.Button(button_frame, text="↻ Refresh Models", command=self._populate_model_list).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Load Model", command=self._load_model).pack(side="left", padx=5)
        ttk.Button(button_frame, text="❌ Delete Model", command=self._delete_model).pack(side="right", padx=5)

        # Removed self._populate_model_list() from here

        # Model Comparison Section
        comparison_frame = ttk.LabelFrame(self.model_management_frame, text="Model Comparison")
        comparison_frame.pack(padx=10, pady=10, fill="both", expand=True)

        ttk.Label(comparison_frame, text="Select Models for Comparison:").pack(padx=5, pady=5, anchor="w")
        
        self.compare_model_listbox = tk.Listbox(comparison_frame, selectmode=tk.MULTIPLE, height=5)
        self.compare_model_listbox.pack(padx=5, pady=5, fill="x")
        
        compare_scrollbar = ttk.Scrollbar(comparison_frame, orient="vertical", command=self.compare_model_listbox.yview)
        compare_scrollbar.pack(side="right", fill="y")
        self.compare_model_listbox.config(yscrollcommand=compare_scrollbar.set)

        ttk.Button(comparison_frame, text="Compare Selected Models", command=self._compare_models).pack(pady=10)

        self.model_comparison_tree = ttk.Treeview(comparison_frame, show="headings")
        self.model_comparison_tree.pack(padx=5, pady=5, fill="both", expand=True)

        compare_result_scrollbar = ttk.Scrollbar(comparison_frame, orient="vertical", command=self.model_comparison_tree.yview)
        compare_result_scrollbar.pack(side="right", fill="y")
        self.model_comparison_tree.config(yscrollcommand=compare_result_scrollbar.set)

    def _setup_hyperparam_opt_tab(self):
        opt_controls_frame = ttk.LabelFrame(self.hyperparam_opt_frame, text="Optimization Controls")
        opt_controls_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(opt_controls_frame, text="Config Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.opt_config_combobox = ttk.Combobox(opt_controls_frame, state="readonly")
        self.opt_config_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self._populate_opt_config_combobox()

        ttk.Label(opt_controls_frame, text="Number of Trials:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.num_trials_entry = ttk.Entry(opt_controls_frame)
        self.num_trials_entry.insert(0, "10")
        self.num_trials_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(opt_controls_frame, text="Hyperparameters to Optimize:").grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.hp_text = scrolledtext.ScrolledText(opt_controls_frame, wrap=tk.WORD, width=60, height=10)
        self.hp_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.hp_text.insert(tk.END, "# Example: \nlearning_rate: [loguniform, 1e-5, 1e-1]\nbatch_size: [int, 32, 128]\n")

        self.start_opt_button = ttk.Button(opt_controls_frame, text="▶ Start Optimization", command=self._start_hyperparam_optimization)
        self.start_opt_button.grid(row=4, column=0, padx=5, pady=10)
        self.stop_opt_button = ttk.Button(opt_controls_frame, text="■ Stop Optimization", command=self._stop_hyperparam_optimization, state="disabled")
        self.stop_opt_button.grid(row=4, column=1, padx=5, pady=10)

        opt_controls_frame.grid_columnconfigure(1, weight=1)
        opt_controls_frame.grid_rowconfigure(3, weight=1)

        opt_results_frame = ttk.LabelFrame(self.hyperparam_opt_frame, text="Optimization Results")
        opt_results_frame.pack(padx=10, pady=10, fill="both", expand=True)

        results_columns = ("trial", "value", "params")
        self.opt_results_tree = ttk.Treeview(opt_results_frame, columns=results_columns, show="headings")
        for col in results_columns:
            self.opt_results_tree.heading(col, text=col.capitalize())
            self.opt_results_tree.column(col, width=150, anchor="center")
        self.opt_results_tree.pack(side="left", fill="both", expand=True)

        results_scrollbar = ttk.Scrollbar(opt_results_frame, orient="vertical", command=self.opt_results_tree.yview)
        results_scrollbar.pack(side="right", fill="y")
        self.opt_results_tree.config(yscrollcommand=results_scrollbar.set)

        ttk.Button(opt_results_frame, text="Apply Best Trial Config", command=self._apply_best_trial_config).pack(pady=10)

    def _setup_version_control_tab(self):
        vc_frame = ttk.LabelFrame(self.version_control_frame, text="Git Operations")
        vc_frame.pack(padx=10, pady=10, fill="both", expand=True)

        button_row1 = ttk.Frame(vc_frame)
        button_row1.pack(fill="x", pady=5)
        ttk.Button(button_row1, text="Git Status", command=self._git_status).pack(side="left", padx=5)
        ttk.Button(button_row1, text="Git Add All", command=self._git_add_all).pack(side="left", padx=5)
        ttk.Button(button_row1, text="Git Pull", command=self._git_pull).pack(side="left", padx=5)
        ttk.Button(button_row1, text="Git Push", command=self._git_push).pack(side="left", padx=5)

        ttk.Label(vc_frame, text="Commit Message:").pack(padx=5, pady=5, anchor="w")
        self.commit_message_entry = ttk.Entry(vc_frame)
        self.commit_message_entry.pack(padx=5, pady=5, fill="x")

        ttk.Button(vc_frame, text="Git Commit", command=self._git_commit).pack(padx=5, pady=10)

        ttk.Label(vc_frame, text="Git Output:").pack(padx=5, pady=5, anchor="w")
        self.git_output_text = scrolledtext.ScrolledText(vc_frame, wrap=tk.WORD, width=80, height=20)
        self.git_output_text.pack(padx=5, pady=5, fill="both", expand=True)
        self.git_output_text.config(state="disabled")

    def _setup_settings_tab(self):
        settings_frame = ttk.LabelFrame(self.settings_frame, text="Theme Settings")
        settings_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(settings_frame, text="Select Theme:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.theme_combobox = ttk.Combobox(settings_frame, state="readonly")
        self.theme_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.theme_combobox.bind("<<ComboboxSelected>>", self._apply_theme)

        settings_frame.grid_columnconfigure(1, weight=1)

        self._populate_themes_combobox()

    def _setup_logs_tab(self):
        log_controls_frame = ttk.Frame(self.logs_frame)
        log_controls_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(log_controls_frame, text="❌ Clear Logs", command=self._clear_logs).pack(side="left", padx=5)
        ttk.Button(log_controls_frame, text="💾 Save Logs", command=self._save_logs).pack(side="left", padx=5)
        
        self.logs_text = scrolledtext.ScrolledText(self.logs_frame, wrap=tk.WORD, width=100, height=30)
        self.logs_text.pack(padx=5, pady=5, fill="both", expand=True)
        self.logs_text.config(state="normal")
    
    def _clear_logs(self):
        self.logs_text.delete(1.0, tk.END)
    
    def _save_logs(self):
        try:
            filename = filedialog.asksaveasfilename(title="Save Logs",defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.logs_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Logs saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save logs: {str(e)}")

if __name__ == "__main__":
    root = ThemedTk(theme="arc")
    app = RLGymGUI(root)
    root.mainloop()
