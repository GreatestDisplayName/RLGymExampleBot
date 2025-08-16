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
from src.gui_tabs import DashboardTab, ConfigTab, TrainingTab, VisualizationTab, LeagueTab, ModelManagementTab, HyperparamOptTab, VersionControlTab, HelpTab

import src.gui_tabs
print(f"Imported gui_tabs from: {src.gui_tabs.__file__}")
import json
import csv
import logging
import queue
import threading
import os
from logging.handlers import QueueHandler, QueueListener
from concurrent.futures import ThreadPoolExecutor
from jsonschema import validate, ValidationError
import optuna
import subprocess
import numpy as np
import json
from typing import Optional

THEME_SETTINGS_FILE = "theme_settings.json"
WINDOW_SETTINGS_FILE = "window_settings.json"

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
        
        # Define color tags for different log levels
        self.level_colors = {
            logging.DEBUG: "gray",
            logging.INFO: "black",
            logging.WARNING: "orange",
            logging.ERROR: "red",
            logging.CRITICAL: "purple"
        }
        
        # Configure tags in the text widget
        for level, color in self.level_colors.items():
            self.text_widget.tag_config(logging.getLevelName(level), foreground=color)

    def emit(self, record):
        msg = self.format(record)
        level_name = logging.getLevelName(record.levelno)
        self.text_widget.after(0, lambda: self._update_text(msg, level_name))
    
    def _update_text(self, msg, level_name):
        self.text_widget.insert(tk.END, msg + '\n', level_name)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()


class TrainingMetricsHandler(logging.Handler):
    def __init__(self, tree_widget, episodes_list, rewards_list, losses_list, ax1, ax2, canvas, realtime_metrics=None):
        super().__init__()
        self.tree_widget = tree_widget
        self.episodes_list = episodes_list
        self.rewards_list = rewards_list
        self.losses_list = losses_list
        self.ax1 = ax1
        self.ax2 = ax2
        self.canvas = canvas
        self.realtime_metrics = realtime_metrics
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record):
        msg = self.format(record)
        # Example log format: "Episode: 100, Timesteps: 10000, Avg Reward: 50.2, Loss: 0.123"
        if "Episode:" in msg and "Timesteps:" in msg and "Avg Reward:" in msg:
            try:
                parts = msg.split(", ")
                episode = int(parts[0].split(": ")[1])
                timesteps = int(parts[1].split(": ")[1])
                avg_reward = float(parts[2].split(": ")[1])
                loss = float(parts[3].split(": ")[1]) if len(parts) > 3 and "Loss:" in parts[3] else 0.0
                
                # Update tree widget
                self.tree_widget.after(0, lambda: self.tree_widget.insert("", "end", values=(episode, timesteps, f'{avg_reward:.2f}', f'{loss:.4f}')))
                self.tree_widget.after(0, lambda: self.tree_widget.yview_moveto(1))

                # Update plot data
                self.episodes_list.append(episode)
                self.rewards_list.append(avg_reward)
                self.losses_list.append(loss)

                # Update real-time metrics if provided
                if self.realtime_metrics:
                    self.realtime_metrics['reward'].after(0, lambda: self.realtime_metrics['reward'].config(text=f"{avg_reward:.2f}"))
                    self.realtime_metrics['episode'].after(0, lambda: self.realtime_metrics['episode'].config(text=str(episode)))
                    self.realtime_metrics['timesteps'].after(0, lambda: self.realtime_metrics['timesteps'].config(text=str(timesteps)))
                    self.realtime_metrics['loss'].after(0, lambda: self.realtime_metrics['loss'].config(text=f"{loss:.4f}"))

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
                print(f"Error parsing training log: {msg} - {e}")

class CombinedRLGymGUI:
    def _set_status(self, message):
        if self.status_bar:
            self.status_bar.config(text=message)

    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("RLGym Unified Training Suite")
        master.geometry("1400x900")
        master.minsize(1000, 700)
        
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.launcher = TrainingLauncher()
        self.league_manager = SelfPlayLeague()
        self.running_tasks = set()
        self.log_queue = queue.SimpleQueue()
        self.queue_handler = QueueHandler(self.log_queue)

        # Placeholder for status bar updates
        self.status_bar = None

        # For training plot and metrics
        self.training_episodes = []
        self.training_avg_rewards = []
        self.training_losses = []
        self.realtime_metrics = {}

        # For simulation control
        self.running_simulation = False
        self.simulation_thread = None

        # For hyperparameter optimization
        self.optuna_study = None
        self.optimization_thread = None
        self.leaderboard_sort_col = None
        self.leaderboard_sort_rev = False

        # Main layout with resizable panes for notebook and logs
        self.main_pane = ttk.PanedWindow(master, orient=tk.VERTICAL)
        self.main_pane.pack(expand=True, fill='both')

        self.notebook = ttk.Notebook(self.main_pane)
        self.main_pane.add(self.notebook, weight=3)  # Notebook gets more space initially

        # Create and setup all tabs using new modular classes
        self.dashboard_tab = DashboardTab(self.notebook, self.main_pane, self)
        self.config_tab = ConfigTab(self.notebook, self.main_pane, self)
        self.training_tab = TrainingTab(self.notebook, self.main_pane, self, self.launcher, self.thread_pool, self.training_episodes, self.training_avg_rewards, self.training_losses, self.realtime_metrics)
        self.visualization_tab = VisualizationTab(self.notebook, self.main_pane, self)
        self.league_tab = LeagueTab(self.notebook, self.main_pane, self, self.league_manager, self.thread_pool)
        self.model_management_tab = ModelManagementTab(self.notebook, self.main_pane, self)
        self.hyperparam_opt_tab = HyperparamOptTab(self.notebook, self.main_pane, self, self.thread_pool)
        self.version_control_tab = VersionControlTab(self.notebook, self.main_pane, self)
        self.help_tab = HelpTab(self.notebook, self.main_pane, self)

        # Add tabs in logical order
        self.notebook.add(self.dashboard_tab.frame, text="🏠 Dashboard")
        self.notebook.add(self.config_tab.frame, text="⚙️ Configurations")
        self.notebook.add(self.training_tab.frame, text="🎯 Training")
        self.notebook.add(self.visualization_tab.frame, text="📊 Visualization")
        self.notebook.add(self.league_tab.frame, text="🏆 League")
        self.notebook.add(self.model_management_tab.frame, text="🤖 Model Management")
        self.notebook.add(self.hyperparam_opt_tab.frame, text="🔧 Hyperparameter Optimization")
        self.notebook.add(self.version_control_tab.frame, text="📚 Version Control")
        self.notebook.add(self.help_tab.frame, text="❓ Help")

        # Initial population for some elements that depend on main_app methods
        self.dashboard_tab.populate_model_selector(sorted(list(self.launcher.default_configs.keys()) + [f.stem for f in self.launcher.config_dir.glob("*.json") if f.stem not in self.launcher.default_configs.keys()]))
        self.config_tab._populate_config_list() # Call internal method to populate its list
        self.training_tab._populate_training_config_combobox() # Call internal method to populate its list
        self.league_tab._update_leaderboard_view() # Call internal method to populate its list
        self.model_management_tab._populate_model_list() # Call internal method to populate its list

        self._setup_global_log_view()
        self._setup_logging()
        self._setup_status_bar()

    

        # Start real-time update timer
        self._start_realtime_updates()

        # Restore window geometry
        self._restore_window_geometry()

        # Save geometry on close
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

        logger.info("Combined RLGym GUI initialized.")

        # Load and apply theme setting at startup
        saved_theme = self._load_theme_setting() or "arc"
        self._apply_theme(saved_theme)
        self.theme_combobox.set(saved_theme)

    

    

    
        

    

    

    

    

    

    def _setup_global_log_view(self):
        """Create and configure the global log display area."""
        log_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(log_frame, weight=1)  # Give logs less space initially
        
        # Create scrolled text widget for logs
        self.logs_text = scrolledtext.ScrolledText(
            log_frame, 
            wrap=tk.WORD,
            font=('Consolas', 10),
            state='disabled'
        )
        self.logs_text.pack(expand=True, fill='both', padx=5, pady=5)

    def _update_leaderboard_view(self):
        self.league_tab._update_leaderboard_view()

    def _add_player_dialog(self):
        self.league_tab._add_player_dialog()

    def _remove_selected_player(self):
        self.league_tab._remove_selected_player()

    def _export_league_csv(self):
        self.league_tab._export_league_csv()

    def _import_league_csv(self):
        self.league_tab._import_league_csv()

    def _populate_player_comboboxes(self):
        self.league_tab._populate_player_comboboxes()

    def _play_league_match(self):
        self.league_tab._play_league_match()

    def _simulate_agents(self):
        self.league_tab._simulate_agents()

    def _setup_logging(self):
        text_handler = ThreadSafeTextHandler(self.logs_text)
        text_handler.setLevel(logging.INFO)
        self.queue_listener = QueueListener(self.log_queue, text_handler)
        self.queue_listener.start()
        logger.logger.addHandler(self.queue_handler)

    def _setup_status_bar(self):
        self.status_bar = ttk.Label(self.master, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _load_theme_setting(self):
        theme_settings_path = Path(THEME_SETTINGS_FILE)
        if theme_settings_path.exists():
            try:
                with open(theme_settings_path, 'r') as f:
                    settings = json.load(f)
                    return settings.get("theme")
            except Exception as e:
                logger.error(f"Failed to load theme setting: {e}")
        return None

    def _save_theme_setting(self, theme_name):
        try:
            with open(THEME_SETTINGS_FILE, 'w') as f:
                json.dump({"theme": theme_name}, f)
        except Exception as e:
            logger.error(f"Failed to save theme setting: {e}")

    def _log_activity(self, message):
        self.dashboard_tab._log_activity(message)
    

    

    

    

    

    

    

    

    

    





        dialog = tk.Toplevel(self.master)
        dialog.title("Compare Models")
        dialog.geometry("800x600")

        # Model selection
        selection_frame = ttk.Frame(dialog)
        selection_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(selection_frame, text="Model 1:").grid(row=0, column=0, padx=5, pady=5)
        model1_combo = ttk.Combobox(selection_frame, values=sorted(self.model_listbox.get(0, tk.END)))
        model1_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(selection_frame, text="Model 2:").grid(row=1, column=0, padx=5, pady=5)
        model2_combo = ttk.Combobox(selection_frame, values=sorted(self.model_listbox.get(0, tk.END)))
        model2_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        selection_frame.grid_columnconfigure(1, weight=1)

        # Comparison display
        comparison_frame = ttk.LabelFrame(dialog, text="Comparison")
        comparison_frame.pack(pady=10, padx=10, fill="both", expand=True)

        comparison_text = scrolledtext.ScrolledText(comparison_frame, wrap=tk.WORD)
        comparison_text.pack(fill="both", expand=True, padx=5, pady=5)
        comparison_text.config(state="disabled")

        def do_comparison():
            model1 = model1_combo.get()
            model2 = model2_combo.get()

            if not model1 or not model2:
                messagebox.showerror("Error", "Please select two models to compare.")
                return

            # This is a placeholder for actual comparison logic.
            # In a real application, you would load the models and their metrics.
            comparison_results = f"Comparing {model1} and {model2}:\n\n"
            comparison_results += f"--- {model1} ---\n"
            comparison_results += f"Win Rate: 60%\n"
            comparison_results += f"Avg Reward: 1500\n"
            comparison_results += f"Loss: 0.1\n\n"
            comparison_results += f"--- {model2} ---\n"
            comparison_results += f"Win Rate: 55%\n"
            comparison_results += f"Avg Reward: 1400\n"
            comparison_results += f"Loss: 0.12\n"

            comparison_text.config(state="normal")
            comparison_text.delete("1.0", tk.END)
            comparison_text.insert(tk.END, comparison_results)
            comparison_text.config(state="disabled")

        ttk.Button(dialog, text="Compare", command=do_comparison).pack(pady=10)





    def _start_optimization_thread(self):
        agent_type = self.opt_agent_type_combobox.get()
        n_trials_str = self.opt_n_trials_entry.get().strip()
        study_name = self.opt_study_name_entry.get().strip()

        if not agent_type:
            messagebox.showerror("Error", "Please select an Agent Type.")
            return
        try:
            n_trials = int(n_trials_str)
            if n_trials <= 0:
                raise ValueError("Number of trials must be positive.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid number of trials: {e}")
            return

        if not study_name:
            study_name = None # Optuna will generate a default name

        self._set_status(f"Starting hyperparameter optimization for {agent_type}...")
        self._log_activity(f"Starting hyperparameter optimization for {agent_type} with {n_trials} trials.")
        self.start_opt_button.config(state="disabled")
        self.opt_results_text.config(state="normal")
        self.opt_results_text.delete("1.0", tk.END)
        self.opt_results_text.insert(tk.END, "Optimization started. Please wait...\n")
        self.opt_results_text.config(state="disabled")

        # Run optimization in a separate thread
        self.optimization_thread = threading.Thread(
            target=self._run_optimization_task,
            args=(agent_type, n_trials, study_name)
        )
        self.optimization_thread.start()


    def _update_optimization_results(self, text: str):
        self.opt_results_text.config(state="normal")
        self.opt_results_text.delete("1.0", tk.END)
        self.opt_results_text.insert(tk.END, text)
        self.opt_results_text.config(state="disabled")

        dialog = tk.Toplevel(self.master)
        dialog.title("Compare Models")
        dialog.geometry("800x600")

        # Model selection
        selection_frame = ttk.Frame(dialog)
        selection_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(selection_frame, text="Model 1:").grid(row=0, column=0, padx=5, pady=5)
        model1_combo = ttk.Combobox(selection_frame, values=sorted(self.model_listbox.get(0, tk.END)))
        model1_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(selection_frame, text="Model 2:").grid(row=1, column=0, padx=5, pady=5)
        model2_combo = ttk.Combobox(selection_frame, values=sorted(self.model_listbox.get(0, tk.END)))
        model2_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        selection_frame.grid_columnconfigure(1, weight=1)

        # Comparison display
        comparison_frame = ttk.LabelFrame(dialog, text="Comparison")
        comparison_frame.pack(pady=10, padx=10, fill="both", expand=True)

        comparison_text = scrolledtext.ScrolledText(comparison_frame, wrap=tk.WORD)
        comparison_text.pack(fill="both", expand=True, padx=5, pady=5)
        comparison_text.config(state="disabled")

        def do_comparison():
            model1 = model1_combo.get()
            model2 = model2_combo.get()

            if not model1 or not model2:
                messagebox.showerror("Error", "Please select two models to compare.")
                return

            # This is a placeholder for actual comparison logic.
            # In a real application, you would load the models and their metrics.
            comparison_results = f"Comparing {model1} and {model2}:\n\n"
            comparison_results += f"--- {model1} ---\n"
            comparison_results += f"Win Rate: 60%\n"
            comparison_results += f"Avg Reward: 1500\n"
            comparison_results += f"Loss: 0.1\n\n"
            comparison_results += f"--- {model2} ---\n"
            comparison_results += f"Win Rate: 55%\n"
            comparison_results += f"Avg Reward: 1400\n"
            comparison_results += f"Loss: 0.12\n"

            comparison_text.config(state="normal")
            comparison_text.delete("1.0", tk.END)
            comparison_text.insert(tk.END, comparison_results)
            comparison_text.config(state="disabled")

        ttk.Button(dialog, text="Compare", command=do_comparison).pack(pady=10)

    def _start_optimization_thread(self):
        agent_type = self.opt_agent_type_combobox.get()
        n_trials_str = self.opt_n_trials_entry.get().strip()
        study_name = self.opt_study_name_entry.get().strip()

        if not agent_type:
            messagebox.showerror("Error", "Please select an Agent Type.")
            return
        try:
            n_trials = int(n_trials_str)
            if n_trials <= 0:
                raise ValueError("Number of trials must be positive.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid number of trials: {e}")
            return

        if not study_name:
            study_name = None # Optuna will generate a default name

        self._set_status(f"Starting hyperparameter optimization for {agent_type}.")
        self._log_activity(f"Starting hyperparameter optimization for {agent_type} with {n_trials} trials.")
        self.start_opt_button.config(state="disabled")
        self.opt_results_text.config(state="normal")
        self.opt_results_text.delete("1.0", tk.END)
        self.opt_results_text.insert(tk.END, "Optimization started. Please wait...\n")
        self.opt_results_text.config(state="disabled")

        # Run optimization in a separate thread
        self.optimization_thread = threading.Thread(
            target=self._run_optimization_task,
            args=(agent_type, n_trials, study_name)
        )
        self.optimization_thread.start()


    def _update_optimization_results(self, text: str):
        self.opt_results_text.config(state="normal")
        self.opt_results_text.delete("1.0", tk.END)
        self.opt_results_text.insert(tk.END, text)
        self.opt_results_text.config(state="disabled")

    def _compare_models(self):
        dialog = tk.Toplevel(self.master)
        dialog.title("Compare Models")
        dialog.geometry("800x600")

        # Model selection
        selection_frame = ttk.Frame(dialog)
        selection_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(selection_frame, text="Model 1:").grid(row=0, column=0, padx=5, pady=5)
        model1_combo = ttk.Combobox(selection_frame, values=sorted(self.model_listbox.get(0, tk.END)))
        model1_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(selection_frame, text="Model 2:").grid(row=1, column=0, padx=5, pady=5)
        model2_combo = ttk.Combobox(selection_frame, values=sorted(self.model_listbox.get(0, tk.END)))
        model2_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        selection_frame.grid_columnconfigure(1, weight=1)

        # Comparison display
        comparison_frame = ttk.LabelFrame(dialog, text="Comparison")
        comparison_frame.pack(pady=10, padx=10, fill="both", expand=True)

        comparison_text = scrolledtext.ScrolledText(comparison_frame, wrap=tk.WORD)
        comparison_text.pack(fill="both", expand=True, padx=5, pady=5)
        comparison_text.config(state="disabled")

        def do_comparison():
            model1 = model1_combo.get()
            model2 = model2_combo.get()

            if not model1 or not model2:
                messagebox.showerror("Error", "Please select two models to compare.")
                return

            # This is a placeholder for actual comparison logic.
            # In a real application, you would load the models and their metrics.
            comparison_results = f"Comparing {model1} and {model2}:\n\n"
            comparison_results += f"--- {model1} ---\n"
            comparison_results += f"Win Rate: 60%\n"
            comparison_results += f"Avg Reward: 1500\n"
            comparison_results += f"Loss: 0.1\n\n"
            comparison_results += f"--- {model2} ---\n"
            comparison_results += f"Win Rate: 55%\n"
            comparison_results += f"Avg Reward: 1400\n"
            comparison_results += f"Loss: 0.12\n"

            comparison_text.config(state="normal")
            comparison_text.delete("1.0", tk.END)
            comparison_text.insert(tk.END, comparison_results)
            comparison_text.config(state="disabled")

        ttk.Button(dialog, text="Compare", command=do_comparison).pack(pady=10)

    def _start_optimization_thread(self):
        agent_type = self.opt_agent_type_combobox.get()
        n_trials_str = self.opt_n_trials_entry.get().strip()
        study_name = self.opt_study_name_entry.get().strip()

        if not agent_type:
            messagebox.showerror("Error", "Please select an Agent Type.")
            return
        try:
            n_trials = int(n_trials_str)
            if n_trials <= 0:
                raise ValueError("Number of trials must be positive.")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid number of trials: {e}")
            return

        if not study_name:
            study_name = None # Optuna will generate a default name

        self._set_status(f"Starting hyperparameter optimization for {agent_type}...")
        self._log_activity(f"Starting hyperparameter optimization for {agent_type} with {n_trials} trials.")
        self.start_opt_button.config(state="disabled")
        self.opt_results_text.config(state="normal")
        self.opt_results_text.delete("1.0", tk.END)
        self.opt_results_text.insert(tk.END, "Optimization started. Please wait...\n")
        self.opt_results_text.config(state="disabled")

        # Run optimization in a separate thread
        self.optimization_thread = threading.Thread(
            target=self._run_optimization_task,
            args=(agent_type, n_trials, study_name)
        )
        self.optimization_thread.start()


    def _update_optimization_results(self, text: str):
        self.opt_results_text.config(state="normal")
        self.opt_results_text.delete("1.0", tk.END)
        self.opt_results_text.insert(tk.END, text)
        self.opt_results_text.config(state="disabled")

    

    

    

    

    



    

    

    

    def __del__(self):
        if hasattr(self, 'queue_listener'):
            self.queue_listener.stop()
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)

if __name__ == "__main__":
    root = ThemedTk()
    app = CombinedRLGymGUI(root)
    root.mainloop()
