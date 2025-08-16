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

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.config_frame = ttk.Frame(self.notebook)
        self.training_frame = ttk.Frame(self.notebook)
        self.league_frame = ttk.Frame(self.notebook)
        self.logs_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.config_frame, text="Configurations")
        self.notebook.add(self.training_frame, text="Training")
        self.notebook.add(self.league_frame, text="League")
        self.notebook.add(self.logs_frame, text="Logs")

        self._setup_config_tab()
        self._setup_training_tab()
        self._setup_league_tab()
        self._setup_logs_tab()
        self._setup_logging()

        logger.info("RLGym GUI initialized.")
    
    def _setup_logging(self):
        text_handler = ThreadSafeTextHandler(self.logs_text)
        text_handler.setLevel(logging.INFO)
        self.queue_listener = QueueListener(self.log_queue, text_handler)
        self.queue_listener.start()
        logger.logger.addHandler(self.queue_handler)
    
    def __del__(self):
        if hasattr(self, 'queue_listener'):
            self.queue_listener.stop()
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)

    def _setup_config_tab(self):
        list_frame = ttk.LabelFrame(self.config_frame, text="Available Configurations")
        list_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.config_listbox = tk.Listbox(list_frame, height=20)
        self.config_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.config_listbox.bind("<<ListboxSelect>>", self._on_config_select)

        config_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.config_listbox.yview)
        config_scrollbar.pack(side="right", fill="y")
        self.config_listbox.config(yscrollcommand=config_scrollbar.set)

        details_frame = ttk.LabelFrame(self.config_frame, text="Configuration Details")
        details_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.config_details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, width=50, height=20)
        self.config_details_text.pack(padx=5, pady=5, fill="both", expand=True)
        self.config_details_text.config(state="disabled")

        button_frame = ttk.Frame(self.config_frame)
        button_frame.pack(side="bottom", fill="x", padx=10, pady=5)

        ttk.Button(button_frame, text="Refresh", command=self._populate_config_list).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Create New Config", command=self._create_new_config_dialog).pack(side="left", padx=5)

        self._populate_config_list()

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
            self.config_details_text.config(state="disabled")
            logger.info(f"Displayed details for config: {selected_config_name}")
        except FileNotFoundError:
            messagebox.showerror("Error", f"Configuration '{selected_config_name}' not found.")
            logger.error(f"Config file not found: {selected_config_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config '{selected_config_name}': {e}")
            logger.exception(f"Failed to load config: {selected_config_name}")

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

        ttk.Label(dialog, text="Timesteps:").pack(pady=5)
        timesteps_entry = ttk.Entry(dialog)
        timesteps_entry.pack(pady=5)

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

        success = self.launcher.launch_training(config_name=config_name, resume_from=resume_from, test_after=test_after, render_test=render_test)

        if success:
            messagebox.showinfo("Training Status", "Training session launched successfully!")
        else:
            messagebox.showerror("Training Status", "Failed to launch training session. Check logs for details.")

    def _populate_player_comboboxes(self):
        try:
            player_names = sorted(list(self.league_manager.players.keys()))
            self.player1_combobox['values'] = player_names
            self.player2_combobox['values'] = player_names
            if player_names:
                self.player1_combobox.set(player_names[0])
                if len(player_names) > 1:
                    self.player2_combobox.set(player_names[1])
        except Exception as e:
            logger.error(f"Failed to populate player comboboxes: {e}")
            messagebox.showerror("Error", f"Failed to load players for comboboxes: {e}")

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
    
    def _update_leaderboard_view(self):
        for i in self.leaderboard_tree.get_children():
            self.leaderboard_tree.delete(i)
        
        leaderboard = self.league_manager.get_leaderboard()
        for i, (name, data) in enumerate(leaderboard.iterrows()):
            self.leaderboard_tree.insert("", "end", values=(i + 1, name, data["rating"], data["wins"], data["losses"], data["draws"]))

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

        ttk.Button(training_controls_frame, text="Launch Training", command=self._launch_training_session).grid(row=3, column=0, columnspan=2, pady=10)

        training_controls_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.training_frame, text="Training progress and monitoring will appear here.").pack(padx=10, pady=10, fill="both", expand=True)

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

        self.play_match_button = ttk.Button(play_match_frame, text="Play Match", command=self._play_league_match)
        self.play_match_button.grid(row=3, column=0, columnspan=2, pady=10)
        play_match_frame.grid_columnconfigure(1, weight=1)

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

        ttk.Button(add_player_frame, text="Add Player", command=self._add_league_player).grid(row=3, column=0, columnspan=2, pady=10)
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

        ttk.Button(create_agent_frame, text="Create Agent", command=self._create_league_agent).grid(row=3, column=0, columnspan=2, pady=10)
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

        ttk.Button(train_agent_frame, text="Train Agent", command=self._train_league_agent).grid(row=3, column=0, columnspan=2, pady=10)
        train_agent_frame.grid_columnconfigure(1, weight=1)

        self._populate_player_comboboxes()
        self._populate_train_agent_combobox()

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

    def _setup_logs_tab(self):
        log_controls_frame = ttk.Frame(self.logs_frame)
        log_controls_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(log_controls_frame, text="Clear Logs", command=self._clear_logs).pack(side="left", padx=5)
        ttk.Button(log_controls_frame, text="Save Logs", command=self._save_logs).pack(side="left", padx=5)
        
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