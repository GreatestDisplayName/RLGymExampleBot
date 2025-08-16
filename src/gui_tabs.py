import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path
import logging
import datetime

# Assuming logger is configured elsewhere and imported
logger = logging.getLogger(__name__)

class BaseTab:
    def __init__(self, notebook, parent_frame, title, main_app_instance):
        self.notebook = notebook
        self.parent_frame = parent_frame
        self.title = title
        self.main_app = main_app_instance
        self.frame = ttk.Frame(parent_frame)
        self.notebook.add(self.frame, text=title)

    def setup_ui(self):
        # This method should be overridden by subclasses to set up their specific UI
        raise NotImplementedError

class DashboardTab(BaseTab):
    def __init__(self, notebook, parent_frame, main_app_instance):
        super().__init__(notebook, parent_frame, "🏠 Dashboard", main_app_instance)
        self.main_app = main_app_instance # Reference to the main CombinedRLGymGUI instance
        self.realtime_metrics = {}
        self.activity_text = None # Will be set up in setup_ui
        self.theme_combobox = None # Will be set up in setup_ui
        self.dashboard_model_selector = None # Will be set up in setup_ui
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Real-time metrics section
        metrics_frame = ttk.LabelFrame(main_frame, text="🔴 Real-time Training Metrics")
        metrics_frame.pack(fill="x", pady=(0, 10))

        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(padx=10, pady=10)

        # Create metric displays in a grid
        metrics = [
            ("Current Reward:", "reward", "0.0"),
            ("Episode:", "episode", "0"),
            ("Timesteps:", "timesteps", "0"),
            ("Loss:", "loss", "0.0"),
            ("Learning Rate:", "learning_rate", "0.001"),
            ("Epsilon:", "epsilon", "1.0"),
            ("Batch Size:", "batch_size", "32"),
            ("Training Status:", "status", "Idle")
        ]

        for i, (label_text, key, default_value) in enumerate(metrics):
            row, col = i // 4, (i % 4) * 2
            ttk.Label(metrics_grid, text=label_text).grid(row=row, column=col, padx=5, pady=2, sticky="w")
            value_label = ttk.Label(metrics_grid, text=default_value, font=("Arial", 10, "bold"))
            value_label.grid(row=row, column=col+1, padx=5, pady=2, sticky="w")
            self.realtime_metrics[key] = value_label

        # Quick actions section
        actions_frame = ttk.LabelFrame(main_frame, text="⚡ Quick Actions")
        actions_frame.pack(fill="x", pady=(0, 10))

        actions_grid = ttk.Frame(actions_frame)
        actions_grid.pack(padx=10, pady=10)

        # Action buttons
        ttk.Button(actions_grid, text="🚀 Quick Train", command=self._quick_train).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(actions_grid, text="🔍 Load Model", command=self._quick_load_model).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(actions_grid, text="💾 Save Model", command=self._quick_save_model).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(actions_grid, text="📊 View Progress", command=lambda: self.main_app.notebook.select(2)).grid(row=0, column=3, padx=5, pady=5)

        # Model selector
        model_frame = ttk.LabelFrame(main_frame, text="🤖 Active Model")
        model_frame.pack(fill="x", pady=(0, 10))

        model_controls = ttk.Frame(model_frame)
        model_controls.pack(padx=10, pady=10, fill="x")

        ttk.Label(model_controls, text="Current Model:").pack(side="left")
        self.dashboard_model_selector = ttk.Combobox(model_controls, state="readonly")
        self.dashboard_model_selector.pack(side="left", padx=10, fill="x", expand=True)

        # Recent activity section
        activity_frame = ttk.LabelFrame(main_frame, text="📈 Recent Activity")
        activity_frame.pack(fill="both", expand=True)

        # Activity log with limited size
        self.activity_text = scrolledtext.ScrolledText(activity_frame, wrap=tk.WORD, height=10)
        self.activity_text.pack(padx=10, pady=10, fill="both", expand=True)
        self.activity_text.config(state="disabled")

        # Theme Settings
        theme_frame = ttk.LabelFrame(main_frame, text="🎨 Theme Settings")
        theme_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(theme_frame, text="Select Theme:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.theme_combobox = ttk.Combobox(theme_frame, state="readonly")
        self.theme_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        available_themes = self.main_app.master.get_themes()
        self.theme_combobox['values'] = sorted(available_themes)
        
        self.theme_combobox.bind("<<ComboboxSelected>>", lambda event: self.main_app._apply_theme(self.theme_combobox.get()))

        ttk.Button(theme_frame, text="Toggle Dark Mode", command=self.main_app._toggle_dark_mode).grid(row=0, column=2, padx=5, pady=5)

        theme_frame.grid_columnconfigure(1, weight=1)

    def _quick_train(self):
        self.main_app._quick_train()

    def _quick_load_model(self):
        self.main_app._quick_load_model()

    def _quick_save_model(self):
        self.main_app._quick_save_model()

    def _log_activity(self, message):
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.activity_text.config(state="normal")
        self.activity_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.activity_text.see(tk.END)
        
        # Keep only last 100 lines
        lines = self.activity_text.get("1.0", tk.END).split('\n')
        if len(lines) > 100:
            self.activity_text.delete("1.0", f"{len(lines)-100}.0")
        
        self.activity_text.config(state="disabled")

    def update_realtime_metrics(self, metrics_data):
        for key, value in metrics_data.items():
            if key in self.realtime_metrics:
                self.realtime_metrics[key].config(text=str(value))

    def populate_model_selector(self, models):
        self.dashboard_model_selector['values'] = models
        if models:
            self.dashboard_model_selector.set(models[0])

class ConfigTab(BaseTab):
    def __init__(self, notebook, parent_frame, main_app_instance):
        super().__init__(notebook, parent_frame, "⚙️ Configurations", main_app_instance)
        self.main_app = main_app_instance
        self.config_search_var = tk.StringVar()
        self.config_list = None
        self.config_details_text = None
        self.setup_ui()

    def setup_ui(self):
        # Left: list and search
        list_frame = ttk.LabelFrame(self.frame, text="Configurations")
        list_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        search_frame = ttk.Frame(list_frame)
        search_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(search_frame, text="🔍").pack(side='left')
        self.config_search_entry = ttk.Entry(search_frame, textvariable=self.config_search_var)
        self.config_search_entry.pack(fill='x', expand=True, side='left', padx=5)
        self.config_search_var.trace_add("write", self._filter_config_list)

        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(expand=True, fill='both', padx=5, pady=5)
        self.config_list = ttk.Treeview(tree_frame, columns=("File Name",), show="headings")
        self.config_list.heading("File Name", text="File Name")
        self.config_list.pack(side='left', expand=True, fill='both')
        list_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.config_list.yview)
        list_scrollbar.pack(side='right', fill='y')
        self.config_list.configure(yscrollcommand=list_scrollbar.set)
        self.config_list.bind('<<TreeviewSelect>>', self._on_config_select)

        # Right: details and actions
        details_frame = ttk.LabelFrame(self.frame, text="Configuration Details")
        details_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.config_details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, width=60)
        self.config_details_text.pack(expand=True, fill='both', padx=5, pady=5)
        self.config_details_text.config(state="disabled")

        button_frame = ttk.Frame(details_frame)
        button_frame.pack(side="bottom", fill="x", padx=5, pady=5)
        ttk.Button(button_frame, text="🔄 Refresh List", command=self._populate_config_list).pack(side="left", padx=5)
        ttk.Button(button_frame, text="➕ Create New", command=self._create_new_config_dialog).pack(side="left", padx=5)
        ttk.Button(button_frame, text="✅ Save", command=self._save_config).pack(side="right", padx=5)
        ttk.Button(button_frame, text="💾 Save As...", command=self._save_config_as).pack(side="right", padx=5)

        # Initial population
        self._populate_config_list()

    def _filter_config_list(self, *args):
        self.main_app._filter_config_list(*args)

    def _populate_config_list(self):
        self.main_app._populate_config_list()

    def _on_config_select(self, event):
        self.main_app._on_config_select(event)

    def _save_config(self):
        self.main_app._save_config()

    def _save_config_as(self):
        self.main_app._save_config_as()

    def _create_new_config_dialog(self):
        self.main_app._create_new_config_dialog()

class TrainingTab(BaseTab):
    def __init__(self, notebook, parent_frame, main_app_instance, launcher, thread_pool, training_episodes, training_avg_rewards, training_losses, realtime_metrics):
        super().__init__(notebook, parent_frame, "🎯 Training", main_app_instance)
        self.main_app = main_app_instance
        self.launcher = launcher
        self.thread_pool = thread_pool
        self.training_episodes = training_episodes
        self.training_avg_rewards = training_avg_rewards
        self.training_losses = training_losses
        self.realtime_metrics = realtime_metrics
        self.training_config_combobox = None
        self.resume_from_entry = None
        self.test_after_var = tk.BooleanVar(value=True)
        self.render_test_var = tk.BooleanVar(value=False)
        self.profile_var = tk.BooleanVar(value=False)
        self.training_launch_button = None
        self.training_log_text = None
        self.training_metrics_tree = None
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.canvas = None
        self.setup_ui()

    def setup_ui(self):
        training_frame = ttk.Frame(self.frame)
        training_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Training controls
        training_controls_frame = ttk.LabelFrame(training_frame, text="Training Controls")
        training_controls_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(training_controls_frame, text="Configuration:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.training_config_combobox = ttk.Combobox(training_controls_frame, state="readonly")
        self.training_config_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.training_config_combobox.bind("<<ComboboxSelected>>", self._on_training_config_select)

        ttk.Label(training_controls_frame, text="Resume From:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.resume_from_entry = ttk.Entry(training_controls_frame)
        self.resume_from_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Checkbutton(training_controls_frame, text="Test after training", variable=self.test_after_var).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(training_controls_frame, text="Render test", variable=self.render_test_var).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(training_controls_frame, text="Enable Performance Profiling", variable=self.profile_var).grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        self.training_launch_button = ttk.Button(training_controls_frame, text="🚀 Launch Training", command=self._launch_training_thread)
        self.training_launch_button.grid(row=4, column=0, columnspan=2, pady=10)

        training_controls_frame.grid_columnconfigure(1, weight=1)

        # Training logs and metrics
        training_display_pane = ttk.PanedWindow(training_frame, orient=tk.VERTICAL)
        training_display_pane.pack(expand=True, fill="both")

        # Log display
        log_frame = ttk.LabelFrame(training_display_pane, text="Training Log")
        training_display_pane.add(log_frame, weight=1)
        self.training_log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.training_log_text.pack(expand=True, fill="both", padx=5, pady=5)

        # Metrics display
        metrics_frame = ttk.LabelFrame(training_display_pane, text="Training Metrics")
        training_display_pane.add(metrics_frame, weight=1)

        metrics_pane = ttk.PanedWindow(metrics_frame, orient=tk.HORIZONTAL)
        metrics_pane.pack(expand=True, fill="both")

        # Metrics table
        tree_frame = ttk.Frame(metrics_pane)
        metrics_pane.add(tree_frame, weight=1)
        self.training_metrics_tree = ttk.Treeview(tree_frame, columns=("Episode", "Timesteps", "Avg Reward", "Loss"), show="headings")
        self.training_metrics_tree.heading("Episode", text="Episode")
        self.training_metrics_tree.heading("Timesteps", text="Timesteps")
        self.training_metrics_tree.heading("Avg Reward", text="Avg Reward")
        self.training_metrics_tree.heading("Loss", text="Loss")
        self.training_metrics_tree.pack(expand=True, fill="both", side='left')
        tree_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.training_metrics_tree.yview)
        tree_scrollbar.pack(side='right', fill='y')
        self.training_metrics_tree.configure(yscrollcommand=tree_scrollbar.set)

        # Metrics plot
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        plot_frame = ttk.Frame(metrics_pane)
        metrics_pane.add(plot_frame, weight=2)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill="both")

        self.main_app._populate_training_config_combobox() # Populate on init

    def _on_training_config_select(self, event):
        self.main_app._on_training_config_select(event)

    def _launch_training_thread(self):
        self.main_app._launch_training_thread()

class VisualizationTab(BaseTab):
    def __init__(self, notebook, parent_frame, main_app_instance):
        super().__init__(notebook, parent_frame, "📊 Visualization", main_app_instance)
        self.main_app = main_app_instance
        self.vis_type_combo = None
        self.vis_fig = None
        self.vis_canvas = None
        self.vis_canvas_widget = None
        self.setup_ui()

    def setup_ui(self):
        vis_frame = ttk.Frame(self.frame)
        vis_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Visualization controls
        controls_frame = ttk.LabelFrame(vis_frame, text="Visualization Controls")
        controls_frame.pack(fill="x", pady=(0, 10))

        control_grid = ttk.Frame(controls_frame)
        control_grid.pack(padx=10, pady=10)

        ttk.Label(control_grid, text="Visualization Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.vis_type_combo = ttk.Combobox(control_grid, values=[
            "Training Progress", "Policy Heatmap", "Value Function", 
            "Activation Patterns", "Reward Distribution", "Episode Length"
        ], state="readonly")
        self.vis_type_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.vis_type_combo.set("Training Progress")

        ttk.Button(control_grid, text="🔄 Update Visualization", 
                  command=self._update_visualization).grid(row=0, column=2, padx=10, pady=5)

        control_grid.grid_columnconfigure(1, weight=1)

        # Matplotlib visualization area
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.vis_fig = Figure(figsize=(12, 8), dpi=100)
        self.vis_canvas = FigureCanvasTkAgg(self.vis_fig, master=vis_frame)
        self.vis_canvas_widget = self.vis_canvas.get_tk_widget()
        self.vis_canvas_widget.pack(fill="both", expand=True)

        # Initialize with default visualization
        self._update_visualization()

    def _update_visualization(self):
        self.main_app._update_visualization()

class LeagueTab(BaseTab):
    def __init__(self, notebook, parent_frame, main_app_instance, league_manager, thread_pool):
        super().__init__(notebook, parent_frame, "🏆 League", main_app_instance)
        self.main_app = main_app_instance
        self.league_manager = league_manager
        self.thread_pool = thread_pool
        self.leaderboard_tree = None
        self.player1_combobox = None
        self.player2_combobox = None
        self.num_games_entry = None
        self.play_match_button = None
        self.setup_ui()

    def setup_ui(self):
        # Main layout container
        main_pane = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        main_pane.pack(expand=True, fill='both', padx=10, pady=10)

        # Left frame for leaderboard
        leaderboard_frame = ttk.LabelFrame(main_pane, text="🏆 Leaderboard")
        main_pane.add(leaderboard_frame, weight=2)

        # Extended leaderboard columns
        columns = ("rank", "name", "rating", "mmr", "wins", "losses", "win_rate")
        self.leaderboard_tree = ttk.Treeview(leaderboard_frame, columns=columns, show="headings")
        for col in columns:
            self.leaderboard_tree.heading(col, text=col.replace("_", " ").title(), command=lambda c=col: self._sort_leaderboard(c, False))
        self.leaderboard_tree.column("rank", width=60, anchor="center")
        self.leaderboard_tree.column("name", width=140, anchor="center")
        self.leaderboard_tree.column("rating", width=80, anchor="center")
        self.leaderboard_tree.column("mmr", width=80, anchor="center")
        self.leaderboard_tree.column("wins", width=60, anchor="center")
        self.leaderboard_tree.column("losses", width=60, anchor="center")
        self.leaderboard_tree.column("win_rate", width=80, anchor="center")
        self.leaderboard_tree.pack(fill="both", expand=True, side='left')
        
        leaderboard_scrollbar = ttk.Scrollbar(leaderboard_frame, orient="vertical", command=self.leaderboard_tree.yview)
        leaderboard_scrollbar.pack(side='right', fill='y')
        self.leaderboard_tree.config(yscrollcommand=leaderboard_scrollbar.set)

        # Right frame for controls and match playing
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)

        # Control buttons under leaderboard
        controls_frame = ttk.LabelFrame(right_frame, text="Player Management")
        controls_frame.pack(fill="x", padx=10, pady=(10,10))
        ttk.Button(controls_frame, text="🔄 Refresh", command=self._update_leaderboard_view).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="➕ Add Player", command=self._add_player_dialog).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="➖ Remove Player", command=self._remove_selected_player).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="⬆ Export CSV", command=self._export_league_csv).pack(side="right", padx=5)
        ttk.Button(controls_frame, text="⬇ Import CSV", command=self._import_league_csv).pack(side="right", padx=5)

        self._update_leaderboard_view()

        # Play Match
        play_match_frame = ttk.LabelFrame(right_frame, text="⚔️ Play Match")
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

        self.play_match_button = ttk.Button(play_match_frame, text="▶ Play Match", command=self._simulate_agents)
        self.play_match_button.grid(row=3, column=0, columnspan=2, pady=10)
        play_match_frame.grid_columnconfigure(1, weight=1)

        self._populate_player_comboboxes()

    def _sort_leaderboard(self, col, reverse):
        self.main_app.leaderboard_sort_col = col
        self.main_app.leaderboard_sort_rev = reverse
        data = [(self.leaderboard_tree.set(child, col), child) for child in self.leaderboard_tree.get_children("")]
        
        def sort_key(item):
            val = item[0]
            try:
                return float(val.strip('%'))
            except (ValueError, AttributeError):
                return val

        data.sort(key=sort_key, reverse=reverse)

        for index, (val, child) in enumerate(data):
            self.leaderboard_tree.move(child, "", index)

        self.leaderboard_tree.heading(col, command=lambda: self._sort_leaderboard(col, not reverse))

    def _update_leaderboard_view(self):
        self.main_app._update_leaderboard_view()

    def _add_player_dialog(self):
        self.main_app._add_player_dialog()

    def _remove_selected_player(self):
        self.main_app._remove_selected_player()

    def _export_league_csv(self):
        self.main_app._export_league_csv()

    def _import_league_csv(self):
        self.main_app._import_league_csv()

    def _populate_player_comboboxes(self):
        self.main_app._populate_player_comboboxes()

    def _play_league_match(self):
        self._simulate_agents()

    def _simulate_agents(self):
        player1_name = self.player1_combobox.get()
        player2_name = self.player2_combobox.get()
        num_games_str = self.num_games_entry.get().strip()

        if not player1_name or not player2_name:
            messagebox.showerror("Error", "Please select both players.")
            return
        if player1_name == player2_name:
            messagebox.showerror("Error", "Cannot simulate a match against yourself.")
            return

        try:
            num_games = int(num_games_str)
            if num_games < 1:
                messagebox.showerror("Error", "Number of games must be at least 1.")
                return
        except ValueError:
            messagebox.showerror("Error", "Number of games must be an integer.")
            return

        self.main_app._log_activity(f"Simulating {num_games} match(es): {player1_name} vs {player2_name}")
        # Placeholder for actual simulation logic
        messagebox.showinfo("Simulation Results", f"Simulation completed: {player1_name} vs {player2_name}")
        self.main_app._log_activity(f"Simulation completed: {player1_name} vs {player2_name}")

class ModelManagementTab(BaseTab):
    def __init__(self, notebook, parent_frame, main_app_instance):
        super().__init__(notebook, parent_frame, "🤖 Model Management", main_app_instance)
        self.main_app = main_app_instance
        self.model_listbox = None
        self.model_details_text = None
        self.setup_ui()

    def setup_ui(self):
        model_pane = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        model_pane.pack(expand=True, fill='both', padx=10, pady=10)

        list_frame = ttk.LabelFrame(model_pane, text="🤖 Available Models")
        model_pane.add(list_frame, weight=1)

        self.model_listbox = tk.Listbox(list_frame, height=20)
        self.model_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.model_listbox.bind("<<ListboxSelect>>", self._on_model_select)

        model_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.model_listbox.yview)
        model_scrollbar.pack(side="right", fill="y")
        self.model_listbox.config(yscrollcommand=model_scrollbar.set)

        details_frame = ttk.LabelFrame(model_pane, text="📋 Model Details")
        model_pane.add(details_frame, weight=2)

        self.model_details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, width=50, height=20)
        self.model_details_text.pack(padx=5, pady=5, fill="both", expand=True)
        self.model_details_text.config(state="disabled")

        button_frame = ttk.Frame(details_frame)
        button_frame.pack(side="bottom", fill="x", padx=5, pady=5)

        ttk.Button(button_frame, text="🔄 Refresh Models", command=self._populate_model_list).pack(side="left", padx=5)
        ttk.Button(button_frame, text="📥 Load Model", command=self._load_model).pack(side="left", padx=5)
        ttk.Button(button_frame, text="📊 Compare Models", command=self._compare_models).pack(side="left", padx=5)
        ttk.Button(button_frame, text="🗑️ Delete Model", command=self._delete_model).pack(side="right", padx=5)

        self._populate_model_list() # Populate on init

    def _populate_model_list(self):
        self.main_app._populate_model_list()

    def _on_model_select(self, event):
        self.main_app._on_model_select(event)

    def _load_model(self):
        self.main_app._load_model()

    def _delete_model(self):
        self.main_app._delete_model()

    def _compare_models(self):
        self.main_app._compare_models()

class HyperparamOptTab(BaseTab):
    def __init__(self, notebook, parent_frame, main_app_instance, thread_pool):
        super().__init__(notebook, parent_frame, "🔧 Hyperparameter Optimization", main_app_instance)
        self.main_app = main_app_instance
        self.thread_pool = thread_pool
        self.opt_config_combobox = None
        self.setup_ui()

    def setup_ui(self):
        opt_controls_frame = ttk.LabelFrame(self.frame, text="🔧 Optimization Controls")
        opt_controls_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(opt_controls_frame, text="Config Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.opt_config_combobox = ttk.Combobox(opt_controls_frame, state="readonly")
        self.opt_config_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Button(opt_controls_frame, text="▶ Start Optimization", 
                  command=lambda: self.main_app._log_activity("Hyperparameter optimization started")).grid(row=1, column=0, columnspan=2, pady=10)

        opt_controls_frame.grid_columnconfigure(1, weight=1)

class VersionControlTab(BaseTab):
    def __init__(self, notebook, parent_frame, main_app_instance):
        super().__init__(notebook, parent_frame, "📚 Version Control", main_app_instance)
        self.main_app = main_app_instance
        self.git_output_text = None
        self.setup_ui()

    def setup_ui(self):
        vc_frame = ttk.LabelFrame(self.frame, text="📚 Version Control & Developer Tools")
        vc_frame.pack(padx=10, pady=10, fill="both", expand=True)

        button_row = ttk.Frame(vc_frame)
        button_row.pack(fill="x", pady=5)
        ttk.Button(button_row, text="Git Status", command=lambda: self.main_app._log_activity("Git status checked")).pack(side="left", padx=5)
        ttk.Button(button_row, text="Git Add All", command=lambda: self.main_app._log_activity("All files added to git")).pack(side="left", padx=5)

        self.git_output_text = scrolledtext.ScrolledText(vc_frame, wrap=tk.WORD, height=20)
        self.git_output_text.pack(padx=5, pady=5, fill="both", expand=True)

class HelpTab(BaseTab):
    def __init__(self, notebook, parent_frame, main_app_instance):
        super().__init__(notebook, parent_frame, "❓ Help", main_app_instance)
        self.main_app = main_app_instance
        self.help_doc_combobox = None
        self.help_text_display = None
        self.setup_ui()

    def setup_ui(self):
        help_frame = ttk.Frame(self.frame)
        help_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Dropdown for selecting help documents
        doc_selection_frame = ttk.Frame(help_frame)
        doc_selection_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(doc_selection_frame, text="Select Document:").pack(side="left", padx=5)
        self.help_doc_combobox = ttk.Combobox(doc_selection_frame, state="readonly")
        self.help_doc_combobox.pack(side="left", fill="x", expand=True, padx=5)

        self.help_doc_combobox['values'] = ["README.md", "docs/wiki/Troubleshooting.md"]
        self.help_doc_combobox.set("README.md") # Default selection
        self.help_doc_combobox.bind("<<ComboboxSelected>>", self._load_help_document)

        # ScrolledText for displaying document content
        self.help_text_display = scrolledtext.ScrolledText(help_frame, wrap=tk.WORD)
        self.help_text_display.pack(expand=True, fill="both")
        self.help_text_display.config(state="disabled")

        # Load the default document on startup
        self._load_help_document()

    def _load_help_document(self, event=None):
        selected_doc = self.help_doc_combobox.get()
        file_path = Path(selected_doc)

        if not file_path.exists():
            self.help_text_display.config(state="normal")
            self.help_text_display.delete("1.0", tk.END)
            self.help_text_display.insert(tk.END, f"Error: Document '{selected_doc}' not found.")
            self.help_text_display.config(state="disabled")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.help_text_display.config(state="normal")
            self.help_text_display.delete("1.0", tk.END)
            self.help_text_display.insert(tk.END, content)
            self.help_text_display.config(state="disabled")
        except Exception as e:
            self.help_text_display.config(state="normal")
            self.help_text_display.delete("1.0", tk.END)
            self.help_text_display.insert(tk.END, f"Error loading document: {e}")
            self.help_text_display.config(state="disabled")
