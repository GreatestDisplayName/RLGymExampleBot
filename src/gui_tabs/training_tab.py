import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .base_tab import BaseTab

class TrainingTab(BaseTab):
    def __init__(self, notebook, parent_frame, main_app_instance, **kwargs):
        super().__init__(notebook, parent_frame, "🎯 Training", main_app_instance)
        self.main_app = main_app_instance
        
        # Initialize with defaults or from kwargs
        self.launcher = kwargs.get('launcher', getattr(main_app_instance, 'launcher', None))
        self.thread_pool = kwargs.get('thread_pool', getattr(main_app_instance, 'thread_pool', None))
        self.training_episodes = kwargs.get('training_episodes', [])
        self.training_avg_rewards = kwargs.get('training_avg_rewards', [])
        self.training_losses = kwargs.get('training_losses', [])
        self.realtime_metrics = kwargs.get('realtime_metrics', {})
        
        # UI components
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
        
        try:
            self.setup_ui()
        except Exception as e:
            print(f"Error initializing TrainingTab: {e}")
            # Continue with partial initialization to avoid crashing the entire app

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
        plot_frame = ttk.Frame(metrics_pane)
        metrics_pane.add(plot_frame, weight=2)
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill="both")

        # Initialize with default values if possible
        self._initialize_training_config()

    def _on_training_config_select(self, event):
        if hasattr(self.main_app, '_on_training_config_select'):
            self.main_app._on_training_config_select(event)

    def _launch_training_thread(self):
        if hasattr(self.main_app, '_launch_training_thread'):
            self.main_app._launch_training_thread()
            
    def _populate_training_config_combobox(self):
        """Safely populate the training config combobox"""
        if not hasattr(self, 'training_config_combobox') or not self.training_config_combobox:
            return
            
        try:
            # Get configs from launcher if available
            if hasattr(self, 'launcher') and self.launcher:
                default_configs = list(getattr(self.launcher, 'default_configs', {}).keys())
                config_dir = getattr(self.launcher, 'config_dir', None)
                if config_dir and hasattr(config_dir, 'glob'):
                    custom_configs = [f.stem for f in config_dir.glob("*.json") 
                                   if f.stem not in default_configs]
                    all_configs = sorted(default_configs + custom_configs)
                    self.training_config_combobox['values'] = all_configs
                    if all_configs:
                        self.training_config_combobox.set(all_configs[0])
        except Exception as e:
            print(f"Error populating training config combobox: {e}")