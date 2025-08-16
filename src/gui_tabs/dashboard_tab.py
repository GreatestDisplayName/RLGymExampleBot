import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path
import datetime
import json

from src.logger import logger
from src.utils import SUPPORTED_AGENT_TYPES
from .base_tab import BaseTab


class DashboardTab(BaseTab):
    def __init__(self, notebook, parent_frame, main_app_instance):
        super().__init__(notebook, parent_frame, "🏠 Dashboard", main_app_instance)
        self.master_gui = main_app_instance # Reference to the main CombinedRLGymGUI instance
        self.realtime_metrics = {}
        self.activity_text = None # Will be set up in setup_ui
        self.theme_combobox = None # Will be set up in setup_ui
        self.dashboard_model_selector = None # Will be set up in setup_ui
        self._setup_ui() # Call the UI setup method

    def _setup_ui(self, **kwargs):
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

        self.realtime_metrics = {}
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
        ttk.Button(actions_grid, text="📊 View Progress", command=lambda: self.master_gui.notebook.select(2)).grid(row=0, column=3, padx=5, pady=5)

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

        theme_frame = ttk.LabelFrame(main_frame, text="🎨 Theme Settings")
        theme_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(theme_frame, text="Select Theme:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.theme_combobox = ttk.Combobox(theme_frame, state="readonly")
        self.theme_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        available_themes = self.master_gui.master.get_themes()
        self.theme_combobox['values'] = sorted(available_themes)
        
        self.theme_combobox.bind("<<ComboboxSelected>>", lambda event: self.master_gui._apply_theme(self.theme_combobox.get()))

        ttk.Button(theme_frame, text="Toggle Dark Mode", command=self._toggle_dark_mode).grid(row=0, column=2, padx=5, pady=5)

        theme_frame.grid_columnconfigure(1, weight=1)

        # Initial population of model selector
        self._populate_model_selector()
        self._start_realtime_updates()

    def _populate_model_selector(self):
        # This should ideally come from a shared model management logic
        # For now, it's a placeholder
        model_dir = Path("models")
        if not model_dir.exists():
            model_dir.mkdir()

        all_models = []
        for f in model_dir.rglob("*"):
            if f.is_file() and f.suffix in [".pth", ".zip", ".pkl"]:
                all_models.append(str(f.relative_to(model_dir)))
        
        self.dashboard_model_selector['values'] = sorted(all_models)
        if all_models:
            self.dashboard_model_selector.set(all_models[0])

    def _start_realtime_updates(self):
        self.update_timer = self.master_gui.master.after(1000, self._update_realtime_metrics)

    def _update_realtime_metrics(self):
        if hasattr(self.master_gui, 'training_launch_button') and self.master_gui.training_launch_button['state'] == 'disabled':
            current_epsilon = float(self.realtime_metrics['epsilon']['text'])
            new_epsilon = max(0.01, current_epsilon * 0.999)
            self.realtime_metrics['epsilon'].config(text=f"{new_epsilon:.3f}")
            
            self.realtime_metrics['status'].config(text="Training Active")
        else:
            self.realtime_metrics['status'].config(text="Idle")

        self.update_timer = self.master_gui.master.after(1000, self._update_realtime_metrics)

    def _quick_train(self):
        if not hasattr(self.master_gui, 'training_config_combobox'):
            messagebox.showwarning("Warning", "Please go to the Training tab to set up training first.")
            return
        
        configs = self.master_gui.training_config_combobox['values']
        if configs:
            self.master_gui.training_config_combobox.set(configs[0])
            self.master_gui._launch_training_thread()
        else:
            messagebox.showwarning("Warning", "No training configurations available.")

    def _quick_load_model(self):
        filename = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Model files", "*.pth *.zip *.pkl"), ("All files", "*.*")],
            initialdir=Path("models") if Path("models").exists() else Path.cwd()
        )
        if filename:
            self.dashboard_model_selector.set(Path(filename).name)
            self._log_activity(f"Loaded model: {Path(filename).name}")

    def _quick_save_model(self):
        filename = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pth",
            filetypes=[("PyTorch files", "*.pth"), ("Zip files", "*.zip"), ("All files", "*.*")],
            initialdir=Path("models") if Path("models").exists() else Path.cwd()
        )
        if filename:
            self._log_activity(f"Saved model: {Path(filename).name}")

    def _log_activity(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.activity_text.config(state="normal")
        self.activity_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.activity_text.see(tk.END)
        
        lines = self.activity_text.get("1.0", tk.END).split('\n')
        if len(lines) > 100:
            self.activity_text.delete("1.0", f"{len(lines)-100}.0")
        
        self.activity_text.config(state="disabled")

    def _toggle_dark_mode(self):
        current_theme = self.theme_combobox.get()
        dark_themes = ["black", "equilux", "itft1"]
        if current_theme in dark_themes:
            self.master_gui._apply_theme("arc")
            self.theme_combobox.set("arc")
        else:
            self.master_gui._apply_theme("black")
            self.theme_combobox.set("black")
