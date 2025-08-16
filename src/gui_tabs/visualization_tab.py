import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .base_tab import BaseTab

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
        self.vis_fig = Figure(figsize=(12, 8), dpi=100)
        self.vis_canvas = FigureCanvasTkAgg(self.vis_fig, master=vis_frame)
        self.vis_canvas_widget = self.vis_canvas.get_tk_widget()
        self.vis_canvas_widget.pack(fill="both", expand=True)

        # Initialize with default visualization
        self._update_visualization()

    def _update_visualization(self):
        if hasattr(self.main_app, '_update_visualization'):
            self.main_app._update_visualization()
        else:
            # Clear the figure and show a message
            self.vis_fig.clear()
            ax = self.vis_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Visualization not available\nin this version', 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            self.vis_canvas.draw()