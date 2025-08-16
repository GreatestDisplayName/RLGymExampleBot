import tkinter as tk
from tkinter import ttk

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