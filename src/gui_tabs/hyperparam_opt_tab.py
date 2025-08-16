from .base_tab import BaseTab

class HyperparamOptTab(BaseTab):
    def _setup_ui(self, **kwargs):
        ttk.Label(self.frame, text="Hyperparameter Optimization Tab Content").pack()