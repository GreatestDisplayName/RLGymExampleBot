from .base_tab import BaseTab

class HelpTab(BaseTab):
    def _setup_ui(self, **kwargs):
        ttk.Label(self.frame, text="Help Tab Content").pack()