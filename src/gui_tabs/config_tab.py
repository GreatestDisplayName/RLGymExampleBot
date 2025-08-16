import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

from .base_tab import BaseTab

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

        # Initial population deferred; main_app will call after full GUI construction


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