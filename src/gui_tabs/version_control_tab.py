import tkinter as tk
from tkinter import ttk
from .base_tab import BaseTab

class VersionControlTab(BaseTab):
    def __init__(self, notebook, parent_frame, main_app_instance):
        super().__init__(notebook, parent_frame, "📚 Version Control", main_app_instance)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the version control UI"""
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Version control status
        status_frame = ttk.LabelFrame(main_frame, text="Repository Status")
        status_frame.pack(fill="x", pady=(0, 10))
        
        # Status text area
        self.status_text = tk.Text(status_frame, height=8, wrap=tk.WORD)
        self.status_text.pack(expand=True, fill="both", padx=5, pady=5)
        self.status_text.insert("1.0", "No repository information available.")
        self.status_text.config(state="disabled")
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=5)
        
        ttk.Button(button_frame, text="🔄 Refresh", command=self._refresh_status).pack(side="left", padx=5)
        ttk.Button(button_frame, text="⬆️ Commit & Push", command=self._commit_and_push).pack(side="left", padx=5)
        ttk.Button(button_frame, text="⬇️ Pull", command=self._pull_changes).pack(side="left", padx=5)
        
        # Changes list
        changes_frame = ttk.LabelFrame(main_frame, text="Changes")
        changes_frame.pack(fill="both", expand=True)
        
        # Changes treeview
        columns = ("Status", "File", "Path")
        self.changes_tree = ttk.Treeview(
            changes_frame,
            columns=columns,
            show="headings",
            selectmode="extended"
        )
        
        # Configure columns
        for col in columns:
            self.changes_tree.heading(col, text=col)
            self.changes_tree.column(col, width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(changes_frame, orient="vertical", command=self.changes_tree.yview)
        self.changes_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack tree and scrollbar
        self.changes_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Initialize with some example data
        self._refresh_status()
    
    def _refresh_status(self):
        """Refresh the repository status"""
        self.status_text.config(state="normal")
        self.status_text.delete("1.0", tk.END)
        self.status_text.insert("1.0", "Repository status updated at: 2023-01-01 12:00:00\n"
                                  "Branch: main\n"
                                  "No uncommitted changes.")
        self.status_text.config(state="disabled")
        
        # Clear and repopulate changes
        for item in self.changes_tree.get_children():
            self.changes_tree.delete(item)
            
        # Add example changes
        example_changes = [
            ("M", "gui.py", "src/gui_tabs/"),
            ("A", "new_feature.py", "src/features/"),
            ("D", "old_file.py", "src/utils/")
        ]
        
        for status, filename, path in example_changes:
            self.changes_tree.insert("", "end", values=(status, filename, path))
    
    def _commit_and_push(self):
        """Handle commit and push action"""
        print("Commit and push functionality will be implemented here")
    
    def _pull_changes(self):
        """Handle pull changes action"""
        print("Pull changes functionality will be implemented here")