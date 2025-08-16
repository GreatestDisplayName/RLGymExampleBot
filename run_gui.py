import tkinter as tk
from ttkthemes import ThemedTk
from gui import RLGymGUI

if __name__ == "__main__":
    root = ThemedTk()
    app = RLGymGUI(root)
    root.mainloop()
