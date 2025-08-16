#!/usr/bin/env python
"""
🚀 RLGym Pro v7.0 - Fixed for rlgym-toolkit + Port 8080
"""

import os
import sys
import time
import threading
import numpy as np

# ======================
#  🔧 DEPENDENCY CHECK
# ======================

def check_deps():
    errors = []
    try:
        import rlgym
        if not hasattr(rlgym, "make"):
            errors.append("rlgym is old version. Install rlgym-toolkit")
    except ImportError:
        errors.append("rlgym not found: pip install git+https://github.com/AechPro/rlgym-toolkit")

    try:
        import rlgym_tools
    except ImportError:
        errors.append("rlgym-tools: pip install git+https://github.com/AechPro/rlgym-tools")

    try:
        import stable_baselines3
    except ImportError:
        errors.append("stable-baselines3: pip install stable-baselines3==2.2.1")

    try:
        import torch
    except ImportError:
        errors.append("torch: pip install torch==2.7.1")

    try:
        import onnxruntime
    except ImportError:
        errors.append("onnxruntime: pip install onnxruntime")

    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        errors.append("PyQt6: pip install PyQt6")

    try:
        from fastapi import FastAPI
    except ImportError:
        errors.append("fastapi: pip install fastapi uvicorn")

    if errors:
        print("❌ Missing packages:")
        for e in errors:
            print(f"   {e}")
        print("\n💡 Install with:")
        print("   pip install git+https://github.com/AechPro/rlgym-toolkit")
        print("   pip install git+https://github.com/AechPro/rlgym-tools")
        print("   pip install stable-baselines3==2.2.1 torch==2.7.1 onnxruntime PyQt6 fastapi uvicorn")
        sys.exit(1)

    print("✅ All dependencies OK")

check_deps()

# Now safe to import
import rlgym
import rlgym_tools
from stable_baselines3 import PPO
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import QTimer
from fastapi import FastAPI
import uvicorn


# ======================
#  🧠 GLOBAL
# ======================

class Global:
    stats = {"mode": "idle", "fps": 0, "infer_ms": 0}
    running = True


# ======================
#  🧠 TRAINING
# ======================

def train_ppo():
    print("🎮 Creating RLGym environment...")
    try:
        env = rlgym.make(
            tick_skip=8,
            team_size=1,
            obs_builder=rlgym_tools.extra_obs.AdvancedObs()
        )
        model = PPO("MlpPolicy", env, verbose=1, device="auto")
        model.learn(total_timesteps=5000)  # Fast demo
        print("✅ Training complete!")
        with Global.lock:
            Global.stats["mode"] = "trained"
    except Exception as e:
        print(f"❌ Training failed: {e}")


# ======================
#  🌐 WEB DASHBOARD (Port 8080)
# ======================

web_app = FastAPI()

@web_app.get("/stats")
def get_stats():
    return Global.stats.copy()

def run_web():
    uvicorn.run(web_app, host="127.0.0.1", port=8080, log_level="warning")  # Changed port


# ======================
#  🖥️ GUI
# ======================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RLGym Pro Fixed")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()
        self.label = QLabel("Status: Idle")
        layout.addWidget(self.label)

        btn_train = QPushButton("Train PPO")
        btn_train.clicked.connect(lambda: threading.Thread(target=train_ppo, daemon=True).start())
        layout.addWidget(btn_train)

        btn_web = QPushButton("Open Web Dashboard (http://localhost:8080)")
        btn_web.clicked.connect(lambda: os.startfile("http://localhost:8080"))
        layout.addWidget(btn_web)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_label)
        self.timer.start(500)

    def update_label(self):
        self.label.setText(f"Mode: {Global.stats['mode']}")


# ======================
#  🚀 MAIN
# ======================

if __name__ == "__main__":
    # Start web server on port 8080
    threading.Thread(target=run_web, daemon=True).start()

    # Start GUI
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
