# Setup Guide

This page details how to set up the RLGymExampleBot project.

## 1. Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.8+**: The project is developed and tested with Python 3.8 and above. You can download Python from [python.org](https://www.python.org/downloads/).
*   **Git**: For cloning the project repository. Download from [git-scm.com](https://git-scm.com/downloads).
*   **Rocket League**: The game itself, as RLGym interacts with it.

## 2. Project Setup

1.  **Clone the Repository**:
    Open your terminal or command prompt and clone the project:
    ```bash
    git clone https://github.com/your-repo/RLGymExampleBot.git
    cd RLGymExampleBot
    ```
    *(Replace `https://github.com/your-repo/RLGymExampleBot.git` with the actual repository URL if different.)*

2.  **Create and Activate a Virtual Environment**:
    It is highly recommended to use a Python virtual environment to manage dependencies and avoid conflicts with other Python projects.
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    Navigate to the `src` directory and install the required Python packages.
    ```bash
    cd src
    pip install -r requirements.txt
    ```
    Then, return to the project root:
    ```bash
    cd ..
    ```

4.  **Install RLBot Framework**:
    The RLBot framework is necessary for running your trained bots in Rocket League.
    ```bash
    pip install rlbot
    ```

## 3. Verify Setup

You can verify your environment setup by running a simple test:
```bash
python -c "import gymnasium; import stable_baselines3; print('Environment setup looks good!')"
```
If this command runs without errors, your basic Python environment is correctly configured.

For more detailed verification, you can test the RLGym environment:
```bash
cd src
python training_env.py
cd ..
```
This will run a quick test of the training environment.

[Home](Home.md)