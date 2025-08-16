# GUI Guide

This page provides documentation for the RLGym Workflow Manager Graphical User Interface (`gui.py`).

The GUI offers a user-friendly way to interact with the RLGym project, allowing you to manage training configurations, launch training sessions, and view results without needing to use command-line arguments extensively.

## 1. Running the GUI

To start the GUI, navigate to your project root directory (`RLGymExampleBot/`) and run the `gui.py` script:

```bash
python gui.py
```

## 2. GUI Structure

The GUI is organized into several tabs, each dedicated to a specific aspect of the RLGym workflow:

*   **Configurations**: Manage training configurations (view, select, create).
*   **Training**: Launch and monitor training sessions (controls to be implemented).
*   **League**: Manage the self-play league (controls to be implemented).
*   **Logs**: View training and application logs (viewer to be implemented).

## 3. Configurations Tab

This tab allows you to manage your training configurations.

### Available Configurations

*   Displays a list of all available training configurations (both default and custom).
*   You can select a configuration from the list to view its details.

### Configuration Details

*   Shows the JSON content of the selected configuration in a read-only text area.

### Actions

*   **Refresh**: Updates the list of available configurations.
*   **Create New Config**: Opens a dialog to create a new training configuration by providing a name, agent type, and timesteps.

## 4. Training Tab (Future Implementation)

This tab is intended to provide controls for launching and monitoring training sessions. Its functionality is currently a placeholder.

## 5. League Tab (Future Implementation)

This tab is intended to provide controls for managing the self-play league. Its functionality is currently a placeholder.

## 6. Logs Tab (Future Implementation)

This tab is intended to provide a viewer for training and application logs. Its functionality is currently a placeholder.

## 7. Backend Integration

The GUI interacts with the `src/launch_training.py` module (via the `TrainingLauncher` class) to handle backend logic such as:

*   Listing configurations.
*   Loading configuration details.
*   Creating new configurations.
*   (Future) Launching training sessions.

## 8. Error Handling

The GUI includes basic error handling using `tkinter.messagebox` to display errors to the user (e.g., "Configuration not found", "Failed to load config"). Detailed errors are also logged to the console and log files via the `src/logger.py` module.

[Home](Home.md)