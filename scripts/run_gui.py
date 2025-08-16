"""
This script provides a convenient way to launch the RLBot GUI directly from your bot project.

Using the GUI, you can easily set up matches with custom settings, which is ideal for testing and debugging your bot.
For developers using an IDE like PyCharm or VS Code, this script also facilitates breakpoint debugging.
"""

import sys
import subprocess

def check_and_start_gui():
    """Checks for rlbot_gui and starts it, or provides installation instructions."""
    try:
        # Check if rlbot_gui is installed
        __import__('rlbot_gui')
    except ImportError:
        print("❌ Error: 'rlbot_gui' is not installed.", file=sys.stderr)
        response = input("Would you like to install it now? (y/n): ").lower()
        if response == 'y':
            try:
                print("Installing rlbot_gui...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "rlbot_gui"])
                print("✅ rlbot_gui installed successfully. Please run the script again.")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install rlbot_gui. Pip exited with code {e.returncode}.", file=sys.stderr)
                print("Please try running 'pip install rlbot_gui' manually.", file=sys.stderr)
        else:
            print("Installation cancelled. Please install rlbot_gui to use the GUI.")
        sys.exit(1)

    # If the import was successful, start the GUI
    from rlbot_gui import gui
    print("Launching RLBot GUI...")
    gui.start()

def main():
    """Main function to run the script."""
    check_and_start_gui()

if __name__ == '__main__':
    main()