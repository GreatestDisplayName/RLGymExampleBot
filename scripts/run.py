"""
This script is the main entry point for running the RLGym bot.

It ensures that all necessary dependencies are installed and then launches the RLBot framework.
"""

import sys
import subprocess
import pkg_resources

def check_dependencies(requirements_path: str):
    """Checks if all dependencies from the requirements file are installed."""
    try:
        with open(requirements_path, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        pkg_resources.require(requirements)
        print("✅ All dependencies are satisfied.")
        return True
    except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as e:
        print(f"❌ Dependency check failed: {e}")
        return False
    except FileNotFoundError:
        print(f"⚠️ Could not find '{requirements_path}'. Skipping dependency check.")
        return True # Assume dependencies are met if file is missing

def install_dependencies(requirements_path: str):
    """Installs dependencies from the given requirements file using pip."""
    print(f"Installing dependencies from {requirements_path}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path, "--upgrade"])
        print("✅ Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies. Pip exited with code {e.returncode}.", file=sys.stderr)
        print("Please try running 'pip install -r requirements.txt' manually.", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function to run the bot."""
    print("--- RLGym Bot Runner ---")
    requirements_file = 'requirements.txt'

    if not check_dependencies(requirements_file):
        print("Some dependencies are missing or have version conflicts.")
        install_dependencies(requirements_file)
        print("Restarting script to load new packages...")
        
        # Re-execute the script in a new process
        try:
            subprocess.check_call([sys.executable] + sys.argv)
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to restart script. Please start it again manually. Error code: {e.returncode}", file=sys.stderr)
            sys.exit(1)

    print("Launching RLBot framework...")
    try:
        from rlbot import runner
        runner.main()
    except ImportError as e:
        print(f"❌ Error: Could not import rlbot. {e}", file=sys.stderr)
        print("Please ensure RLBot is installed correctly ('pip install rlbot').", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        print("Press Enter to close.")
        input()
        sys.exit(1)

if __name__ == '__main__':
    main()