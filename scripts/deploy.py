import os
import shutil
import zipfile
from pathlib import Path

def deploy():
    """
    Packages the bot for distribution.
    """
    print("Creating deployment package...")

    # Create dist directory
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)

    # Create a temporary directory for the package contents
    temp_dir = Path("dist/temp_package")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # Files and directories to include in the package
    include_items = [
        "src",
        "models",
        "configs",
        "requirements.txt",
        "run_gui.bat",
        "run_gui.py",
        "gui.py",
        "theme_settings.json",
        "README.md",
        "LICENSE"
    ]

    # Copy items to the temporary directory
    for item in include_items:
        item_path = Path(item)
        if item_path.is_dir():
            shutil.copytree(item_path, temp_dir / item)
        else:
            shutil.copy(item_path, temp_dir / item)

    # Create the zip file
    zip_path = dist_dir / "RLGymExampleBot.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                zipf.write(file_path, file_path.relative_to(temp_dir))

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)

    print(f"Deployment package created at: {zip_path}")

if __name__ == "__main__":
    deploy()
