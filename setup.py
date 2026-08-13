#!/usr/bin/env python3
"""Check/install dependencies for visibility study."""

import importlib.util
import shutil
import subprocess
import sys

PYTHON_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "xarray": "xarray",
    "netCDF4": "netCDF4",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scikit-learn": "scikit-learn",
    "metar-taf-parser-mivek": "metar-taf-parser-mivek"
}


def ask_yes_no(prompt):
    return input(f"{prompt} [y/N]: ").strip().lower() in {"y", "yes"}


def main():
    print(f"Python executable: {sys.executable}\n")

    missing = [
        pip_name
        for import_name, pip_name in PYTHON_PACKAGES.items()
        if importlib.util.find_spec(import_name) is None
    ]

    if missing:
        print("Missing Python packages:")
        for package in missing:
            print(f"  - {package}")

        if ask_yes_no("\nInstall missing Python packages?"):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *missing]
            )
        else:
            print("Python package installation skipped.")
    else:
        print("All required Python packages are installed.")

    print()

    print("\nSetup check complete.")


if __name__ == "__main__":
    main()