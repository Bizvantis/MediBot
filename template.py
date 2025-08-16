import os
from pathlib import Path
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# List of files and folders to create
list_of_files = [
    "SRC/__init__.py",
    "SRC/helper.py",
    "SRC/prompt.py",
    ".env",
    "setup.py",
    "app.py",
    "research/trials.ipynb",
]

# Loop through the list of files and create them
for filepath in list_of_files:
    filepath = Path(filepath) # Converts path to be compatible with the current operating system
    filedir, filename = os.path.split(filepath) # Splits the path into directory and file name

    if filedir != "":
        os.makedirs(filedir, exist_ok=True) # Creates the directory if it doesn't exist
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass # Creates an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")