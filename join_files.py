import numpy as np
import pandas as pd
import os
import glob

import subprocess

def save_tree_output_to_file(output_file):
    try:
        result = subprocess.run(['tree'], capture_output=True, text=True)

        with open(output_file, 'w') as file:
            file.write(result.stdout)

    except Exception as e:
        pass


def get_script_files(directory=None, extension=None, ignore_folders=None):
    if directory is None:
        directory = "."
    if extension is None:
        extension = ["all"]
    if isinstance(extension, str):
        extension = [extension]

    # Ensure ignore_folders is a list if provided
    if ignore_folders is None:
        ignore_folders = []
    elif isinstance(ignore_folders, str):
        ignore_folders = [ignore_folders]

    paths = []
    if "all" in extension:
        # Recursively get all files
        all_files = glob.glob(os.path.join(directory, "**", "*"), recursive=True)
        for f in all_files:
            if os.path.isfile(f):
                # Get the relative path from the base directory
                rel_path = os.path.relpath(f, directory)
                # Split the relative path into its components (folders and file)
                parts = os.path.normpath(rel_path).split(os.sep)
                # If any folder in the path is in the ignore list, skip the file
                if any(folder in ignore_folders for folder in parts[:-1]):
                    continue
                paths.append(f)
    else:
        # Recursively get files matching the provided extension(s)
        for ext in extension:
            files = glob.glob(os.path.join(directory, "**", f"*.{ext}"), recursive=True)
            for f in files:
                if os.path.isfile(f):
                    rel_path = os.path.relpath(f, directory)
                    parts = os.path.normpath(rel_path).split(os.sep)
                    if any(folder in ignore_folders for folder in parts[:-1]):
                        continue
                    paths.append(f)
    return paths

def load_script_files(script_files):
    script_dict = {}
    for file in script_files:
        try:
            with open(file, 'r', encoding='utf-8', errors='replace') as f:
                script_dict[file] = f.readlines()
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return script_dict

def list_to_text(script_dict, initial_text=""):
    divider = "\n\n" + "=" * 80 + "\n\n"
    script_text = initial_text
    if initial_text != "":
        script_text += divider
    for file_name, script in script_dict.items():
        script_text += "FILE NAME: " + file_name + "\n\n"
        for line in script:
            script_text += line
        script_text += divider
    return script_text

def text_to_file(text, filename):
    with open(filename + ".txt", 'w') as f:
        f.write(text)


def scripts_to_file(directory, extension, initial_text, filename, filter_files=None, ignore_files=None, ignore_folders=None):
    script_files = get_script_files(directory, extension, ignore_folders)
    if filter_files is not None:
        filter_files = [directory + "/" + file for file in filter_files]
        script_files = [file for file in script_files if file in filter_files]
    if ignore_files is not None:
        ignore_files = [directory + "/" + file for file in ignore_files]
        script_files = [file for file in script_files if file not in ignore_files]
    script_list = load_script_files(script_files)
    script_text = list_to_text(script_list, initial_text)
    text_to_file(script_text, filename)

ALL_EXPLANATION = "The following contains a tree of the directory and all the important files of the current version of my project: "

if __name__ == "__main__":
    if os.path.exists("scripts"):
        os.system("rm -r scripts")
    os.mkdir("scripts")
    save_tree_output_to_file("scripts/tree_output.txt")
    scripts_to_file(
        "portfolio_management", "py", "Package files:", "scripts/portfolio_management",
        ignore_files=["__init__.py", "fx.py", "backtest.py"]
    )
    scripts_to_file(
        "tests", "py", "Test files:", "scripts/tests.txt",
        ignore_files=["__init__.py"]
    )
    scripts_to_file(
        "doc", "all", "Sphix Documentation", "scripts/doc", ignore_folders="build"
    )
    scripts_to_file(
        "scripts", "txt", ALL_EXPLANATION,
        "scripts/all"
    )
    os.system("cat scripts/all.txt | pbcopy")
