#!/usr/bin/env python3

import os
import subprocess

def find_files(directory, extensions):
    """Find all files with given extensions in a directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                yield os.path.join(root, file)

def main():
    """Main function to run clang-format on specified files."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    compiler_dir = os.path.join(repo_root, 'orchestra-compiler')

    # Excluded .td files for now to resolve the TableGen formatting issue.
    extensions = ['.cpp', '.h']

    files_to_format = list(find_files(compiler_dir, extensions))

    if not files_to_format:
        print("No files to format.")
        return

    print(f"Found {len(files_to_format)} files to format. Formatting...")

    try:
        # Note: The environment setup installed clang-format-20
        subprocess.run(['clang-format-20', '-i'] + files_to_format, check=True)
        print("Formatting complete.")
    except FileNotFoundError:
        print("Error: 'clang-format-20' not found. Make sure it is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during formatting: {e}")

if __name__ == "__main__":
    main()
