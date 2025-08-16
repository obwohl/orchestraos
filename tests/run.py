#!/usr/bin/env python3

import os
import sys
import subprocess

# The lit executable is a python script.
lit_executable = '/usr/lib/llvm-20/build/utils/lit/lit.py'

if not os.path.exists(lit_executable):
    print(f"Error: 'lit' not found at {lit_executable}.")
    sys.exit(1)

# Get the directory of this script.
test_dir = os.path.dirname(os.path.abspath(__file__))

# Build the lit command.
command = ['/usr/bin/python3', lit_executable, '-v', '--debug', test_dir]

# Set the PYTHONPATH to include the lit modules.
env = os.environ.copy()
env['PYTHONPATH'] = '/usr/lib/llvm-20/build/utils/lit' + os.pathsep + env.get('PYTHONPATH', '')
env['MY_OBJ_ROOT'] = os.path.abspath(os.path.join(test_dir, '..', 'orchestra-compiler', 'build'))

# Run lit.
process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

# Print the output.
print(process.stdout)
print(process.stderr, file=sys.stderr)

# Exit with the same return code as lit.
sys.exit(process.returncode)
