import lit.formats
import os

# name: The name of this test suite.
config.name = 'Orchestra'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not lit.util.is_string)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# substitutions: A list of substitutions to make in test files.
# We don't need substitutions now that we are modifying the PATH.

# Add the LLVM tools to the PATH.
config.llvm_tools_dir = "/usr/lib/llvm-20/bin"
config.environment['PATH'] = config.llvm_tools_dir + os.pathsep + config.environment['PATH']

# Add the orchestra-opt directory to the PATH.
# The `MY_OBJ_ROOT` environment variable is set by the `run.py` script.
obj_root = os.environ.get('MY_OBJ_ROOT')
if obj_root:
    orchestra_opt_dir = os.path.join(obj_root, 'orchestra-opt')
    config.environment['PATH'] = orchestra_opt_dir + os.pathsep + config.environment['PATH']
