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
config.substitutions.append(('%orchestra-opt', '/app/orchestra-compiler/build/orchestra-opt/orchestra-opt'))

# Add the LLVM tools to the PATH.
config.llvm_tools_dir = "/usr/lib/llvm-20"
config.environment['PATH'] = os.path.join(config.llvm_tools_dir, 'bin') + os.pathsep + config.environment['PATH']
