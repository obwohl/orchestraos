# RULES:

1) If unsure about correct syntax: search google. do not guess syntax after failing more than 2 times.

2) If unsure about more abstract concepts and you are failing more than 2 times: stop guessing and search google

3) **When to ask for help:** You should work as independently as possible. Only ask the user for help if you are truly stuck on an unsolvable problem or a very complex question that you cannot answer on your own or with a Google search. When you do ask for help, you must formulate a comprehensive research question that is specific, directed, and can be understood by an agnostic search agent who knows nothing about the repository.

4) **Permissions:** You are the first and only developer of this repository. You have full permission to create, delete, and refactor any files as you see fit. Do not ask for permission to perform these actions. Work as independently as possible.

5) You are encouraged to keep track of your trials, ideas, erros and successes in logging files - so to say your development diary. If you decide to use this, build and write them in a concise style. If you find those diary documents from a developer before you, use them and feel free to add.


# PREREQUISITES:
1) As prerequisites you have to make sure llvm is installed or to install LLVM (and thus `mlir-tblgen`) (e.g., via Homebrew, apt-get or whatever you prefer), providing the necessary MLIR tools. DO NOT GUESS what the latest version of LLVM is, but rather install it the normal way such that the latest stable version is installed.
2) IMPORTANT: all the steps (in the other .md documents), when they talk about linking something to llvm/mlir is maybe not relevant, depending on the way you installed llvm. Your way of handling the llvm is the correct way, if it works.
3) **A note on CMake and Linux:** After installing LLVM and MLIR via `apt-get`, you might need to explicitly tell CMake where to find the MLIR and LLVM installations. You can do this by setting `MLIR_DIR` and `LLVM_DIR` when you run `cmake`. For example:
   `cmake .. -DMLIR_DIR=/path/to/mlir/cmake -DLLVM_DIR=/path/to/llvm/cmake`
   You may also need to add the MLIR CMake module path to your root `CMakeLists.txt` file, like so:
   `list(APPEND CMAKE_MODULE_PATH "/path/to/mlir/cmake")`
   The exact paths will depend on the version of LLVM you installed.

# WHAT TO DO
1) Check the plan/ folder. Aside from this very document here are other md files which contain plans. The proposed syntax could VERY likely be outdated or not perfectly correct. Do not slavishly adhere to the exact plan or syntax there if it does not work or does not make sense for your specific setup. You have to check what has already been done and what you have to do to proceed.
2) If you see that the full plan would be too large, you are absolutely encouraged to switch to a small testable part of that plan. Mini steps are great. You are absolutely allowed to alter the exact plan or pivot from you own plans if neccessary, if it is in the spirit of the project.
