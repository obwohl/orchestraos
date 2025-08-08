# RULES:

1) If unsure about correct syntax: search google. do not guess syntax after failing more than 2 times.

2) If unsure about more abstract concepts and you are failing more than 2 times: stop guessing and search google

3) If syntax or very complex question(s) (it could be a very new concept, which is totally non-standard) could not be solved by your own or by a quick google search:
Create a distinct research question for the user which he can research then. The question must be formulated in a way that a totally agnostic search agent (who does not know anything about our repo) can perform the research for your question. Make it VERY specific and directed in a way that your question(s) will be answered comprehensively.

3) You are the first and only developer of this repo. That means feel free to create and delete and refactor anything you like. Pretty much the most files will not be there yet. You have to create them, and you do not need to ask anybody about that. Work as independent as possible.


4) You are encouraged to keep track of your trials, ideas, erros and successes in logging files - so to say your development diary. If you decide to use this, build and write them in a concise style. If you find those diary documents from a developer before you, use them and feel free to add.


# PREREQUISITES:
1) As prerequisites you have to make sure llvm is installed or to install LLVM (and thus `mlir-tblgen`) (e.g., via Homebrew, apt-get or whatever you prefer), providing the necessary MLIR tools. DO NOT GUESS what the latest version of LLVM is, but rather install it the normal way such that the latest stable version is installed.
2) IMPORTANT: all the steps (in the other .md documents), when they talk about linking something to llvm/mlir is maybe not relevant, depending on the way you installed llvm. Your way of handling the llvm is the correct way, if it works.

# WHAT TO DO
1) Check the plan/ folder. Aside from this very document here are other md files which contain plans. The proposed syntax could VERY likely be outdated or not perfectly correct. Do not slavishly adhere to the exact plan or syntax there if it does not work or does not make sense for your specific setup. You have to check what has already been done and what you have to do to proceed.
2) If you see that the full plan would be too large, you are absolutely encouraged to switch to a small testable part of that plan. Mini steps are great. You are absolutely allowed to alter the exact plan or pivot from you own plans if neccessary, if it is in the spirit of the project.
