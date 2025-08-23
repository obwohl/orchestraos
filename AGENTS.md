### **General Workflow Guidelines**

Work as autonomously as possible. Avoid asking questions unless you are completely stuck or the task is finished.
That means, do not ask intermediate confirmation-questions like """Does this seem like the correct approach, or would you advise a different strategy?""" - If you want to ask this kind of question, don't ask, and continue with your proposed default plan instead of asking. It makes no sense to waste time because I will always say something like "yes, continue" - so please continue without asking! You can take my confirmation as granted.

**0. Test the project as it is  - sanity check. If this fails, fix it**

**1. Understand the Project Context**

* Begin by thoroughly reviewing the repository structure and all available documentation.

**2. Execute and Verify**

* Read and understand the architecture/ documents completely. So read every single of them.
* Then check the repo (**and especially the status_quo.md**) against the to_do.md and decide on what to work next. Don't solve a very large problem in one chunk. Frame and formulate a tiny, testable, achievable step.
* Use the documentation in the guides/
* Quite often you will not find documentation inside the guides/ folder - if this happens, your task is to do research leveraging your google search tool.
* **Crucially**, use self-verification loops (e.g., running tests, checking logs) to confirm your changes are successful.

---

### **Handling Challenges**

* **DO NOT GIVE UP.**
* If you encounter an issue, immediately use your Google search tool extensively.
* If you are still stuck after searching, reframe the problem. Try to miniaturize it and solve a smaller, testable part.
* Repeat this loop at least twice, ensuring each attempt is a new approach. Do not repeat the same errors.

If you are still stuck after multiple attempts, and only then, you may ask for help by writing a comprehensive deep-research question for a separate agent. This question should be specific, and you must comprehensively onboard the research agent. Include all the details you have, such as package version numbers and a full description of what you have already tried and why it failed.

---

### **Finalizing and Committing**

* **Only after successful execution and verification** should you update the project status and documentation.
* **Update `docs/project-status/status-quo.md` to reflect the new state** THIS IS IMPORTANT!
* Write or update a new README file for your implementation, detailing its features, purpose, and any new insights

* Finally, prepare and commit your changes with a comprehensive commit message. ONLY COMMITS WITH THE DOCUMENTATION UPDATES AS OUTLINED!!