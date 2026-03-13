# Notebooks vs Real Engineering

Your instinct is correct. Notebooks are a genuinely problematic way to build anything meant to last, and the ML industry has a complicated relationship with this fact.

---

## What notebooks are good for

Exploration. You have a new dataset and you don't know what's in it. You want to load it, look at some rows, plot a histogram, try a transformation, see what happens. The notebook format — run a cell, see output immediately, run the next cell — is genuinely well-suited to this mode of work.

This is what RStudio was designed for too, and it's a legitimate use case. When you're figuring out what the data looks like and what you want to do with it, the interactive loop is valuable. The problem is when this exploration environment gets mistaken for a software engineering environment.

---

## The specific problems

**Hidden state.** This is the worst one. In a notebook, variables persist between cells. You can run cell 7, then delete it, then run cells 1–6 and 8–10 in order — and cell 10 can still use the variable that cell 7 set, even though cell 7 no longer exists. The notebook appears to work but is silently depending on something invisible.

The classic disaster: a notebook that runs fine for the person who wrote it, in the order they happened to run it, but fails for anyone else running it top-to-bottom. The "restart kernel and run all cells" button exists specifically because this is so common, but it frequently reveals that the notebook is broken in ways nobody noticed.

**Version control is a nightmare.** A Jupyter notebook is a JSON file that contains your code, all your outputs (including base64-encoded images), and execution counts. Running a cell changes the file even if you didn't change any code. A git diff of a notebook is almost unreadable. Reviewing a notebook pull request is nearly impossible.

**No refactoring.** Modern editors can rename a variable across a whole codebase, extract a function, find all usages. In a notebook, you're doing this by hand. Most notebook environments have minimal refactoring support because the global-mutable-state model makes it hard to reason about what a "rename" would affect.

**Testing is not a thing.** You can't write unit tests for notebook cells in any natural way. The standard engineering practice of "write a function, write a test for it, know it works" doesn't apply. The notebook itself is the test — which means the test is manual, nonrepeatable, and dependent on hidden state.

**Imports and dependencies are scattered.** In a normal Python module, all imports are at the top. In notebooks, imports end up wherever someone needed them, cells get moved around, and figuring out what a notebook actually depends on requires reading the whole thing.

---

## Why the ML world uses them anyway

Because they're genuinely good for the exploration phase, and ML has a lot of exploration phase.

When a researcher is trying ten different ways to preprocess a dataset, or comparing five model architectures, or doing exploratory data analysis before deciding what approach to take — the interactive, visual, incremental notebook style is useful. You see your plots inline. You try something, it doesn't work, you try something else, the history is right there.

The problem is cultural: this exploration environment became the production environment. People would explore in a notebook, get something working, and then deploy the notebook. Or clean it up slightly and call it "production code." Or paste the notebook cells into a script without restructuring them into proper functions.

This is the original sin. Notebooks are fine as scratch paper. They become a problem when the scratch paper is the final product.

---

## What production ML engineering actually looks like

The companies doing ML engineering seriously treat ML code like software. That means:

**Plain Python files, not notebooks.** Your model definition is a `.py` file with proper functions and classes. Your training script is a `.py` file you can run from the command line. Your data preprocessing is a module you can import and test.

**Notebooks only for exploration and presentation.** Some teams have an explicit rule: notebooks are for initial exploration and for final reporting/visualization. Anything that runs in production is a Python module.

**Testing.** You test your data preprocessing functions. You test that your model's output shape is what you expect. You test that your loss function returns sensible values on known inputs. `pytest` is the standard tool.

**Experiment tracking.** Tools like [Weights & Biases](https://wandb.ai) or [MLflow](https://mlflow.org) replace the notebook as the place you track "I tried this hyperparameter, got this accuracy, here's the loss curve." You log metrics programmatically from your training script and view them in a dashboard. This is version-controllable and reproducible in a way notebooks aren't.

**Pipelines.** Tools like [DVC](https://dvc.org) (Data Version Control) or [Metaflow](https://metaflow.org) treat ML workflows as DAGs of steps with explicit inputs and outputs. You define "preprocess data → train model → evaluate → deploy" as a pipeline where each step is a function, inputs and outputs are tracked, and steps are only rerun when their inputs change. This is the ML equivalent of a build system.

---

## The comparison to RStudio

RStudio is more like a real IDE than Jupyter. It has a file browser, a proper editor, a console, and an environment inspector — and importantly, the distinction between "source file" and "interactive console" is clearer. You write functions in `.R` files and test them interactively in the console. The file is the canonical thing; the console is the scratch space.

Jupyter blurs this distinction almost completely. The notebook is both the scratch space and the file, simultaneously. That blurring is its appeal and its problem.

The R community also has [Targets](https://books.ropensci.org/targets/) — a pipeline framework very similar to what DVC and Metaflow do for Python. The idea of defining ML workflows as reproducible dependency graphs rather than linear notebook execution is one the R community was arguably ahead of Python on.

Quarto (which evolved out of RMarkdown) is also more honest about the "this is a document, not a software module" framing than Jupyter is. A Quarto document is explicitly a report that happens to contain executable code. A Jupyter notebook pretends it's both a report and a software artifact at the same time.

---

## Practical advice

If you end up writing any ML code:

Write your actual logic in `.py` files. Import them, test them, version control them normally. Use a notebook only if you're making a chart or exploring data you don't understand yet.

If you inherit notebook code, the first thing to do is "restart kernel and run all cells" to see if it actually works, then extract the meaningful parts into functions in `.py` files and write tests for them.

The notebook is fine. The notebook as a software engineering artifact is not.

---

## The one thing notebooks genuinely do better

Sharing results with non-engineers. A Jupyter notebook with inline plots, explanatory text, and code is an excellent format for communicating findings to a product manager, a biologist, or a client who wants to understand what happened without running code themselves. This is the use case where the format earns its keep.

For that purpose — communication, not engineering — it's good. The mistake is letting it leak out of that role.
