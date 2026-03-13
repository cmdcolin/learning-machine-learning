# Why Python?

A fair question. Python is slow, dynamically typed, has a notorious packaging ecosystem, can't run in a browser, and the significant-whitespace thing alienates almost every developer who comes from another language. So why did the most computationally intensive field in software engineering end up built entirely on it?

The answer is mostly historical accident, one killer library, and a feedback loop that became impossible to escape.

---

## Python is not actually running your ML code

This is the key insight that makes everything else make sense.

When you call `np.dot(a, b)` in NumPy, Python is not doing the matrix multiplication. It's calling into a compiled C library (BLAS/LAPACK) that does it. When PyTorch trains a model on a GPU, the computation is happening in CUDA C++. When JAX compiles a function with `@jit`, it's generating XLA machine code.

Python is the remote control. The television is written in C, C++, CUDA, and Fortran.

This is what Python was designed to be — a "glue language" that makes it easy to call into compiled code from other languages. That design decision, made in the early 1990s, turned out to be exactly right for scientific computing, where the hard performance work happens in compiled extensions and humans just need a convenient way to orchestrate it.

So Python's slowness largely doesn't matter for ML. The bottleneck is never Python. It's the GPU.

---

## NumPy was the big bang

In 2006, Travis Oliphant released NumPy — a Python library for fast array operations. It gave Python:

- Arrays that stored data contiguously in memory (unlike Python lists)
- C-speed math over those arrays
- A consistent interface that other libraries could build on

This was the moment Python became viable for science. Before NumPy, Python was a scripting language people used to glue together shell commands. After NumPy, it had a foundation for a real scientific ecosystem.

SciPy built on NumPy. Matplotlib built on NumPy. Pandas built on NumPy. TensorFlow, PyTorch, and JAX all build on NumPy's conventions and array interface. The entire ML ecosystem traces back to one library from 2006.

If NumPy had been written in Ruby or Perl or Lua, ML would probably be in that language instead.

---

## Jupyter notebooks locked it in for research

Around 2011, IPython evolved into what became Jupyter notebooks — an interactive computing environment where you mix code, output, and text in a single document. You write a cell, run it, see the result immediately, write the next cell.

For research this was transformative. Instead of writing a script, running it, reading output, editing it, running it again — you could explore data interactively. You could see your plots inline. You could share a notebook with someone and they could run it and see exactly what you saw.

Jupyter became the standard way researchers share ML experiments. Which meant all the tutorials were notebooks. Which meant all the documentation assumed notebooks. Which meant new people learned ML in notebooks. Which meant Python.

This isn't an accident of technical superiority. It's an accident of timing and adoption. Jupyter happened to be the right tool at the right moment, it was Python, and that mattered more than anything else.

---

## Dynamic typing was actually useful for this specific work

For production software, static types are clearly better. You catch mistakes early, your editor helps you, the code is self-documenting.

But research is different. When you're exploring data you don't fully understand yet, you don't know what types your variables should be. You're loading a dataset and figuring out what's in it. You're trying a function on an array to see what shape it produces. You're running quick experiments where half the code gets thrown away.

Dynamic typing lets you move fast in this mode. You don't have to satisfy a type checker for code you're going to delete in an hour. You can reshape an array and immediately see what comes out without declaring the new type. The REPL (interactive Python prompt) and notebooks work naturally with dynamic types.

This is the same reason JavaScript works well for frontend exploration — you're often figuring out what shape your data is as you go.

Python also added optional type hints in 2014 (now used widely in larger codebases), so the "no types" thing is less true than it used to be. Modern PyTorch code in production usually has type annotations. But by then the ecosystem was already established.

---

## Academia chose Python and that decided everything

Around 2010–2015, universities started teaching Python for data science and machine learning. Before that, the options were MATLAB (expensive, proprietary, engineering-focused), R (statistics-focused, hostile syntax), or nothing accessible.

Python hit the sweet spot: free, general-purpose, beginner-friendly, and now with NumPy/SciPy, capable of real scientific work.

Students learned ML in Python. They graduated, joined companies, and continued using Python. They wrote papers with Python code. Other researchers read those papers and used the Python code. The feedback loop compounded for a decade.

By the time it was obvious that Python had won, the switching cost was enormous. Every model, every tutorial, every library, every paper. You can't switch away from Python without abandoning the entire ecosystem, and the ecosystem is the point.

---

## JavaScript's equivalent moment hasn't happened yet

You might wonder why JavaScript didn't take this role — it runs everywhere, it's the most widely used language in the world, and the browser is the most universal computing platform.

A few reasons:

**Timing.** Node.js launched in 2009, well after the scientific Python ecosystem was established. By then the network effects were already forming.

**No NumPy equivalent.** JavaScript never got a serious numerical computing library at the right moment. TensorFlow.js and ONNX.js exist now, but they're catching up rather than leading.

**The browser sandbox.** For a long time, JavaScript couldn't access GPU compute (WebGL for general computation was awkward, WebGPU only became available recently), couldn't do efficient multi-threading, and had floating-point performance that lagged compiled code badly.

**The type situation cut both ways.** JavaScript's type coercion (the `==` vs `===` mess, `[] + {}`, etc.) gave dynamic typing a bad reputation at exactly the moment Python was getting credit for its cleaner dynamic typing. TypeScript came along eventually but it's an add-on, not the default.

This is changing. WebGPU is here. TensorFlow.js is capable of real work. Transformers.js can run large models in the browser. The future of ML inference (running models, not training them) may genuinely be JavaScript. But training and research will stay Python for a long time because of where all the models live.

---

## The packaging situation

You mentioned Python feels weird, and the packaging ecosystem is probably part of that feeling. pip, conda, venv, virtualenv, poetry, pyenv — there are too many tools and they don't always cooperate. Dependency conflicts are common. `requirements.txt` is fragile. This is a genuine weakness.

It happened because Python grew organically across different communities (scientific computing, web development, scripting) each solving the same problems in different ways. Unlike npm, which was designed for a specific ecosystem from the start, Python's packaging was retrofitted onto a language that predated the idea of package management.

It works. It's not pleasant. It's an artifact of success — so many people needed so many things from Python that multiple incompatible solutions evolved.

---

## The summary

Python is dominant in ML because:

1. It was designed as a glue language, which turned out to be exactly what you need when the real computation happens in C/CUDA
2. NumPy gave it a foundation that everything else built on, and that happened before any competitor had an equivalent
3. Jupyter notebooks became the standard medium for ML research and they were Python
4. Academia adopted it and produced a generation of ML practitioners who only knew Python
5. Network effects: all the models, all the tutorials, all the libraries, all the papers

It's not that Python is the best language for the job. It's that Python got there first, built the ecosystem, and the ecosystem became the moat.

This is not unique to Python. JavaScript is the language of the web for very similar reasons — not because it was the best design, but because it was built into browsers in 1995 and everything that followed had to be compatible with it. Inertia is the most powerful force in technology.
