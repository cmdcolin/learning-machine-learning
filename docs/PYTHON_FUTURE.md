# Efforts to Fix Python

Python has real problems — slow, no enforced types, a packaging disaster, and a fundamental threading limitation that's been a known issue for 30 years. Here's what's actually being done about each one.

---

## Tooling: the Astral takeover

The most TypeScript-like story happening in Python right now is a company called **Astral**. They're systematically replacing the fragmented Python toolchain with fast, unified tools written in Rust.

**[Ruff](https://github.com/astral-sh/ruff)** — a linter and formatter that replaces flake8, pylint, isort, pyupgrade, and about a dozen other tools simultaneously. It's 10–100× faster than what it replaces because it's Rust. Adoption has been extremely fast — major projects switched within months of release. It's now the default linter for most new Python projects.

**[uv](https://github.com/astral-sh/uv)** — a package manager that replaces pip, venv, virtualenv, pip-tools, and pipenv. Also Rust. Installing packages is 10–100× faster. Handles virtual environments cleanly. Resolves dependencies correctly. If you use Python today, use uv — it makes the packaging situation actually tolerable.

Together these two tools are doing what the TypeScript compiler did for JavaScript: replacing a chaotic ecosystem of overlapping half-solutions with one fast, well-designed thing. The difference from TypeScript is that Astral is a company doing this from outside the core language team, not a language redesign. Python itself isn't changing — just everything around it.

---

## Types: getting there, slowly

Python type hints exist (added 2014) and the tooling is real, but the culture never fully committed the way JavaScript did with TypeScript.

**[Pyright](https://github.com/microsoft/pyright)** — Microsoft's type checker for Python, the same engine that powers the Python extension in VS Code. Fast, strict, and the best option if you want real type safety. If you set it to strict mode it'll catch a lot.

**[mypy](https://mypy-lang.org)** — the original Python type checker, now also Microsoft-funded. Slower than pyright, more conservative. Still widely used.

**[basedpyright](https://github.com/DetachHead/basedpyright)** — a stricter community fork of pyright that closes loopholes Microsoft left open for backward compatibility. For people who actually want types enforced.

The problem isn't the tooling — it's that it's opt-in and the scientific/ML culture never made it mandatory. You can write a full PyTorch training script with zero type annotations and nothing complains. Jupyter notebooks actively resist types because you're writing exploratory throw-away cells.

The Python type system also has genuine limitations. Because Python is so dynamic, the type checker often has to give up and mark something as `Any`, which infects everything it touches. TypeScript can enforce types deeply because JavaScript's runtime behavior is relatively predictable. Python's runtime does enough surprising things that the type checker sometimes can't reason about it.

Newer Python versions have made the type syntax nicer:
- 3.9: `list[int]` instead of `List[int]` from the typing module
- 3.10: `int | None` instead of `Optional[int]`
- 3.12: cleaner type alias syntax

It's improving. It's just not TypeScript.

---

## Speed: multiple serious efforts

**CPython getting faster** — The core Python interpreter (CPython) has gotten meaningfully faster in recent versions. Python 3.11 was about 25% faster than 3.10. Python 3.12 continued gains. There's a Microsoft-funded project called "Faster CPython" that's been systematically speeding up the interpreter since 2021. It's real progress, just slow by compiled language standards.

**[PyPy](https://www.pypy.org)** — an alternative Python interpreter with a JIT compiler. Often 5–10× faster than CPython for pure Python code. The problem: it's not 100% compatible with all C extensions, and since most ML libraries are C extensions (NumPy, PyTorch), compatibility issues make it hard to use in practice. Useful for certain kinds of Python programs, not really the ML ecosystem.

**[Numba](https://numba.pydata.org)** — JIT compiler specifically for numerical Python. You put `@numba.jit` on a function that does heavy numerical computation and it compiles to fast machine code. Works well for the kinds of loops that are common in scientific code. Used in cases where NumPy isn't vectorizable.

**[Mypyc](https://mypyc.readthedocs.io)** — compiles type-annotated Python to C extensions. If you have type annotations, mypyc can produce native code. Used to compile mypy itself, making it 4× faster. Not widely adopted yet but interesting direction.

**The GIL removal** — see below, this is the big one.

---

## The GIL: a 30-year-old problem finally being solved

The GIL (Global Interpreter Lock) is a lock inside the Python interpreter that prevents more than one thread from executing Python code at the same time. Yes, really. Python has had threads for 30 years and they've never been able to run in true parallel.

The reason it exists: Python uses reference counting for memory management. Reference counting requires updating counters when objects are created and destroyed. Without the GIL, multiple threads modifying those counters simultaneously would corrupt memory. The GIL was the simplest solution in 1992.

The consequence: Python programs that use threading can't use multiple CPU cores for CPU-bound work. For ML training, this doesn't matter much because the real work is on the GPU. But for CPU-bound Python code, you're limited to one core.

**Python 3.13 (2024)** introduced an experimental no-GIL build. You can compile Python without the GIL and get true multi-threading. It's optional for now — removing it entirely would break extension modules that assumed the GIL existed. The plan is to make it the default eventually.

This is genuinely a big deal structurally. It's been a fundamental limitation of Python for its entire existence and it's finally being addressed.

---

## Mojo: the wildcard

**[Mojo](https://www.modular.com/mojo)** is the most ambitious effort, and the most speculative.

Chris Lattner — the person who created LLVM (the compiler infrastructure that powers Clang, Swift, Rust, and many others) and created Swift at Apple — left Apple, did a stint at Google Brain, and then started a company to build a new language for AI.

Mojo is designed as a superset of Python: valid Python code is valid Mojo code. But Mojo compiles to native machine code, supports SIMD and low-level memory control, has a proper type system, and is designed to run on heterogeneous hardware (CPUs, GPUs, custom accelerators). Claimed benchmarks show it running certain numerical operations 35,000× faster than Python.

The pitch: you write Python, you get compiled-language performance. Gradually add types and lower-level control where you need speed.

It's early. The ecosystem doesn't exist yet. It's not open source in the traditional sense (the compiler is proprietary though the standard library is open). Whether it can actually displace Python in ML is unclear — the switching cost of the Python ecosystem is enormous and it's the same network effects problem that keeps Python dominant.

But it's the most serious attempt to build a better Python for ML that has the technical credibility to potentially succeed. Worth watching.

---

## The honest assessment

The tooling story (Ruff, uv) is genuinely good news and happening fast. If you're writing Python today, these tools make the experience meaningfully better.

The types story is partial. The tools are real and pyright in strict mode actually works. But the culture in ML and data science hasn't fully committed, so you're often working with libraries that have incomplete type stubs or just punt to `Any`.

The speed story doesn't matter that much for ML because Python isn't the bottleneck — the GPU is. For other kinds of Python programs it matters more.

The GIL story is genuinely significant long-term but won't change your daily experience for a while.

Mojo is interesting and could matter in 3–5 years. Right now it's a bet on Chris Lattner's track record.

The comparison to TypeScript is apt though. TypeScript happened fast and completely because one company made a big push, Angular forced adoption, and the community was motivated. Python's improvements are happening through many separate efforts without a coordinating forcing function. It's getting better, just slower and less coherently than JavaScript's transformation.
