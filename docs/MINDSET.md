# On Learning This Stuff

Some honest thoughts for when it feels overwhelming.

---

## The intimidation is correct

ML is genuinely hard and the field moves extremely fast. Most content online is written by people who already know it, for other people who already mostly know it. You're not misreading the situation when it feels dense and impenetrable. It is dense and impenetrable if you approach it the wrong way.

The way to not approach it the wrong way: start with something small that runs, change one number, watch what happens. That's it. Everything else follows from that.

---

## On being a slow learner

Slow learners often end up understanding things better than fast ones.

Fast learners frequently build a thin layer of pattern recognition — they can talk about things, use the vocabulary, follow along in a tutorial. But that understanding falls apart under pressure. When something breaks in an unexpected way, they don't have a real model of what's happening underneath.

Slow learners, when they finally get something, tend to really get it. The frustration you feel when something isn't clicking yet is the feeling of actually building a solid model of the thing in your head rather than just memorizing the surface. That's more valuable.

The goal isn't to learn fast. The goal is to understand clearly. Those are different things and it's worth separating them in your head.

---

## On not having ideas

You don't need an idea before you start. This is probably the most common misconception about creative and technical work.

Ideas don't come first. They come from being inside a thing long enough that you notice a gap, get annoyed by a limitation, or wonder "what if I changed this." Every project that turns into something real starts with someone tinkering and noticing something unexpected. The idea was downstream of the doing.

So don't wait to feel inspired. The plan is: pick the next small thing, build it badly, and trust that something interesting will reveal itself. It always does. The people who seem to always have ideas aren't more creative — they're just further along in the doing, so they have more surface area for ideas to attach to.

---

## On "it's better to just use what other people built"

This instinct is correct. Most of the time, using a well-built tool is absolutely the right call. You should not feel bad about this.

The reason to understand how things work underneath isn't to replace the tools — it's so that when something breaks, you know why. When a model gives a weird result, you can reason about what's happening instead of shrugging. When you're choosing between two approaches, you can make an informed decision instead of guessing.

The career value isn't "I can build a transformer from scratch." It's "I understand what a transformer is doing well enough that I'm not confused when it behaves unexpectedly." That understanding comes from having built a small version once, not from doing it professionally forever.

You built a working training loop. That's the small version. You have more intuition now than most people who claim ML experience.

---

## On reading papers

Don't read papers. Seriously.

Papers are written by researchers for other researchers. The format is optimized for peer review, not for understanding. They assume significant prior knowledge, bury the intuition in formalism, and often explain the least interesting parts in the most detail.

The good stuff from every important paper gets turned into a blog post, a YouTube video, or a code example within weeks. The 3Blue1Brown videos and the Karpathy videos in the roadmap — those contain the same knowledge as the papers, in a form designed to actually teach.

Read papers only if you get deep into a very specific topic and find yourself wanting the original source. That's years away and optional even then.

---

## On the career question

Here's what actually differentiates people in this field.

Most people who use ML tools are cargo-culting. They copy a tutorial, swap in their dataset, call it done. They don't know why the learning rate matters or what to do when the loss doesn't go down. They can't debug. They can't adapt when things behave unexpectedly.

You knowing those things — even just at the MNIST level — already puts you ahead of a lot of people who claim ML on their resume. The gap between "I ran a tutorial" and "I understand what's happening" is large, and most people don't cross it.

The next level isn't knowing more techniques. It's getting comfortable with uncertainty and learning to debug. Being able to look at a loss curve behaving strangely and form a hypothesis about why. That skill comes from doing small projects and paying close attention to what happens when things go wrong. It is the most transferable thing you can build.

---

## What to actually do right now

Don't think about DNA models or transformers or protein folding. Run the browser training demo. Change the learning rate to 0.01 and watch the loss bounce around. Change it to 0.0001 and watch it barely move. Set hidden units to 16 and see how much worse accuracy gets. Set epochs to 100 and see if you can spot the point where more training stops helping.

Just play with the thing you already have. Notice what changes and what doesn't. Ask why.

The next idea will come from there. Not from a roadmap.
