# Pitfalls — How Machine Learning Goes Wrong

A catalog of the ways ML projects fail, from subtle technical mistakes to organizational delusions. Roughly ordered from "you'll hit this soon" to "you'll hit this later."

---

## Data leakage — the silent result-killer

The most common way to produce a model that looks great and works terribly.

Leakage means information from your test set accidentally influenced your training process. The model appears to generalize because it secretly saw the answers.

It comes in many forms:

**Direct leakage:** test data ends up in training data. A shuffling bug. A dataset that was pre-split wrong. Common and hard to catch.

**Preprocessing leakage:** you normalize your data using the mean and standard deviation of the entire dataset — including test examples — before splitting. The normalization parameters have "seen" the test data. The correct approach is to compute statistics only on training data, then apply those same statistics to validation and test.

**Temporal leakage:** you're predicting whether a patient will be readmitted to the hospital. Your training data includes features measured after readmission. The model learns to predict something it would never have access to at prediction time.

**Target leakage:** a feature directly encodes the answer. Predicting whether a loan will default, using a feature called "did the customer miss a payment" — which is recorded after default has already happened. The model gets near-perfect accuracy in evaluation and is useless in production where that feature doesn't exist yet.

The consistent pattern: evaluation looks much better than production performance. If your results seem too good, check for leakage.

---

## Not having a baseline

People reach for neural networks immediately. Before you train anything, you should know: what does the dumbest possible approach get?

For MNIST: what accuracy do you get if you just predict "the most common digit in the training set" for every image? What about a linear classifier with no hidden layers? What about nearest-neighbor matching?

A baseline tells you how much your fancy model is actually adding. If your neural network gets 72% and the baseline gets 68%, you should think hard about whether a neural network is the right tool. If the baseline gets 15%, the neural network's 72% is clearly valuable.

This sounds obvious but it is skipped constantly. Researchers publish models that beat a weak baseline when a slightly less weak baseline would have matched them.

---

## Using accuracy as your metric when classes are imbalanced

Imagine you're building a model to detect a rare disease that affects 1% of the population. You train a model, evaluate it, and get 99% accuracy. Incredible.

Your model predicts "healthy" for every single patient.

When one class dramatically outnumbers others, accuracy is a meaningless metric. The model can score well by ignoring the minority class entirely — which is usually exactly the class you care about.

Better metrics for imbalanced problems:
- **Precision:** of everything the model flagged as positive, how many actually were?
- **Recall:** of everything that was actually positive, how many did the model catch?
- **F1 score:** a single number combining both
- **AUC-ROC:** how well does the model separate the two classes across all possible thresholds?

In biology this matters enormously. Pathogenic mutations are rare. Cancer-relevant sequences are rare. Protein binding sites are rare. A model that reports high accuracy on these problems is almost certainly just predicting "normal" for everything.

---

## Distribution shift — training data doesn't match real life

You train a model on data collected in one context and deploy it in another. The training distribution and the deployment distribution are different. The model fails in ways you didn't see coming.

Classic examples:

A model trained to detect pneumonia from chest X-rays learned that the presence of a portable X-ray machine (used only for very sick patients who can't be moved) was a strong predictor of pneumonia. In the training hospital this was a real signal. At a different hospital with different equipment protocols, it was noise — and the model failed.

A skin cancer classifier trained mostly on images from fair-skinned patients performed significantly worse on dark-skinned patients. The distribution of training data didn't match the distribution of patients it was deployed to treat.

An NLP model trained on formal text fails on informal text. A fraud detection model trained on 2019 data fails to detect 2020 fraud because fraud patterns changed.

The fix is partly technical (actively test for distribution shift, monitor in production) and partly organizational (be honest about where your training data came from and whether it actually represents your deployment context).

---

## Overfitting you can't see

You're careful. You have a separate validation set. You check validation accuracy and it looks good. You deploy.

But you ran 200 experiments, checking validation accuracy each time, and deployed the one that happened to score highest. You've inadvertently overfit to the validation set through your own decision-making.

This is called **"researcher degrees of freedom"** or, less charitably, a form of p-hacking. You had 200 chances to get lucky. The one model you selected as "best" might just be the one where the validation set happened to align with its particular failure modes.

The fix: the test set is touched once. The validation set is for making decisions during development. Keep them genuinely separate and don't let validation performance guide your choice of whether to release the final model — that's what the test set is for.

In the research world this has caused widespread reproducibility problems. Models reported at state-of-the-art performance on benchmarks often fail to hold up when tested carefully on harder benchmarks, because researchers optimized for the benchmark implicitly through repeated iteration.

---

## Optimizing for the benchmark instead of the problem

Benchmarks become targets. Targets get gamed.

ImageNet was a useful benchmark for a decade. Eventually the field was so focused on ImageNet performance that models were optimized for its specific quirks — image sizes, class definitions, evaluation procedure — rather than for general visual understanding. Some models that scored highly on ImageNet performed surprisingly poorly on natural photographs taken outside the benchmark's collection conditions.

GLUE and SuperGLUE were benchmarks for language understanding. They got saturated — models reached human-level performance — and researchers later found that those models had learned statistical shortcuts in the specific benchmark format rather than genuine language understanding.

The lesson: the moment a benchmark becomes the goal rather than a measurement tool, it starts misrepresenting what you actually care about. A model with 98% on your benchmark and 60% on your real users' problems is not a good model.

This isn't unique to ML — it's Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure."

---

## Reproducibility — if you can't reproduce it, it didn't happen

ML research has a serious reproducibility crisis. A large fraction of published results cannot be reproduced by other labs, and sometimes not even by the original authors.

Sources of irreproducibility:

**Random seeds.** Training involves randomness — weight initialization, data shuffling, dropout. Different random seeds can produce meaningfully different results. A result that only holds for one lucky seed is not a robust result.

**Missing implementation details.** Papers describe the algorithm but omit the specific hyperparameter tuning, the data preprocessing decisions, the learning rate schedule, the early stopping criteria — all of which significantly affect results.

**Hardware differences.** Floating-point arithmetic is not fully deterministic across different GPUs and even different software versions. Small differences compound.

**Selective reporting.** The experiment that worked gets written up. The twelve that didn't get quietly discarded.

Practical implication: when you read that a model achieves X% accuracy on a benchmark, treat that number with skepticism until you've seen it reproduced independently. For your own work: log everything, set random seeds, re-run your own experiments to confirm they're stable.

---

## "More data will fix it" — sometimes true, often not

More data helps. But:

More data of the same kind won't fix a model that's systematically biased by the kind of data it has. If your training data only contains X-rays from one hospital, more X-rays from that hospital won't fix the distribution shift problem when you deploy at a different hospital.

More data won't fix a wrong model architecture. If your model structurally can't represent the patterns in your data, more data just gives it more examples of the thing it can't learn.

More labeled data won't help if your labels are wrong. Noisy labels — mislabeled training examples — are common and damaging. A model trained on 10% mislabeled data will learn some of those mistakes. More data with 10% mislabeled will have more mislabeled examples.

More data won't fix a bad evaluation metric. If you're measuring the wrong thing, more data makes you more confidently wrong.

The right mental model: more data helps when the bottleneck is statistical — when the model doesn't have enough examples to distinguish signal from noise. When the bottleneck is something else (architecture, labels, distribution, metric), more data is a distraction.

---

## "A bigger model will fix it" — similar

Scaling works, up to a point. But a bigger model trained on the same flawed data learns the flaws more thoroughly. A bigger model with a wrong architecture is a bigger wrong architecture. A bigger model with biased training data encodes the bias more deeply.

There's also a practical version of this mistake: using a huge model when a small one would suffice. Large models are slow to run, expensive to host, hard to update, and harder to debug. If a 10M parameter model gets you 95% accuracy and a 1B parameter model gets you 96%, the 10M model is usually the right choice.

---

## Not understanding your evaluation metric

A surprising number of projects report a metric without fully understanding what it measures.

F1 score: the harmonic mean of precision and recall. Weighted toward whichever is lower. Good for imbalanced classes. Not a good metric if false positives and false negatives have very different costs.

AUC-ROC: the probability that the model ranks a random positive example higher than a random negative example. Threshold-independent. But insensitive to class imbalance — can look good when the model is failing on the minority class.

BLEU score (translation quality): measures n-gram overlap between generated and reference text. Correlates weakly with human judgments of translation quality. Still used because it's easy to compute.

Perplexity (language model quality): measures how surprised the model is by test text. Lower is better. But perplexity on held-out text doesn't directly predict downstream task performance.

The pattern: every metric is a proxy for something you actually care about, and every proxy has failure modes. Knowing what a metric doesn't capture is as important as knowing what it does.

---

## Wrong preprocessing at inference time

You train with specific preprocessing: normalize pixels to [0, 1], resize images to 224×224, lowercase text, remove punctuation. You deploy, and the inference pipeline preprocesses differently — forgets to normalize, uses a slightly different resizing method, leaves punctuation in.

The model performs mysteriously worse in production. The weights are correct but the inputs are in a different distribution than what the model was trained on.

This is extremely common and surprisingly hard to catch. The model doesn't crash — it just gives wrong answers silently. The fix is to treat preprocessing as part of the model, not separate from it, and test the full pipeline end-to-end.

---

## Confusing correlation with causation

A model that predicts well is not a model that understands causes.

A model trained to predict ice cream sales might learn that drowning deaths are a strong predictor. Both correlate with summer. The model isn't wrong — the correlation is real. But using this model to reduce drowning deaths by reducing ice cream sales would be wrong.

This matters in biology especially. A DNA sequence model might learn that a certain motif correlates with gene expression. Is the motif causing expression, or does it happen to co-occur with something that causes expression? The model can't tell you. Only an experiment can.

ML models find correlations. They do this very well. Interpreting those correlations causally requires domain knowledge and experimental validation that the model can't provide.

---

## Treating ML like regular software engineering

In regular software engineering, if the code is correct, the output is correct. There's a direct logical chain from inputs to outputs that you can trace and verify.

In ML, correctness is statistical and approximate. Your model is not "correct" in the way a sorting algorithm is correct. It's probably right, some percentage of the time, on inputs that look like your training data.

This causes problems:

**Unit testing works differently.** You can test that your preprocessing code runs correctly. You can test that your model's output shapes are right. You can't write a unit test that says "and the model is smart" — that's what evaluation on held-out data tells you.

**Code review misses the important stuff.** The model's behavior is determined by the training data and the training process, not (just) the code. You can review the code and miss that the training data is biased, the evaluation metric is wrong, or the hyperparameters cause unstable training.

**Debugging is harder.** A bug in regular code causes an error or wrong output you can trace. A bug in ML causes predictions that are quietly off in ways that may only show up in production or in careful evaluation. The model doesn't know something is wrong; it just learned something you didn't intend.

The necessary shift: you have to think probabilistically about model behavior, invest heavily in data quality and evaluation methodology, and assume that your model has failure modes you haven't found yet.

---

## Pitfalls specific to biology (DNA and protein ML)

**Sequence similarity contamination.** In biology, similar sequences tend to have similar functions. If your training and test sets contain related sequences from the same organism or gene family, your test set is not truly independent. The model can "remember" training examples by similarity. This inflates apparent performance dramatically.

The fix is called a **homology split** — you cluster sequences by similarity and put entire clusters into either train or test, never both. This is harder than random splitting and produces more honest (lower) numbers. Most published models use random splits, which is why results often don't hold up in practice.

**Evolutionary bias.** There are vastly more sequences from E. coli and human than from most of the tree of life. A model trained without accounting for this will perform well on common organisms and fail on the unusual ones. The diversity of life is not represented proportionally in sequence databases.

**Functional annotation quality.** The labels in biological databases are often inferred computationally, not measured experimentally. Training on inferred labels means training on a model's predictions as ground truth. Errors in the database propagate into your model.

**The gap between sequence and function.** Two proteins can have nearly identical sequences but different functions (because one mutation changed a key active site). Two proteins can have completely different sequences but the same function (convergent evolution). Sequence similarity is a noisy proxy for functional similarity, which is often what you actually care about.

---

## The soft ones — organizational and psychological pitfalls

**Using ML because it's interesting, not because it's right.** Sometimes a simple rule works. "Flag any transaction over $10,000" doesn't need a neural network. Before reaching for ML, ask: what does the best possible non-ML solution look like, and how much is ML actually adding?

**No clear definition of success before starting.** "Build a model to detect cancer" is not a success criterion. "Detect cancer with >90% sensitivity and >70% specificity on the held-out test set, with inference under 50ms per sample" is one. Without a clear upfront definition, projects drift, stakeholders disagree about whether it worked, and you can always find a framing that makes results look good.

**Underestimating data collection.** Most ML projects are bottlenecked by data, not algorithms. Getting labeled data is expensive, slow, and harder than it looks. The labels are inconsistent. The data is biased. Edge cases are underrepresented. Projects that budget 10% of their time for data work and 90% for modeling get this backwards.

**Deploying without monitoring.** A model that works today may fail silently next year as the world changes and its training data goes stale. Without monitoring for distribution shift and performance degradation, you won't know it's failing until the damage is done.

**Survivorship bias in papers and tutorials.** You read about the experiments that worked. The field publishes successes. You don't read about the three years that researcher spent on approaches that didn't pan out. This creates a misleading impression that ML is more straightforward than it is, which leads to unrealistic timelines and expectations.
