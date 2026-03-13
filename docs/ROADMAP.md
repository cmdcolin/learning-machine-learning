# Learning Roadmap — From MNIST to DNA Language Models

A personal reference document. Covers everything built in this project, where to go next, and why.

---

## Where you are now

You have a working project that:
- Trains a digit classifier in Python using JAX (the "serious" model: 784→1024→512→10, 1.3M parameters, 60K images, ~98% accuracy)
- Trains a smaller version entirely in your browser using TypeScript (784→128→10, 102K parameters, 500 images, ~85–92% accuracy)
- Runs inference in the browser on drawings you make, using either CPU or WebGPU

You understand, at a working level:
- What a neural network is (a function with many adjustable numbers)
- What training is (finding good values for those numbers by running gradient descent)
- What loss is (a score of how wrong the predictions are)
- What backpropagation is (working out which weights caused the error, using the chain rule)
- What the Adam optimizer does (gradient descent with memory — adapts the step size per parameter)
- What ReLU, softmax, and cross-entropy are
- The difference between CPU and WebGPU inference

---

## The core idea that connects everything

Every ML model in this document — digit classifiers, language models, DNA models — uses the same fundamental loop:

```
1. Run input through the network (forward pass)
2. Measure how wrong the output was (loss)
3. Figure out which weights caused the error (backpropagation)
4. Nudge those weights to reduce the error (optimizer step)
5. Repeat on the next batch
```

The architecture changes. The data changes. The loss function sometimes changes. But this loop is always the same. Once you understand it from MNIST, you understand the skeleton of everything else.

---

## Concepts to solidify before moving on

### Overfitting vs generalization

The model has seen only 500 training images. If you train for too many epochs, it starts memorizing those specific images rather than learning the general concept of what a "7" looks like. Accuracy on training data keeps improving; accuracy on new images plateaus or gets worse. This is **overfitting**.

The fix is more data, regularization (weight decay, dropout), or stopping earlier. The JAX model trains on 60K images with AdamW (weight decay built in), which is why it generalizes much better.

**To see this yourself:** In `TrainView.tsx`, set `TOTAL_EPOCHS = 100` and watch what happens to accuracy after epoch ~30 on only 500 training examples.

### The hyperparameter game

These numbers in the code are called **hyperparameters** — they're not learned by the network, they're chosen by you:

| Hyperparameter | File | Default | What happens if you change it |
|---|---|---|---|
| Hidden layer size | `train.ts:11` | 128 | Bigger = more expressive but slower and needs more data |
| Learning rate | `TrainView.tsx:17` | 0.001 | Too high = loss bounces/diverges; too low = very slow convergence |
| Batch size | `TrainView.tsx:16` | 32 | Small = noisy updates, fast; Large = smooth updates, fewer steps/epoch |
| Epochs | `TrainView.tsx:15` | 20 | More = risk of overfitting; fewer = underfitting |

Most of practical ML engineering is choosing these well. There is no formula — you experiment and watch what the loss curve does.

### What the loss curve shape tells you

```
Loss falling steeply then leveling off → normal, healthy training
Loss bouncing up and down wildly      → learning rate too high
Loss barely moving                    → learning rate too low, or network too small
Loss falling, then rising again       → overfitting
Loss stuck high from the start        → something broken (bug, bad init, wrong lr)
```

---

## What to try next, in order

### 1. Fashion-MNIST

**Why:** Identical format to MNIST (28×28 grayscale, 10 classes, 60K images) but clothing items — t-shirt, sneaker, bag, etc. Harder because classes look more similar. Zero code changes needed, just swap the data.

**What you'll learn:** The same model that gets 98% on digits gets ~87% on clothing. That gap is real and it tells you something about problem difficulty that MNIST's easiness hides.

**Data:** https://github.com/zalandoresearch/fashion-mnist — same binary format as MNIST.

### 2. 2D toy problems

**Why:** Before tackling harder image datasets, spend time with problems you can actually see. A 2D classification problem lets you watch the decision boundary update in real time.

**What you'll learn:**
- The XOR problem can't be solved by a single layer (no straight line separates it) — this is exactly why hidden layers exist
- Adding neurons widens the boundary; adding layers makes it more complex
- The shape of the boundary is determined entirely by the learned weights

**Where:** [TensorFlow Playground](https://playground.tensorflow.org) — no setup, runs in browser. Try XOR with 0 hidden layers first (watch it fail), then add one layer of 4 neurons (watch it succeed).

### 3. CIFAR-10 with an MLP

**Why:** 32×32 color images of 10 object categories (planes, cars, dogs, etc.). With a flat MLP you'll hit a ceiling around 50–55% no matter what you try. That ceiling is the lesson — it motivates convolutional networks.

**What you'll learn:** Images have spatial structure. Flattening a 32×32 grid destroys the fact that neighboring pixels are related. An MLP has no way to learn "this pattern appears in the top-left corner" separately from "this pattern appears in the bottom-right corner." CNNs solve this.

### 4. CNNs

**Why:** The key architectural breakthrough for image recognition. A convolutional layer slides a small filter (e.g. 3×3) across the image, looking for local patterns. The same filter works regardless of where in the image the pattern appears.

**What you'll learn:**
- Convolution = the filter is just another set of learned weights, same backprop
- Pooling = downsample the spatial dimensions while keeping what was found
- CNN on CIFAR-10 → ~90% accuracy vs ~55% with MLP

**In JavaScript:** [TensorFlow.js CIFAR-10 example](https://github.com/tensorflow/tfjs-examples/tree/master/cifar10-core) runs in the browser.

**To read:** [CNN Explainer](https://poloclub.github.io/cnn-explainer/) — interactive visualization of what each filter learns.

---

## Language models

### The one-sentence version

A language model does one thing: predict the next token given everything before it. The training signal is cross-entropy loss — exactly what you used for MNIST. The difference is that instead of "which digit class?", the question is "which character comes next?".

### How it connects to what you built

In MNIST: one image → one label (10 classes).
In a language model: one sequence of N tokens → N predictions, each "what comes next?". Every position in the sequence is a training example. A sequence of 1000 characters gives you 999 training examples from a single input.

The loss means the same thing. The optimizer is the same. Backpropagation is the same. What changes is the architecture and how you represent the input.

### Tokens and embeddings

In MNIST the input was 784 raw floats. In a language model, the input is a sequence of discrete symbols (characters, words, DNA bases). Each symbol is first converted to a vector of floats using an **embedding table** — essentially a lookup table where each token gets its own learned row. Those vectors are then what gets processed by the network.

```
"A" → [0.2, -0.5, 1.1, ...]   (a learned vector for A)
"T" → [0.8,  0.3, -0.2, ...]  (a learned vector for T)
```

The network learns the embeddings during training. Similar tokens end up with similar vectors.

### RNNs and why they were replaced

The first approach to sequences was **Recurrent Neural Networks (RNNs)**: process one token at a time, maintain a hidden state that carries information forward. The problem is that the hidden state is a fixed-size vector — information from early in the sequence gets compressed and overwritten as the sequence gets longer. An RNN reading token 800 has largely forgotten what happened at token 50.

LSTMs (Long Short-Term Memory) improved this with learned "gates" that decide what to remember and what to forget. Better, but still fundamentally limited by the bottleneck of compressing history into a fixed vector.

### Transformers and attention

Transformers (2017) solved the long-range problem by letting every position in the sequence directly attend to every other position.

**Attention in plain English:** For each token in the sequence, compute a weighted average of all other tokens' representations. The weights (attention scores) are learned — the network decides which tokens are relevant to which. A regulatory element 500 base pairs upstream can directly influence the representation of a gene downstream, without having to survive being compressed through a bottleneck.

This is why transformers dominate everything now — language, images, proteins, DNA. The architecture scales well, parallelizes well on GPUs, and handles long-range dependencies that RNNs couldn't.

**To understand attention visually:** [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the best explanation, no code required.

**To build one from scratch:** [Karpathy's nanoGPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) (~2 hours). He builds a working GPT-style transformer in ~200 lines of Python. Worth doing before touching DNA models — you'll understand what you're using.

### The learning path through LMs

```
makemore (bigram) → makemore (MLP) → makemore (transformer) → nanoGPT → DNA
```

[makemore playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — Karpathy builds each of these from scratch, one video per architecture.

---

## DNA language models

### Why DNA is a natural fit for language models

DNA is a sequence of 4 characters: A (adenine), T (thymine), G (guanine), C (cytosine). That's it. Vocabulary size 4, vs ~50,000 for English word-level models.

A language model trained on DNA learns the statistical structure of genomic sequences — which patterns tend to appear together, what protein-coding regions look like, what regulatory elements look like, how sequences vary across species. It doesn't know any biology going in. It learns it from the sequences.

```
DNA:     ...ATGCGATCGATCG...   (alphabet size: 4)
English: ...the cat sat on...  (alphabet size: ~50K words)
```

Same model. Same training. Different string.

### What the model is actually learning

Some concrete things a DNA language model learns to predict:

- **Codons:** DNA is read in triplets (codons), each encoding one amino acid. ATG always starts a protein. TAA, TAG, TGA always stop it. A good model learns this from data without being told.
- **Reading frames:** The same stretch of DNA encodes completely different proteins depending on where you start reading (offset by 0, 1, or 2 bases). The model has to track this implicitly.
- **Regulatory elements:** Short sequences upstream of genes (promoters, enhancers) that control when and how much the gene is expressed. These can be hundreds or thousands of bases away from the gene they regulate — a long-range dependency that attention handles naturally.
- **Conservation:** Sequences that are identical across many species are usually functionally important (evolution would have changed them if they weren't). Multi-species training lets the model learn this.

### Tokenization strategies for DNA

How you split DNA into tokens matters more than you might expect.

**Character-level (simplest):** Each base is one token. Vocabulary = {A, T, G, C, N}. Simple, but sequences are very long.

**k-mer tokenization:** Split into overlapping chunks of length k. k=6 gives 4^6 = 4,096 possible tokens. Used by the original DNABERT. Captures local patterns but the vocabulary grows exponentially with k.

**BPE (Byte Pair Encoding):** The same tokenization used by GPT models on English text. Learns which substrings appear frequently and merges them into tokens. DNABERT-2 uses this and finds it significantly better than k-mers — the model learns biologically meaningful chunks without them being hand-specified.

### The major DNA language models

**[DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2)** (2023)
Best starting point. BERT-style (bidirectional — predicts randomly masked tokens rather than just next token, so it has context from both sides). BPE tokenization. Pre-trained on 135 species. Has fine-tuned versions for specific tasks. The paper explains the BPE tokenization argument clearly.

**[Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer)** (InstaDeep/EMBL, 2023)
Family of models 500M–2.5B parameters, trained on 3,202 human genomes and 850 other species. Available on Hugging Face with fine-tuned versions for 18 genomics tasks (promoter prediction, splice site detection, chromatin accessibility, etc.).

**[HyenaDNA](https://github.com/HazyResearch/hyena-dna)** (2023)
Handles sequences up to 1 million base pairs in a single context window. Standard transformers have a quadratic cost in sequence length (attention over N tokens costs N²), which limits them to tens of thousands of bases. HyenaDNA uses a different operator (Hyena) that scales linearly. Essential for whole-chromosome-scale modeling.

**[EVO](https://github.com/evo-design/evo)** (Arc Institute, 2024)
7B parameters trained on 2.7 million prokaryotic genomes. Can do generation (design new sequences) not just prediction. Currently state-of-the-art for predicting the effect of mutations. The paper has a very readable introduction that's worth reading even if you don't use the model.

**[Caduceus](https://github.com/kuleshov-group/caduceus)** (2024)
Bidirectional Mamba architecture for DNA. Mamba is a newer sequence model (like Hyena — sub-quadratic scaling) and Caduceus extends it to handle both strands of the double helix simultaneously, which is biologically correct (both strands carry information).

### What these models can do

**Trained on raw sequences, zero biological labels:**
- Predict whether a sequence is in a coding region or not
- Identify likely splice sites
- Predict chromatin accessibility (whether DNA is "open" and accessible for transcription)
- Cluster similar regulatory elements

**Fine-tuned on labeled data:**
- Predict the effect of a single nucleotide mutation on protein function
- Classify promoter sequences by strength
- Identify transcription factor binding sites
- Predict gene expression levels from DNA sequence

**Generation (EVO, some others):**
- Design novel protein-coding sequences with desired properties
- Generate plausible synthetic genomes

### The long-range dependency problem in genomics

This is why HyenaDNA and similar models matter. In English, most dependencies are short — a pronoun refers to a noun from a few sentences ago. In DNA:

- An **enhancer** (regulatory element) can activate a gene from 1 million base pairs away
- **Topologically associating domains (TADs)** — 3D folding structures that bring distant DNA regions into physical proximity — span hundreds of thousands of bases
- **Structural variants** (large deletions, inversions, translocations) have effects that span millions of bases

Standard BERT-style models have a maximum context of ~512 tokens. Nucleotide Transformer goes to ~6,000. HyenaDNA goes to 1,000,000. The right context length depends on the biological question you're asking.

### Running DNA models in JavaScript

All the major models above are on Hugging Face. [Transformers.js](https://huggingface.co/docs/transformers.js) can run many Hugging Face models directly in the browser or in Node.js with no Python required:

```javascript
import { pipeline } from '@xenova/transformers';
// Most encoder models (BERT-style) work this way
const classifier = await pipeline('text-classification', 'model-name');
const result = await classifier('ATGCGATCGATCG');
```

Not all DNA models have Transformers.js-compatible exports yet, but this is improving quickly.

---

## Protein language models

### How proteins relate to DNA

DNA is instructions. Proteins are the machines that do the actual work in a cell — enzymes, receptors, structural components, motors. The path from DNA to protein is:

```
DNA sequence (ATGCGT...)
    ↓  transcription
RNA sequence (AUGCGU...)
    ↓  translation (read in triplets called codons)
Amino acid sequence (Met-Arg-...)
    ↓  folding (the sequence determines the 3D shape)
Protein structure (the shape determines the function)
```

A **protein language model** operates on amino acid sequences — not DNA. The vocabulary is 20 standard amino acids, each represented by a single letter (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y). So vocabulary size 20, vs 4 for DNA and 50K for English.

### Why protein LMs are a bigger deal than they might seem

Proteins are harder than DNA in one critical way: **sequence determines 3D shape, and shape determines function**. Two proteins with very different sequences can fold into nearly identical shapes and do the same job. Two proteins with very similar sequences can fold completely differently. The mapping from sequence to structure to function is deeply nonlinear and was unsolved for 50 years.

AlphaFold2 (2021) effectively solved protein structure prediction. Language models built on top of protein sequences have since unlocked:

- Predicting whether a mutation makes a protein more or less stable
- Designing new proteins with desired functions from scratch
- Predicting which proteins interact with each other
- Understanding evolutionary relationships from sequence alone

This is arguably the highest-impact application of language models in science right now.

### The major protein language models

**[ESM-2](https://github.com/facebookresearch/esm)** (Meta AI, 2022)
The most widely used protein LM. Trained on 250 million protein sequences from UniRef. Released in multiple sizes (8M to 15B parameters). BERT-style masked language modeling — predict randomly masked amino acids given the rest of the sequence. The learned representations encode structural and functional information surprisingly well.

ESM-2 embeddings (the vector representation of a protein sequence) are now used as features in nearly every protein ML pipeline. You take a sequence, run it through ESM-2, and the output vectors capture things like secondary structure, solvent accessibility, and evolutionary conservation — without those labels ever appearing in training.

**[ESMFold](https://github.com/facebookresearch/esm)** (Meta AI, 2022)
Uses ESM-2 embeddings to predict 3D protein structure. 1000× faster than AlphaFold2 with slightly lower accuracy — fast enough to fold an entire proteome in hours. Same repo as ESM-2.

**[ProtTrans](https://github.com/agemagician/ProtTrans)** (2021)
A family of protein LMs (BERT, XLNet, Albert, T5 variants) trained on protein sequences. Good baseline models that are well-studied. Useful if you want to compare architectures.

**[ESM3](https://github.com/evolutionaryscale-ai/esm)** (EvolutionaryScale, 2024)
Multimodal — reasons over sequence, structure, and function simultaneously. You can prompt it with partial structure and ask it to complete a sequence, or prompt it with sequence and get structure back. 98 billion parameters in the largest version. The "generative" model for proteins.

**[Progen2](https://github.com/salesforce/progen)** (Salesforce, 2023)
Autoregressive (GPT-style, predicts next amino acid rather than masked tokens). Can generate novel protein sequences conditioned on a functional tag. Useful for protein design tasks where you want to generate sequences in a specific protein family.

### Protein vs DNA: what's different

| | DNA | Protein |
|---|---|---|
| Alphabet | 4 (ATGC) | 20 amino acids |
| What it encodes | Instructions | Functional machines |
| Key challenge | Long-range regulation | Sequence→structure→function |
| Typical length | Thousands–millions of bases | Hundreds–thousands of residues |
| Ground truth labels | Gene databases, functional assays | Structure databases (PDB), mutational studies |
| Main models | DNABERT-2, HyenaDNA, EVO | ESM-2, ESMFold, ESM3 |

### The structure prediction breakthrough

Before AlphaFold2, predicting protein structure from sequence was an unsolved problem that had resisted 50 years of effort. The structure determines the function, so not being able to predict it was a massive bottleneck.

AlphaFold2 (DeepMind, 2021) solved it to near-experimental accuracy for most proteins. It uses attention heavily, similar to a transformer, but with specialized geometric reasoning layers that understand 3D coordinates. The paper is [here](https://www.nature.com/articles/s41586-021-03819-2) and the code is [here](https://github.com/google-deepmind/alphafold).

What's important to understand: AlphaFold2 solved *prediction* (given a sequence, what's the structure). The open problem now is *design* (given a desired function, what's a sequence that would fold correctly to achieve it). ESM3, ProteinMPNN, and RFdiffusion are working on this.

### Useful tools

**[Hugging Face protein models](https://huggingface.co/models?search=protein)** — ESM-2 and ProtTrans models are all available here with Transformers.js-compatible exports for some of them.

**[UniProt](https://www.uniprot.org)** — the main protein sequence database. 200M+ sequences. Swiss-Prot (manually reviewed) and TrEMBL (computationally predicted). Most protein LMs trained on this.

**[PDB (Protein Data Bank)](https://www.rcsb.org)** — 200K+ experimentally determined protein structures. The ground truth AlphaFold trained on.

**[AlphaFold Database](https://alphafold.ebi.ac.uk)** — ESMFold/AlphaFold predictions for virtually the entire known proteome (~200M proteins). You can look up structure predictions for any protein sequence.

### The connection between DNA and protein LMs for your purposes

DNA LMs and protein LMs are often used together in genomics pipelines:

```
Genomic DNA sequence
    ↓  DNA LM (DNABERT-2, HyenaDNA)
    → find coding regions, predict which genes are expressed
    ↓  translate DNA → amino acid sequence
    ↓  Protein LM (ESM-2)
    → predict protein structure, function, interaction partners
    → predict effect of mutations on protein stability
```

For practical work in computational biology, you'll likely use both. Start with ESM-2 for proteins (it's the most accessible and widely used) and DNABERT-2 for DNA.

### Papers to read

- [ESM-2: Language models of protein sequences at the scale of evolution enable accurate structure prediction](https://www.biorxiv.org/content/10.1101/2022.07.20.500902) — the main ESM-2 paper, readable introduction
- [Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences](https://www.pnas.org/doi/10.1073/pnas.2016239118) — the original ESM paper explaining why unsupervised training on sequences learns structural information
- [AlphaFold2 paper](https://www.nature.com/articles/s41586-021-03819-2) — landmark paper, the introduction and results sections are accessible even without following all the architecture details
- [Large language models in the life sciences — Nature Methods 2024](https://www.nature.com/articles/s41592-024-02249-2) — a recent survey covering both DNA and protein LMs together, good overview of the full field

---

## Reading list, roughly ordered

### Start here
1. [3Blue1Brown Neural Networks playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — 4 videos, watch before anything else
2. [Neural Networks and Deep Learning — Michael Nielsen](http://neuralnetworksanddeeplearning.com) — free, builds MNIST from scratch in plain English
3. [Karpathy makemore series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — builds up to a transformer from scratch

### Go deeper on transformers
4. [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/) — best visual explanation of attention
5. [Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — ~2 hours, builds a working GPT
6. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original transformer paper, dense but worth reading after the above

### DNA/genomics ML
7. [Deep learning for genomics — Eraslan et al., Nature Methods 2019](https://www.nature.com/articles/s41592-019-0551-3) — good survey of the field before transformers dominated
8. [DNABERT-2 paper](https://arxiv.org/abs/2306.15006) — explains the tokenization argument clearly
9. [EVO paper](https://www.biorxiv.org/content/10.1101/2024.02.27.582234) — readable introduction, good motivation for why DNA LMs matter
10. [Genomics in the age of large language models — Nature Genetics 2023](https://www.nature.com/articles/s41588-023-01573-3) — overview of open problems in the field

### Protein ML
11. [Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences](https://www.pnas.org/doi/10.1073/pnas.2016239118) — the original ESM paper, explains why sequence training learns structure
12. [ESM-2 paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902) — the main practical protein LM paper, accessible introduction
13. [AlphaFold2 paper](https://www.nature.com/articles/s41586-021-03819-2) — the landmark structure prediction paper; read at least the introduction and results
14. [Large language models in the life sciences — Nature Methods 2024](https://www.nature.com/articles/s41592-024-02249-2) — covers both DNA and protein LMs together, good current overview of the whole field

### Reference books (go deeper when ready)
11. [Dive into Deep Learning — d2l.ai](https://d2l.ai) — free, every concept has runnable code
12. [Deep Learning — Goodfellow, Bengio, Courville](https://www.deeplearningbook.org) — free, academic, thorough
13. [Adam paper — Kingma & Ba 2014](https://arxiv.org/abs/1412.6980) — 9 pages, directly explains what `adamStep()` in this repo does
14. [Speech and Language Processing — Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slpdraft/) — free, standard NLP textbook, good chapters on neural LMs

### Interactive tools
- [TensorFlow Playground](https://playground.tensorflow.org) — watch a network train on 2D problems in real time
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) — visualizes what CNN filters learn
- [Distill.pub](https://distill.pub) — ML research as interactive web articles

---

## The big picture

```
MNIST (this repo)
  ↓ same loss, same backprop, harder data
Fashion-MNIST / CIFAR-10 MLP
  ↓ same data, new architecture
CNNs on images
  ↓ same transformer architecture, text data
Character-level LM (makemore)
  ↓ scale up, add more data
GPT-style LM
  ↓                              ↓
  ↓  same model, DNA alphabet    same model, protein alphabet
  ↓                              ↓
DNA language model           Protein language model
(DNABERT-2, HyenaDNA, EVO)   (ESM-2, ESM3, Progen2)
  ↓                              ↓
Find regulatory elements      Predict 3D structure (ESMFold)
Predict gene expression       Predict mutation effects
Identify functional variants  Design novel proteins
  ↓                              ↓
  └──────────────┬───────────────┘
                 ↓
     Use both together in a pipeline:
     DNA LM finds coding regions →
     translate to amino acids →
     protein LM predicts structure and function
```

Everything in this chain runs the same core loop (forward → loss → backward → update). The architecture gets more sophisticated, the data gets more specialized, but the foundation you built understanding MNIST transfers directly to all of it.
