# Label Noise and Grokking Dynamics

Code for **"Investigating the Impact of Low to Moderate Level Label Noise on Neural Network Grokking Dynamics,"** published in the *Journal of High School Sciences (JHSS)*. Research mentored by **Prof. Shaddin Dughmi** (University of Southern California).

---

## Overview

**Grokking** is a surprising behavior in neural network training: a model first memorizes its training data (high training accuracy, poor test accuracy), and then, often long afterward, suddenly *generalizes*, with test accuracy jumping to near-perfect.

Real-world datasets are rarely clean. Some fraction of labels are almost always wrong. This project asks:

> **How does a small-to-moderate amount of label noise (0–15% corrupted labels) change when, and whether, a neural network groks?**

## Key Finding

**Time-to-grok grows exponentially with the share of corrupted training labels.** Even modest amounts of label noise substantially delay generalization, and the delay compounds as noise increases.

<!-- Add your main results plot here, e.g.: -->
<!-- ![Time-to-grok vs. label noise](figures/time_to_grok_vs_noise.png) -->

## Method

1. **Task and data:** [Describe the task, e.g., modular arithmetic such as (a + b) mod p, the standard setting for studying grokking.]
2. **Label noise:** For each experiment, a fixed fraction of training labels (0%, [list levels, e.g., 2.5%, 5%, 7.5%, 10%, 12.5%, 15%]) is randomly replaced with incorrect labels. Test labels are kept clean, so test accuracy measures true generalization.
3. **Model and training:** [Describe the model, e.g., a small transformer or MLP] trained with [optimizer, e.g., AdamW with weight decay] in PyTorch, on GPU (CUDA).
4. **Measuring time-to-grok:** Time-to-grok is defined as [your definition, e.g., the training step at which test accuracy first exceeds 95%].
5. **Analysis:** Time-to-grok is compared across noise levels, and a curve is fit to the results to characterize the relationship.

## Repository Structure

```
[file_name].py      # [what it does, e.g., model definition and training loop]
[file_name].py      # [what it does, e.g., runs experiments across noise levels]
[file_name].py      # [what it does, e.g., analysis and plotting]
requirements.txt    # Python dependencies
```

## Setup

Requires Python 3 and PyTorch. A CUDA-capable GPU is recommended for full experiments.

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Example: train with 5% label noise
python [file_name].py [arguments, e.g., --noise 0.05]
```

[Explain any other options, e.g., number of training steps, random seed, or how to run the full sweep across noise levels.]

## Paper

[Link to the published paper in the Journal of High School Sciences]

## Acknowledgments

Thank you to Prof. Shaddin Dughmi (USC) for mentorship and guidance throughout this research.
