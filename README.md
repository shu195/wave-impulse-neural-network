# Wave-Impulse Network experiments (corrected README)

> **Correction notice (2026-10-16).** An earlier version of this README claimed the
> 98% MNIST results were obtained with *zero backpropagation*, *O(1) memory*, and a
> *speedup over backprop*. **Those claims were wrong and have been removed.**
> The update rule that reaches ~98% is a hand-written backpropagation (see below).
> Run `python verify_backprop_equivalence.py` to check this yourself.

## What this repo actually contains

Small experiments on MNIST and synthetic tasks, written in C++ and PyTorch, exploring
"wave / echo" style learning rules, 4D (quaternion) neurons, and connection pruning.
Everything here is single-run, small scale, and untuned. Treat it as exploratory.

## The key fact: the 98% update rule *is* backpropagation

`python_wave_mnist.py` and `wave_scaling.cpp` compute, for a sigmoid network with a
softmax/cross-entropy output:

```
delta_L = y_target - y
delta_l = (delta_{l+1} @ W_{l+1}) * sigma'(h_l)      # echo "ripple" backwards
dW_l    = delta_l^T @ h_{l-1} / batch
```

This is exactly the chain rule written by hand (no autograd library, but the same
math). `verify_backprop_equivalence.py` compares it with PyTorch autograd on the same
batch (float64):

```
W1: max|wave - (-autograd)| = 1.3e-18 | cosine = 1.000000000000
W2: max|wave - (-autograd)| = 6.9e-18 | cosine = 1.000000000000
W3: max|wave - (-autograd)| = 2.8e-17 | cosine = 1.000000000000
```

Consequences:
- The ~98% accuracy is what ordinary backprop gives for this architecture. It is **not**
  evidence that backprop is unnecessary.
- It stores layer activations (they are needed to compute the deltas), so it is **not**
  O(1) memory. It only avoids building an autograd tape.
- Earlier speed comparisons were not controlled (different update rules and
  optimizers) and are **not supported**.

## Results (what was measured)

All numbers are from single runs on the full 60k-train / 10k-test MNIST split unless noted.

| Experiment | Network | Update rule | Test accuracy |
| :-- | :-- | :-- | :-- |
| `python_wave_mnist.py` | 784-512-256-10, sigmoid | hand-written backprop + Adam-style moments | 98.18% at epoch 15 (98.26% peak over epochs, picked on the test set, so slightly optimistic) |
| `wave_scaling.cpp` Stage 1 | 784-256-10 | same rule | 97.81% (6 epochs) |
| `wave_scaling.cpp` Stage 2 | 784-512-256-10 | same rule | 98.05% (6 epochs) |
| `wave_scaling.cpp` Stage 3 | 784-512-256-128-64-10 | same rule | 97.77% (6 epochs) |
| Early relaxation prototype | 784-512-10 | finite-step "nudged relaxation" contrast (Equilibrium-Propagation style), plain SGD | 87.6% (plateaued) |

Notes:
- The C++ code is unoptimized single-threaded loops and is slower than PyTorch.
- The 87.6% prototype is the only rule here that is *not* identical to backprop. Why it
  plateaued was **not diagnosed**: the later version changed several things at once
  (exact cross-entropy delta, momentum, and more), so the cause is unknown.

## Exploratory results that are weak evidence only

These used small synthetic tasks, a single seed, and no confidence intervals:

- **Quaternion (4D) neurons:** 75% fewer parameters per connection by construction
  (4 vs 16). On small synthetic recovery tasks at equal parameter count, test loss was
  lower than an MLP (0.1883 vs 0.2793; 0.2898 vs 0.3679 at 16,384 params). Suggestive,
  not established. An MNIST comparison (88.0% vs 69.4%) used a baseline whose accuracy
  *collapsed* during training, which most likely reflects poor baseline tuning, so do
  not read it as evidence.
- **Pruning 70-75% of connections:** accuracy stayed within about 1.5 points of the dense
  model on a 1,000-image test set (standard error around 1 point). This is consistent with
  "little or no loss", **not** a demonstrated accuracy gain.
- "Weight intertwining" (2-hop path reinforcement) showed no clear benefit in the
  equal-parameter comparison (0.0601 vs 0.0574 loss for the static version).

## Not demonstrated

- Training without backpropagation (beyond the weaker 87.6% prototype).
- Memory or speed advantages.
- Any hardware (photonic / neuromorphic) benefit; none was tested.
- Scaling beyond small MLPs on MNIST.

## Run

```bash
g++ -O3 -march=native wave_scaling.cpp -o wave_scaling     # needs raw MNIST in mnist_raw/
python python_wave_mnist.py                                  # downloads MNIST
python verify_backprop_equivalence.py                        # wave rule vs autograd
```

## Files

```
README.md                          this file
verify_backprop_equivalence.py     checks the wave rule equals autograd gradients
python_wave_mnist.py               3-layer network, hand-written backprop
wave_scaling.cpp                   multi-depth C++ version
quaternion_4d_grid.py              Hamilton-product layer (demo only)
RESEARCH_PAPER.md                  speculative notes, NOT verified (see its banner)
```

License: not yet specified. Author: github.com/shu195
