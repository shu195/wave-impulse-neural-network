"""
Checks that the 'wave impulse' update rule used in python_wave_mnist.py equals the
ordinary backpropagation gradient (computed with PyTorch autograd) on the same batch.
Expected: differences ~1e-17 and cosine similarity 1.0 for every layer.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B = 128
x = torch.rand(B, 784, dtype=torch.float64)
lab = torch.randint(0, 10, (B,))
by = torch.zeros(B, 10, dtype=torch.float64)
by.scatter_(1, lab[:, None], 1.0)
W1 = (torch.randn(512, 784, dtype=torch.float64) * (2 / 784) ** .5).requires_grad_()
b1 = torch.zeros(512, dtype=torch.float64, requires_grad=True)
W2 = (torch.randn(256, 512, dtype=torch.float64) * (2 / 512) ** .5).requires_grad_()
b2 = torch.zeros(256, dtype=torch.float64, requires_grad=True)
W3 = (torch.randn(10, 256, dtype=torch.float64) * (2 / 256) ** .5).requires_grad_()
b3 = torch.zeros(10, dtype=torch.float64, requires_grad=True)

# the "wave impulse" rule exactly as in python_wave_mnist.py (no autograd used)
with torch.no_grad():
    h1 = torch.sigmoid(x @ W1.T + b1)
    h2 = torch.sigmoid(h1 @ W2.T + b2)
    y = torch.softmax(h2 @ W3.T + b3, dim=-1)
    d3 = by - y
    d2 = (d3 @ W3) * (h2 * (1 - h2))
    d1 = (d2 @ W2) * (h1 * (1 - h1))
    wave = {'W3': d3.T @ h2 / B, 'W2': d2.T @ h1 / B, 'W1': d1.T @ x / B}

# ordinary autograd gradient of mean cross-entropy
logits = torch.sigmoid(torch.sigmoid(x @ W1.T + b1) @ W2.T + b2) @ W3.T + b3
F.cross_entropy(logits, lab).backward()
auto = {'W3': -W3.grad, 'W2': -W2.grad, 'W1': -W1.grad}  # update direction = -gradient

for k in ['W1', 'W2', 'W3']:
    a, w = auto[k].flatten(), wave[k].flatten()
    cos = float(a @ w / (a.norm() * w.norm()))
    print('%s: max|wave - (-autograd)| = %.3e | cosine = %.12f' % (k, float((a - w).abs().max()), cos))
