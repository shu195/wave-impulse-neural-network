"""
4D Quaternion (Hamilton product) layer demo.
Shows the parameter count: 4 per connection vs 16 for a full 4x4 matrix. This file contains no pruning
code and makes no accuracy claim; see README for what was (weakly) measured.
"""

import torch
import torch.nn as nn
import numpy as np

class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 4 components for each connection: r, x, y, z (4 params vs 16 in standard linear)
        self.r = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.x = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.y = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.z = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

    def forward(self, q):
        # Hamilton product
        r, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        out_r = torch.matmul(r, self.r.T) - torch.matmul(x, self.x.T) - torch.matmul(y, self.y.T) - torch.matmul(z, self.z.T)
        out_x = torch.matmul(r, self.x.T) + torch.matmul(x, self.r.T) + torch.matmul(y, self.z.T) - torch.matmul(z, self.y.T)
        out_y = torch.matmul(r, self.y.T) - torch.matmul(x, self.z.T) + torch.matmul(y, self.r.T) + torch.matmul(z, self.x.T)
        out_z = torch.matmul(r, self.z.T) + torch.matmul(x, self.y.T) - torch.matmul(y, self.x.T) + torch.matmul(z, self.r.T)
        return torch.stack([out_r, out_x, out_y, out_z], dim=-1)

def demo_quaternion_efficiency():
    print("=== Quaternion 4D Parameter Efficiency ===")
    print("Standard 4D-to-4D Linear layer weights: 4 x 4 = 16 parameters")
    print("Quaternion 4D-to-4D neuron weight:      1 x 4 =  4 parameters (75% savings)")

if __name__ == '__main__':
    demo_quaternion_efficiency()
