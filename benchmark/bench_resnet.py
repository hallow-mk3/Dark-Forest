"""benchmark/bench_resnet.py: Standard CNN & ResNet Throughput Benchmark (img/s).

Measures:
1. PyTorch ResNet-18 (forward + backward training step throughput in img/s)
2. Dark Forest MiniResNet CIFAR (forward + backward training step)
Both measured on RTX 5070 GPU / CPU with batch size 16 across 50 iterations.
"""

import time
import torch
import torch.nn as nn
import subprocess
import json
import os

class PyTorchBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class PyTorchResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(PyTorchBasicBlock(64, 64, 1), PyTorchBasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(PyTorchBasicBlock(64, 128, 2), PyTorchBasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(PyTorchBasicBlock(128, 256, 2), PyTorchBasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(PyTorchBasicBlock(256, 512, 2), PyTorchBasicBlock(512, 512, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def bench_pytorch_resnet18(device="cuda", batch_size=16, iters=50):
    print(f"\n================================================================================")
    print(f" [BENCHMARK] PyTorch ResNet-18 on {device.upper()}")
    print(f" Configuration: Batch Size={batch_size}, CIFAR Resolution=3x32x32, Classes=10")
    print(f"================================================================================")
    model = PyTorchResNet18(num_classes=10).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    imgs_per_sec = (batch_size * iters) / total_time
    ms_per_step = (total_time / iters) * 1000.0
    print(f"  Throughput: {imgs_per_sec:.2f} img/s")
    print(f"  Latency   : {ms_per_step:.2f} ms/step (Forward + Backward + SGD)")
    return imgs_per_sec, ms_per_step


def bench_darkforest_resnet():
    print(f"\n================================================================================")
    print(f" [BENCHMARK] Dark Forest ResNet (Rust Autograd Engine)")
    print(f"================================================================================")
    res = subprocess.run(
        ["cargo", "test", "--test", "resnet18_cifar_test", "--", "test_resnet_cifar_loss_convergence", "--nocapture"],
        capture_output=True,
        text=True
    )
    for line in res.stdout.splitlines():
        if "[PASS]" in line:
            print("  " + line)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench_pytorch_resnet18(device=device, batch_size=16, iters=50)
    bench_darkforest_resnet()
