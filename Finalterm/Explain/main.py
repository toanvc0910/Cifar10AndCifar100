# Mount Google Drive to save checkpoints (won't lose progress if disconnected)

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

# === CONFIG ===
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 200
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DROPOUT = 0.3
DEPTH = 28
WIDEN_FACTOR = 4
CUTOUT_SIZE = 16
LABEL_SMOOTHING = 0.1

# Checkpoints saved to Google Drive (persist across sessions)
DRIVE_DIR = '/content/drive/MyDrive/cifar10_wrn'
os.makedirs(DRIVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)
        return x * w


class WideResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.se(out)
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=4, dropout=0.3, num_classes=10, remove_group3=False):
        super().__init__()
        assert (depth - 4) % 6 == 0 # Ép depth phải hợp lệ theo công thức WRN: depth = 6n + 4.
        n = (depth - 4) // 6 # Tính số residual block mỗi group. Với depth=28 thì n=4.

        ch = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        #Tạo danh sách số kênh cho từng stage: ch[0]=16 (stem), ch[1]=64, ch[2]=128, ch[3]=256 khi widen_factor=4.
        
        self.remove_group3 = remove_group3 # Lưu lại cờ ablation để biết có bỏ group3 hay không.
    
        self.conv1 = nn.Conv2d(3, ch[0], 3, 1, 1, bias=False) 
        #in_channels=3 (RGB), out_channels=16, kernel=3x3, stride=1, padding=1.
        #bias=False vì thường đi kèm BatchNorm (bias không cần thiết).
    
        self.group1 = self._make_group(ch[0], ch[1], n, stride=1, dropout=dropout)
        # Tạo Group1 gồm n residual blocks.Kênh 16 -> 64, stride block đầu là 1 (không giảm kích thước ảnh).
        self.group2 = self._make_group(ch[1], ch[2], n, stride=2, dropout=dropout)
        #Tạo Group2 gồm n block. Kênh 64 -> 128, stride block đầu là 2 (giảm spatial còn một nửa).
        self.group3 = self._make_group(ch[2], ch[3], n, stride=2, dropout=dropout)
        
        self.bn_final = nn.BatchNorm2d(ch[3])
        #BatchNorm cuối theo đúng số kênh sau stage cuối (128 hoặc 256).
        self.fc = nn.Linear(ch[3], num_classes)
        #Fully Connected phân loại:đầu vào final_ch, đầu ra num_classes (ví dụ 10 lớp CIFAR-10).
        self._init_weights()

    def _make_group(self, in_ch, out_ch, n_blocks, stride, dropout):
        #Block đầu group dùng stride truyền vào (1 hoặc 2) và có thể đổi số kênh.
        layers = [WideResBlock(in_ch, out_ch, stride, dropout)]
        for _ in range(1, n_blocks):
            layers.append(WideResBlock(out_ch, out_ch, 1, dropout))
            #Các block sau giữ nguyên số kênh, stride=1.
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x) # Qua conv đầu tiên: tăng kênh từ 3 lên 16.
        x = self.group1(x)# Qua group1: trích xuất đặc trưng mức đầu.
        x = self.group2(x)# Qua group2: tăng kênh, giảm spatial.
        x = self.group3(x)# Qua group3: tăng kênh, giảm spatial.
        x = F.relu(self.bn_final(x), inplace=True) # BatchNorm cuối rồi ReLU.
        x = F.adaptive_avg_pool2d(x, 1) # Global average pooling về kích thước 1x1.
        x = torch.flatten(x, 1) # Flatten từ [B, C, 1, 1] thành [B, C].
        return self.fc(x) # Qua FC để ra logits [B, num_classes].

# Debugs========================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Inspect WideResNet architecture quickly (no full training).')
    parser.add_argument('--depth', type=int, default=DEPTH)
    parser.add_argument('--widen-factor', type=int, default=WIDEN_FACTOR)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES)
    parser.add_argument('--batch-size-inspect', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--remove-group3', action='store_true', help='Ablation: remove Block3/Group3.')
    return parser.parse_args()


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def print_param_breakdown(model):
    print('\n=== Parameter Breakdown ===')
    named_parts = [
        ('conv1', model.conv1),
        ('group1', model.group1),
        ('group2', model.group2),
        ('group3', model.group3),
        ('bn_final', model.bn_final),
        ('fc', model.fc),
    ]
    total = count_params(model)
    for name, part in named_parts:
        p = count_params(part)
        ratio = (100.0 * p / total) if total > 0 else 0.0
        print(f'{name:10s}: {p:>10,} ({ratio:>6.2f}%)')
    print(f'total     : {total:>10,}')


def collect_shape_trace(model, x):
    traces = []

    def hook_fn(name):
        def _hook(_, __, out):
            traces.append((name, tuple(out.shape)))
        return _hook

    hooks = [
        model.conv1.register_forward_hook(hook_fn('conv1')),
        model.group1.register_forward_hook(hook_fn('group1')),
        model.group2.register_forward_hook(hook_fn('group2')),
        model.group3.register_forward_hook(hook_fn('group3')),
        model.bn_final.register_forward_hook(hook_fn('bn_final')),
        model.fc.register_forward_hook(hook_fn('fc')),
    ]

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return traces


def benchmark_forward(model, x, warmup=10, iters=30):
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / max(iters, 1)
    return avg_ms


def one_step_backward_check(model, x, num_classes):
    model.train()
    targets = torch.randint(0, num_classes, (x.size(0),), device=x.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, targets)
    loss.backward()

    grad_sq_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sq_sum += p.grad.detach().pow(2).sum().item()
    grad_norm = grad_sq_sum ** 0.5
    optimizer.step()

    return loss.item(), grad_norm


def main():
    args = parse_args()

    inspect_model = WideResNet(
        depth=args.depth,
        widen_factor=args.widen_factor,
        dropout=args.dropout,
        num_classes=args.num_classes,
        remove_group3=args.remove_group3,
    ).to(device)

    print('\n=== Model Setup ===')
    print(f'depth={args.depth}, widen_factor={args.widen_factor}, dropout={args.dropout}, classes={args.num_classes}')
    print(f'remove_group3={args.remove_group3}')
    print(f'total_params={count_params(inspect_model):,}')

    x = torch.randn(args.batch_size_inspect, 3, args.image_size, args.image_size, device=device)

    traces = collect_shape_trace(inspect_model, x)
    print('\n=== Shape Trace ===')
    print(f'input     : {tuple(x.shape)}')
    for name, shape in traces:
        print(f'{name:10s}: {shape}')

    print_param_breakdown(inspect_model)

    avg_ms = benchmark_forward(inspect_model, x, warmup=args.warmup, iters=args.iters)
    throughput = 1000.0 * args.batch_size_inspect / avg_ms if avg_ms > 0 else 0.0
    print('\n=== Speed ===')
    print(f'avg_forward={avg_ms:.3f} ms/batch | throughput={throughput:.2f} samples/s')

    loss, grad_norm = one_step_backward_check(inspect_model, x, args.num_classes)
    print('\n=== One-Step Backward Check ===')
    print(f'loss={loss:.5f} | grad_norm={grad_norm:.5f}')

    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f'peak_gpu_mem={peak_mem_mb:.2f} MB')


if __name__ == '__main__':
    main()