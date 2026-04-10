"""
WideResNet-28-4 + SE Attention

Import: from models.colabmodel import *
Dùng:   model = TestNet()          # CIFAR-10, 32×32
        model = TestNet100()       # CIFAR-100, 32×32
        model = AnimalsNet()       # Animals-5, 224×224 hoặc 32×32
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SE BLOCK
# ============================================================
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


# ============================================================
# WIDE RESIDUAL BLOCK — Pre-activation
# ============================================================
class WideResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.3):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop  = nn.Dropout(dropout)
        self.se    = SEBlock(out_ch, reduction=16)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                      kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = self.drop(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        out = self.se(out)
        return out + self.shortcut(x)


# ============================================================
# WRN-28-4 — CIFAR-10 / CIFAR-100
# ============================================================
# depth=28, widen_factor=4, n=4 blocks/group
# Channels: [16, 64, 128, 256]
class WRN_28_4(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()

        # === Lớp đầu vào: 32×32×3 → 32×32×16 ===
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=1, padding=1, bias=False)

        # === GROUP 1: 32×32, 16→64 ===
        self.block1_1 = WideResBlock(in_ch=16,  out_ch=64, stride=1, dropout=dropout)
        self.block1_2 = WideResBlock(in_ch=64,  out_ch=64, stride=1, dropout=dropout)
        self.block1_3 = WideResBlock(in_ch=64,  out_ch=64, stride=1, dropout=dropout)
        self.block1_4 = WideResBlock(in_ch=64,  out_ch=64, stride=1, dropout=dropout)

        # === GROUP 2: 32→16, 64→128 ===
        self.block2_1 = WideResBlock(in_ch=64,  out_ch=128, stride=2, dropout=dropout)
        self.block2_2 = WideResBlock(in_ch=128, out_ch=128, stride=1, dropout=dropout)
        self.block2_3 = WideResBlock(in_ch=128, out_ch=128, stride=1, dropout=dropout)
        self.block2_4 = WideResBlock(in_ch=128, out_ch=128, stride=1, dropout=dropout)

        # === GROUP 3: 16→8, 128→256 ===
        self.block3_1 = WideResBlock(in_ch=128, out_ch=256, stride=2, dropout=dropout)
        self.block3_2 = WideResBlock(in_ch=256, out_ch=256, stride=1, dropout=dropout)
        self.block3_3 = WideResBlock(in_ch=256, out_ch=256, stride=1, dropout=dropout)
        self.block3_4 = WideResBlock(in_ch=256, out_ch=256, stride=1, dropout=dropout)

        # === Đầu ra ===
        self.bn_final = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256, num_classes)

        self._init_weights()

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
        x = self.conv1(x)          # → (B, 16, 32, 32)

        x = self.block1_1(x)       # → (B, 64, 32, 32)
        x = self.block1_2(x)
        x = self.block1_3(x)
        x = self.block1_4(x)

        x = self.block2_1(x)       # → (B, 128, 16, 16)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.block2_4(x)

        x = self.block3_1(x)       # → (B, 256, 8, 8)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)

        x = F.relu(self.bn_final(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============================================================
# WRN-16-2 — Animals-5
# ============================================================
# depth=16, widen_factor=2, n=2 blocks/group
# Channels: [16, 32, 64, 128]
class WRN_16_2(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=1, padding=1, bias=False)

        # === GROUP 1: giữ size, 16→32 ===
        self.block1_1 = WideResBlock(in_ch=16, out_ch=32, stride=1, dropout=dropout)
        self.block1_2 = WideResBlock(in_ch=32, out_ch=32, stride=1, dropout=dropout)

        # === GROUP 2: ÷2, 32→64 ===
        self.block2_1 = WideResBlock(in_ch=32, out_ch=64, stride=2, dropout=dropout)
        self.block2_2 = WideResBlock(in_ch=64, out_ch=64, stride=1, dropout=dropout)

        # === GROUP 3: ÷2, 64→128 ===
        self.block3_1 = WideResBlock(in_ch=64,  out_ch=128, stride=2, dropout=dropout)
        self.block3_2 = WideResBlock(in_ch=128, out_ch=128, stride=1, dropout=dropout)

        self.bn_final = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, num_classes)

        self._init_weights()

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
        x = self.conv1(x)

        x = self.block1_1(x)
        x = self.block1_2(x)

        x = self.block2_1(x)
        x = self.block2_2(x)

        x = self.block3_1(x)
        x = self.block3_2(x)

        x = F.relu(self.bn_final(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============================================================
# ALIAS — checkmodel.py và ptflops_count.py gọi TestNet()
# ============================================================
# ĐỔI ALIAS NÀY tùy đang test model nào:

def TestNet():
    """Mặc định: WRN-28-4, CIFAR-10, input 32×32"""
    return WRN_28_4(num_classes=10, dropout=0.3)

def TestNet100():
    """WRN-28-4, CIFAR-100, input 32×32"""
    return WRN_28_4(num_classes=100, dropout=0.3)

def AnimalsNet():
    """WRN-16-2, Animals-5, input 224×224 hoặc 32×32"""
    return WRN_16_2(num_classes=5, dropout=0.3)