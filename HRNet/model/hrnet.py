from doctest import FAIL_FAST
from hmac import trans_36
from turtle import forward
from xmlrpc.client import TRANSPORT_ERROR
from sklearn import base, kernel_approximation
from torch import scalar_tensor
import torch.nn as nn

BN_MOMENTUM = 0.1

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, channel):
        """
        构建stage, 实现不同尺度特征的融合
        input_branches: 输入分支数
        channel: 输入第一个分支的特征通道数
        """        
        super().__init__()
        super.input_branches = input_branches
        self.output_branches = output_branches
        
        self.branches = nn.ModuleList()  # Holds submodules in a list.
        for i in range(self.input_branches):  
            # 每个分支先通过4个BasicBlock
            w = channel * (2 ** i)  # 分支对应的通道数
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)
        
        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 输入、输出属于同一分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 输入j分支大于输出分支 
                    # 需要对输入j进行通道调整以及上采样处理, 统一通道大小
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(channel * (2 ** j), channel * (2 ** i), kernel_size=1,
                                      stride=1, bias=False),
                            nn.BatchNorm2d(channel * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 输入分支j小于输出分支i
                    # 需要对输入进行通道调整以及下采样
                    # 这里每次2倍下采样都是通过一个3x3卷积实现
                    ops = []
                    for k in range(i - j - 1):
                        w = channel * (2 ** j)
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(w, w, kernel_size=3, stride=2,
                                          padding=1, bias=False),
                                nn.BatchNorm2d(w, momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层需要下采样 + 调整通道数
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(channel * (2 ** j), channel * (2 ** i), kernel_size=3,
                                      stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(channel * (2 ** i), momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True)
                        )
                    )
                    
                    self.fuse_layers[-1].append(nn.Sequential(*ops))
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        
        # 融合不同尺度信息
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )
            
        return x_fused
    
class HighResolutionNet(nn.Module):
    def __init__(self, base_channel: int = 32, num_joints: int = 17):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])
        
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, channel=base_channel)
        )
        
        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])
        
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, channel=base_channel),
            StageModule(input_branches=3, output_branches=3, channel=base_channel),
            StageModule(input_branches=3, output_branches=3, channel=base_channel),
            StageModule(input_branches=3, output_branches=3, channel=base_channel)
        )
        
        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])
        
        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, channel=base_channel),
            StageModule(input_branches=4, output_branches=4, channel=base_channel),
            StageModule(input_branches=4, output_branches=1, channel=base_channel)
        )
        
        self.final_layer = nn.Conv2d(base_channel, num_joints, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # since now, x is a list
        
        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]
        
        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]
        
        x = self.stage4(x)
        
        x = self.final_layer(x[0])
        
        return x
    
    