from turtle import forward
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """ 
    residual block for ResNet18 and ResNet34
    
    """
    expansion = 1  # 经过block之后 特征维度相比于第一层的输出扩大了expansion倍(放在init外使得外部函数可以访问)
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        """
        Args:
            stride 默认为1 不改变feature map的高宽
            downsample: 实现 "THe dotted shortcuts increase dimensions."

            BasicBlock结构: ((conv - BN - ReLU - conv - BN) + identity) - ReLU
        """
        super().__init__()
        
        # stride = stride 第一个conv可能需要减小feature map的高宽
        # bias = False, 后面接BN, 均值会变为0 bias = True没有意义 
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # ReLU层不含参数 可以复用
        self.relu = nn.ReLU(inplace=True)
        # stride = 1, 后续同一Block中conv 不会改变feature map的形状(vgg 设计思想)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x 
        if self.downsample:
            # dotted shortcut, 通过1x1 conv 改变feature map维度
            # 这里的下采样会将size减半 channel加倍
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    
    相比于BasicBlock, Bottleneck使用1x1 conv将feature map深度降低, 减少
    与中间3x3 conv的计算复杂度, 接着使用1x1 conv将feature map深度升高至初始水平
    类似中间小两头大的沙漏结构
    
    BottleNeck结构 (1x1 conv(降低深度) - BN - ReLU - 3x3 conv(提取特征) - 
    1x1 conv(增加深度) - BN + identity) - ReLU
    """
    expansion = 4  # 最后feature map的深度相比于第一层的输出深度增大了4倍
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        # 1x1 conv squeeze channels
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        # 这里3x3 conv 可能还起降低feature map高宽的作用
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        # 1x1 conv unsqueeze channels
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        if self.downsample:
            # dotted shortcuts
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out
    
class ResNet(nn.Module):
    
    def __init__(self, block, block_nums, num_classes=1000, include_top=True):
        """
        Args:
            block: 基本模块
            block_num: list, 每个模块重复数量
            num_classes: 分类类别数目
            include_top： 包含全连接层分类(没有就是特征提取器?)
        """
        super().__init__()
        self.include_top = include_top
        self.in_channel = 64 # conv2_x第一层接受的通道数
        # conv1 in paper
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layer1 ~ layer4对应conv2_x ~ conv5_x
        self.layer1 = self._make_layer(block, 64, block_nums[0])
        self.layer2 = self._make_layer(block, 128, block_nums[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_nums[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_nums[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # self.in_channel: block中第一层接受的特征channel
        # channel: block中第一层的输出特征的channel
        # stride != 1, 表示经过block后特征的size会发生改变
        # self.in_channel != channel * block.expansion表示经过block后, 特征维度的深度会发生改变
        # 也就是当输入x的size或channel改变时, 需要使用dotted shortcut改变输入的维度
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        # 第一个block可能包含dotted shortcut
        layers.append(
            block(self.in_channel, channel, downsample=downsample, stride=stride)
        )
        
        # 后续层接受的特征通道
        self.in_channel = channel * block.expansion
        
        # block中的后续层
        for _ in range(1, block_num):
            layers.append(
                block(self.in_channel, channel)
            )
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc(x)
        
        return x
            
    
def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

if __name__ == "__main__":
    net = resnet50()
    print(net)
    input = torch.rand((1, 3, 224, 224))
    output = net(input)
    print(output.shape)