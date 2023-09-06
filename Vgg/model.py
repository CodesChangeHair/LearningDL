from turtle import forward
import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super().__init__()
        self.features = features
        self.classfier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        # input: batch x 3 x 224 x 224
        x = self.features(x)
        # batch x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        x = self.classfier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            # 等价于layers.append(nn.MaxPool2d(2, 2))
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v 
    return nn.Sequential(*layers)

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(model_name='vgg16', **kwargs):
    assert model_name in cfgs, "Warnning: model {} is not in configs dict".format(model_name)
    cfg = cfgs[model_name]
    net = VGG(features=make_features(cfg), **kwargs)
    return net

if __name__ == '__main__':
    # test model 主要是检测数据的形状
    net = vgg()
    input = torch.rand((1, 3, 224, 224))
    output = net(input)
    print(output.shape)
    