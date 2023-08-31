from http.client import ImproperConnectionState
from turtle import forward
import torch.nn as nn
import torch 

class AlexNet(nn.Module):
    def __init__(self, num_classes=-1, init_weight=False):
        """ 
        The input for AlexNet is a 224x224 RGB image.
        
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier. 
            
        (copy from mmpose version of AlexNet)
        """
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # (48, 55, 55)
            nn.ReLU(inplace=True),  # inplace：can optionally do the operation in-place 在地计算 不需要申请额外内存空间 但会覆盖原变量
            nn.MaxPool2d(kernel_size=3, stride=2),  # size = (size - 3) / 2 + 1  --> (48, 27, 27)
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # (128, 27, 27),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (128, 13, 13)
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # (192, 13, 13),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # (192, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # (128, 13, 13),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # (128, 6, 6)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # p: probability of an element to be zeroed. Default: 0.5
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )
        if init_weight:
            self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x) 
        if self.num_classes > 0:
            # x = x.view(x.size(0), 128 * 6 * 6)
            x = torch.flatten(x, start_dim=1)  # 从第一维开始(跳过bath维)展平
            x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():  # self.modules(): Returns an iterator over all modules in the network.
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    # test model
    input = torch.rand((1, 3, 224, 224))
    net = AlexNet(num_classes=5)
    output = net(input)
    print(output.shape)