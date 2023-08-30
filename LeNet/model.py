import torch
import torch.nn as nn
import torch.nn.functional as F 

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        """
        num_class: 分类的数量
        """
        super().__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.MaxPool2d(2),
        )
        
        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, self.num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifer(x)
        return x
     
if __name__ == "__main__":
    # 测试模型
    net = LeNet(10)
    print(net)   
    input = torch.rand((1, 3, 32, 32))
    output = net(input)
    print(output.shape)