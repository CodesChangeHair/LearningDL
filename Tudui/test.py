import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "demo/horse.png"
image = Image.open(image_path)
image = image.convert('RGB')  # png格式是4通道, 除颜色通道外还包含透明度通道 这里调用convert()只保留颜色通道
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((32,32))
])
image = transform(image)

# 搭建神经网络
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        # 声明模型包含的Layer
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),  # 将矩阵转为一维
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

network = torch.load("demo/model/nn_29.pth", map_location=torch.device('cpu'))
image = torch.reshape(image, (1, 3, 32, 32))
network.eval()
with torch.no_grad():
    output = network(image)
index = output.argmax(1)
print(index)
    

