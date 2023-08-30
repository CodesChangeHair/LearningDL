from numpy import argmax
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR10数据集类别
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet(num_classes=10)
net.load_state_dict(torch.load('LeNet.pth'))

img = Image.open('horse.png')
img = transform(img)
img = torch.unsqueeze(img, dim=0)  # (C, H, W) --> (B, C, H, W)

with torch.no_grad():
    net.eval()
    output = net(img)
index = output.argmax(1)
print(output)
print('the predict class is {}'.format(classes[index]))

img = Image.open('car.png')
img = transform(img)
img = torch.unsqueeze(img, dim=0)  # (C, H, W) --> (B, C, H, W)

with torch.no_grad():
    net.eval()
    output = net(img)
index = output.argmax(1)
print(output)
print('the predict class is {}'.format(classes[index]))