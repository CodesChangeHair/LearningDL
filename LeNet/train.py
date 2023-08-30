import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import LeNet
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Train on {}".format(device))

# 准备数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    root = "./dataset", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./dataset", train=False, download=True, transform=transform
)

train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print("训练集大小: {}, 测试集大小: {}".format(train_data_size, test_data_size))

# 利用DataLoader加载数据集
train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

# CIFAR10数据集类别
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # tensor的通道为[batch, channel, height, width] 这里一张图片(c, h, w)
    # 而plt.imshow()需要图片的通道为(h, w, c)
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()

def show_train_images(train_dataset, size=8):
    train_data_loader = DataLoader(train_dataset, size)
    for data in train_data_loader:
        imgs, targets = data
        # print labels
        print(' '.join("%5s" % classes[targets[j]] for j in range(size)))
        imshow(torchvision.utils.make_grid(imgs))
        break

net = LeNet(10)
net = net.to(device)
# CrossEntropyLoss: This criterion computes the cross entropy loss between input logits  and target.
# 损失函数已经包含了Softmax, 所以网络中最后没有添加Softmax层
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

Epoch = 10
writer = SummaryWriter("LeNet")
# 记录训练次数和测试次数
total_train_step, total_test_step = 0, 0
for epoch in range(Epoch):
    print("-" * 10 + "第 {} 轮训练".format(epoch) + "-" * 10)
    
    # train
    net.train()
    for data in train_data_loader:
        # forward and compute loss
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, labels)
        
        # backward 
        optimizer.zero_grad()  # 清零梯度 梯度累加可以累积多个batch结果训练而不需要过大的内存
        loss.backward()
        optimizer.step()
        
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("Epoch: {}, Train step: {}, Loss: {}".format(epoch, total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # test
    net.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    # with: 上下文管理器
    # 不计算梯度: 减少计算代价, 不需要内存资源
    with torch.no_grad():
        for data in test_data_loader:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, labels)
            accuracy = (outputs.argmax(1) == labels).sum().item()
            total_test_loss += loss.item()
            total_accuracy += accuracy
    
    print("Total Loss on test dataset: {}".format(total_test_loss))
    print("Accuracy rate on test dataset: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, epoch)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, epoch)

print("Training compeleted")

# 保存模型
save_path = "./LeNet.pth"
torch.save(net.state_dict(), save_path)
print("Model saved")