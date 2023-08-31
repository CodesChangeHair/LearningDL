import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import json 
import os
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from model import AlexNet

# 训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on {}".format(device))

# 准备数据
data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

data_root = os.path.abspath(os.path.join(os.getcwd(), "."))
image_path = data_root + "/dataset/flower_data/" 
train_dataset = datasets.ImageFolder(root=image_path + '/train',
                                     transform=data_transform['train'])
test_dataset = datasets.ImageFolder(root=image_path + '/test',
                                    transform=data_transform['test'])
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print("训练集大小: {}, 测试集大小: {}".format(train_dataset_size, test_dataset_size))

# 保存数据集类别信息
flower_list = train_dataset.class_to_idx
class_dict = dict((val, key) for key, val in flower_list.items())  # 将字典flower_list的键值对互换
json_str = json.dumps(class_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# 通过DataLoader加载数据
batch_size = 32
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))  # (c, h, w) --> (h, w, c) 以符合plt.imshow()
    plt.imshow(npimg)
    plt.show()

def show_train_images(train_dataset, size=8):
    train_dataloader = DataLoader(train_dataset, batch_size=size, shuffle=True)
    for data in train_dataloader:
        imgs, labels = data
        print(' '.join("%5s" % class_dict[j.item()] for j in labels))
        imshow(torchvision.utils.make_grid(imgs))
        break
        
# show_train_images(train_dataset)
        
net = AlexNet(num_classes=5, init_weight=True)
net = net.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

optimizer = optim.Adam(net.parameters(), lr=2e-4)

Epoch = 20
writer = SummaryWriter("AlexNet")
total_train_step = 0
for epoch in range(Epoch):
    print("-" * 10 + "第 {} 轮训练".format(epoch) + "-" * 10)
    
    # train
    net.train()
    for step, data in enumerate(train_dataloader, start=0):
        # forward 
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, labels)
        
        # backward 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 打印训练过程
        rate = (step + 1) / len(train_dataloader)
        a = "*" * int(rate * 50)
        b = "*" * int((1 - rate) * 50)
        print("\rprocess: {:.3f}%[{}->{}] train loss: {:.3f}".format(int(rate * 100), a, b, loss.item()), end="")
        
        total_train_step += 1
        if total_train_step % 100 == 0:
            # print("Epoch: {}, Train step: {}, Loss: {}".format(epoch, total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step) 
    
    # test
    net.eval()  # no dropout
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, labels)
            
            accuracy = (outputs.argmax(1) == labels).sum().item()
            total_accuracy += accuracy
            total_test_loss += loss.item()
        
    print()    
    print("Total loss on test dataset: {}".format(total_test_loss))
    print("Accuracy rate on test dataset: {}".format(total_accuracy / test_dataset_size))
    writer.add_scalar("test_loss", total_test_loss, epoch)
    writer.add_scalar("test_accuracy", total_accuracy / test_dataset_size, epoch)

print("Traing compeleted")

# 保存模型
save_path = "./AlexNet.pth"
torch.save(net.state_dict(), save_path)
print("Model saved")      
        