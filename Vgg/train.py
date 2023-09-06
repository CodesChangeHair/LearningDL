import os 
import sys
import json

import torch
import torch.nn as nn 
from torchvision import transforms, datasets 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from model import vgg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Train on {}".format(device))

data_transtorm = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        # validation阶段可以用randomResize数据增强吗
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])   
}

data_root = os.path.abspath(os.path.join(os.getcwd(), "."))
image_path = os.path.join(data_root, "dataset", "flower_data")

assert os.path.exists(image_path), "{} does not exist.".format(image_path)

train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transtorm['train'])
val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                   transform=data_transtorm['val'])
train_dataset_size = len(train_dataset)
val_dataset_size = len(val_dataset)
print("训练集大小: {}, 验证集大小: {}".format(train_dataset_size, val_dataset_size))

# 存储数据集类别id: 类别名
flower_list = train_dataset.class_to_idx
idx_to_class = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(idx_to_class, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)
    

# 使用DataLoader加载数据
batch_size = 32
num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])    
print("Using {} dataloader workers every process".format(num_workers))

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

model_name = 'vgg16'
net = vgg(model_name=model_name, num_classes=5, init_weights=True)
net = net.to(device)
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)

epochs = 30
total_train_steps = 0
writer = SummaryWriter("VGG")
for epoch in range(epochs):
    print("-" * 10 + "第 {} 轮训练".format(epoch + 1) + "-" * 10)
    
    net.train()
    train_bar = tqdm(train_dataloader, file=sys.stdout)
    for data in train_bar:
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        outputs = net(imgs)
        loss = loss_function(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_steps += 1
        if total_train_steps % 100 == 0:
            writer.add_scalar("train_loss", loss.item(), total_train_steps)

        train_bar.desc = "train epoch[{} / {}], loss: {:.3f}".format(epoch + 1, epochs, loss.item())
        
    net.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_accuracy = 0.0
        val_bar = tqdm(val_dataloader, file=sys.stdout)
        for data in val_bar:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = net(imgs)
            
            accuracy = (outputs.argmax(1) == labels).sum().item()
            total_accuracy += accuracy
            total_loss += loss.item()
        print("epoch: {}, val loss: {}, val accuracy: {}".format(epoch + 1, total_loss, total_accuracy / val_dataset_size))
        writer.add_scalar("test_accuracy", total_accuracy / val_dataset_size, epoch)
        writer.add_scalar("test_loss", total_loss, epoch)
print("Training completed")        

save_path = "./{}Net.pth".format(model_name)
torch.save(net.state_dict(), save_path)
print("Model saved")