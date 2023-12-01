import os
import sys
import json

import torch
import torch.nn as nn   
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} to train".format(device))

data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_root = os.path.abspath(os.path.join(os.getcwd(), "."))
image_path = os.path.join(data_root, "dataset", "flower_data")
assert os.path.exists(image_path), "{} does not exist.".format(image_path)

train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                     transform=data_transform['train'])
val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                   transform=data_transform['val'])

train_dataset_size = len(train_dataset)
val_dataset_size = len(val_dataset)
print("Using {} images for traing, {} images for validation".format(train_dataset_size, val_dataset_size))

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
class_to_idx = train_dataset.class_to_idx
idx_to_class = dict((val, key) for key, val in class_to_idx.items())
# write "idx: flower_type" dict to json
json_str = json.dumps(idx_to_class, indent=4)
with open('classes.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 16
num_workers = min([os.cpu_count(), batch_size if batch_size > 0 else 0, 8])
print("Using {} dataloader workers every process".format(num_workers))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

net = resnet34()
# load pretrain weights
# download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
model_weight_path = "./resnet34-pre.pth"
assert os.path.exists(model_weight_path), "file {} does not exist".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path, map_location='cpu')) 

# change fully connected layer structure
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 5)
net.to(device)

loss_function = nn.CrossEntropyLoss()

params = [param for param in net.parameters() if param.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)

epochs = 5
train_steps = len(train_dataloader)
total_train_steps = 0
writer = SummaryWriter("ResNet34")
for epoch in range(epochs):
    print("-" * 10 + "第 {} 轮训练".format(epoch + 1) + "-" * 10)
    
    # train
    net.train()
    train_bar = tqdm(train_dataloader, file=sys.stdout)
    for data in train_bar:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(images)
        loss = loss_function(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_steps += 1
        if total_train_steps % 50 == 0:
            writer.add_scalar("train loss", loss.item(), total_train_steps)
        
        train_bar.desc = "train epoch[{} / {}] loss: {:.3f}".format(epoch + 1,
                                                                    epochs, loss)
    # validation
    net.eval()
    acc = 0.0
    with torch.no_grad():
        total_loss = 0.0
        total_accuracy = 0.0
        val_bar = tqdm(val_dataloader, file=sys.stdout)
        for data in val_bar:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            
            accuracy = (outputs.argmax(1) == labels).sum().item()
            total_accuracy += accuracy
            total_loss += loss.item()
            
            val_bar.desc = "valid epoch[{} / {}]".format(epoch + 1, epochs)
            
        val_accuracy = total_accuracy / val_dataset_size
        print("[epoch: {}] val accuracy {:.3f}, val loss: {:.3f}".format(
            epoch + 1, val_accuracy, total_loss
        ))
        writer.add_scalar("val accuracy", val_accuracy, epoch)
        writer.add_scalar("val loss", total_loss, epoch)
    
    print("Training completed")
    
save_path = './resNet34.pth'
torch.save(net.state_dict(), save_path)
print("Model parameter saved")