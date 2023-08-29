import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, tensor
import torch

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on device: {}".format(device))

# 准备数据集
train_data = torchvision.datasets.CIFAR10("./dataset", train=True, 
                                          transform=torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, 
                                          transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练集大小: {}, 测试集大小: {}".format(train_data_size, test_data_size))


# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

network = NN()
network = network.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

# 记录训练次数和测试次数
total_train_step, total_test_step = 0, 0

# 训练轮次
epoch = 30

# 添加tensorboard
writer = SummaryWriter("train")

for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i + 1))
    
    # 训练步骤
    network.train()
    # 遍历训练集
    for data in train_dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = network(inputs)
        loss = loss_fn(outputs, labels)
        
        # 使用优化器反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
        
    # 测试步骤
    network.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = network(inputs)
            loss = loss_fn(outputs, labels)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy += accuracy
    
    print("模型在测试集上的Total Loss: {}".format(total_test_loss))
    print("模型在测试集上的准确率: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, i)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, i)

    torch.save(network, "demo/model/nn_{}.pth".format(i))
    print("模型已保存")

# 将模型架构加入tensorboard
input = torch.randn(1, 3, 32, 32)
input = input.to(device)
writer.add_graph(model=network, input_to_model=input)

writer.close()
        
        
