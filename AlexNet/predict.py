from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np
import torch

from model import AlexNet

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))  # (c, h, w) --> (h, w, c) 以符合plt.imshow()
    plt.imshow(npimg)
    plt.show()

# image data process
data_transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load json file
try:
    json_file = open("class_indices.json", 'r')
    class_indices = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)
else:
    print("Load json")

# load model
net = AlexNet(num_classes=5)
model_parameter_path = "./AlexNet.pth"
net.load_state_dict(torch.load(model_parameter_path))

# load image
image = Image.open('sunflower.jpeg')
image = data_transform(image)
imshow(image)
image = torch.unsqueeze(image, dim=0)

net.eval()  # 关闭dropout
with torch.no_grad():
    output = net(image)
output = torch.squeeze(output)  # 压缩掉 batch 维度
predict = torch.softmax(output, dim=0)
predict_index = torch.argmax(predict).numpy()
print(class_indices[str(predict_index)], predict[predict_index].item())
print()

# load image
image = Image.open('dandelion.jpeg')
image = data_transform(image)
imshow(image)
image = torch.unsqueeze(image, dim=0)

net.eval()  # 关闭dropout
with torch.no_grad():
    output = net(image)
output = torch.squeeze(output)  # 压缩掉 batch 维度
predict = torch.softmax(output, dim=0)
predict_index = torch.argmax(predict).numpy()
print(class_indices[str(predict_index)], predict[predict_index].item())
    
    