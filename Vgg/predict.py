import os 
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import vgg 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read class indict
json_path = "class_indices.json"
assert os.path.exists(json_path ), "file {} does not exist".format(json_path)
with open(json_path, 'r') as json_file:
    idx_to_class = json.load(json_file)

# create model
model_name = 'vgg16'
net = vgg(model_name=model_name, num_classes=5)
model_path = "./{}Net.pth".format(model_name)
assert os.path.exists(model_path), "file: '{}' dose not exist.".format(model_path)
net.load_state_dict(torch.load(model_path))

# process input image
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# load image
image_path = "sunflower.jpeg"
assert os.path.exists(image_path), "file {} does not exist".format(image_path)
image = Image.open(image_path)
plt.imshow(image)
plt.show()
image = data_transform(image)
# expand batch dimension
image = torch.unsqueeze(image, dim=0)

net.eval()
with torch.no_grad():
    output = net(image)
output = torch.squeeze(output)  # 压缩掉 batch 维度
predict = torch.softmax(output, dim=0)
predict_index = torch.argmax(predict).numpy()
print(idx_to_class[str(predict_index)], predict[predict_index].numpy())

# load image
image_path = "dandelion.jpeg"
assert os.path.exists(image_path), "file {} does not exist".format(image_path)
image = Image.open(image_path)
plt.imshow(image)
plt.show()
image = data_transform(image)
# expand batch dimension
image = torch.unsqueeze(image, dim=0)

net.eval()
with torch.no_grad():
    output = net(image)
output = torch.squeeze(output)  # 压缩掉 batch 维度
predict = torch.softmax(output, dim=0)
predict_index = torch.argmax(predict).numpy()
print(idx_to_class[str(predict_index)], predict[predict_index].numpy())


