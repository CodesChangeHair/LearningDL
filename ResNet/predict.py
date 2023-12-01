import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create model
net = resnet34(num_classes=5).to(device)

# load model weights
weight_path = "resNet34.pth"
assert os.path.exists(weight_path), "file {} does not exist".format(weight_path)
net.load_state_dict(torch.load(weight_path, map_location=device))

# read classes dict
json_path= "./classes.json"
assert os.path.exists(json_path), "file {} does not exist".format(json_path)
with open(json_path, "r") as json_file:
    classes = json.load(json_file)

data_transform = transforms.Compose(
    [transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# load image
image_path = "sunflower.jpeg"
assert os.path.exists(image_path), "file {} does not exist".format(image_path)
image = Image.open(image_path)
plt.imshow(image)

image = data_transform(image)
# expand batch dimension
image = torch.unsqueeze(image, dim=0)

# prediction
net.eval()
with torch.no_grad():
    output = torch.squeeze(net(image.to(device))).cpu()
predict = torch.softmax(output, dim=0)
predict_class = torch.argmax(predict).numpy()

print_res = "class: {}, prob: {:.3}".format(
    classes[str(predict_class)], predict[predict_class].numpy()
)

plt.title(print_res)
for i in range(len(predict)):
    print("class: {:10} prob: {:.3}".format(
        classes[str(i)], predict[i].numpy()
    ))
plt.show()


