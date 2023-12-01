import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# load images
# 需要遍历预测图像的文件夹

images_root = "dataset/predict"
assert os.path.exists(images_root), "file {} does not exist".format(images_root)

# 读取文件夹下所有图像路径
image_path_list = [os.path.join(images_root, i) for i in os.listdir(images_root)]

# read classes dict
json_path = "./classes.json"
assert os.path.exists(json_path), "file {} does not exist".format(json_path)

with open(json_path, "r") as json_file:
    classes_dict = json.load(json_file)

# create model
net = resnet34(num_classes=5).to(device)

# load model weights
weights_path = "./resNet34.pth"
assert os.path.exists(weights_path), "file {} does not exist".format(weights_path)
net.load_state_dict(torch.load(weights_path, map_location=device))


# prediction
batch_size = 8
with torch.no_grad():
    for ids in range(0, len(image_path_list) // batch_size):
        image_list = []
        for image_path in image_path_list[ids * batch_size: (ids + 1) * batch_size]:
            image = Image.open(image_path)
            image = data_transform(image)
            image_list.append(image)
        
        # 将image_list中所有图片打包成一个batch
        batch_images = torch.stack(image_list, dim=0)
        # predict classes
        output = net(batch_images.to(device)).cpu()
        predict = torch.softmax(output, dim=1)
        probs, classes = torch.max(predict, dim=1)
        
        for idx, (prob, cla) in enumerate(zip(probs, classes)):
            print("image: {} class: {} prob: {:.3}".format(
                image_path_list[ids * batch_size + idx],
                classes_dict[str(cla.numpy())],
                prob.numpy()
            ))