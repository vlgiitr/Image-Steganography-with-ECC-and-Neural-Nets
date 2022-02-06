import os
from itertools import chain
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageNet(Dataset):
  def __init__(self, img_dir, normalize = True, resize = True, train = True, test = False):
    self.img_dir = img_dir
    self.normalize = normalize
    self.train = train   #file paths to load image are different for train dataset
    self.test = test     #file paths to load image are different for train dataset
    self.resize = resize
    self.img_list = []
    for image_dir in os.listdir(img_dir):
      self.img_list.append(os.listdir(os.path.join(img_dir, image_dir, 'images')))
    self.img_list = list(chain.from_iterable(self.img_list))
    #print(len(self.img_list))
    #print(self.img_list[0:10])
    self.img_list = self.img_list[0:8] #check by overfitting

  def __len__(self):
    return len(self.img_list)

  def normalize_image(self, image):
    mean = (0,0,0)
    std = (1,1,1)
    brightness, contrast, saturation, hue = 0.25*np.random.random_sample((4,))
    transform = transforms.Compose([transforms.ToTensor(),
                transforms.ColorJitter(brightness= brightness, contrast=contrast, saturation=saturation, hue=hue),
                transforms.Normalize(mean, std)])
    return transform(image) 

  def __getitem__(self, idx):
    if self.train:
      dir, _ = self.img_list[idx].split("_")
      img_path = os.path.join(self.img_dir, dir, 'images', self.img_list[idx])

    elif self.test:
      img_path = os.path.join(self.img_dir, self.img_list[idx])

    image = Image.open(img_path)
    if self.resize:
      image = image.resize((224,224))   #resize to default 224*224 for imagenet images

    image = np.array(image)/255
    if self.normalize is True:
      image = self.normalize_image(image).to(torch.float32) #get images as 32bit torch float tensors
      #print(image.dtype)
    else:
      image = np.transpose(image,(2,0,1))
      image = torch.from_numpy(image).type(torch.float32) #get images as 32bit torch float tensors
      #print(image.dtype)
    return image

  def collate_fn(data):
    """ data: is a list of tuples with (example, label, length) """
    images, targets = zip(*data)
    images = torch.stack(images)
    return (images, targets)

if __name__=='__main__':
  train_dir = "./data/preprocessed/tiny-imagenet-200/train"
  #dataset = ImageNet(img_dir = './data/host_images')
  dataset = ImageNet(img_dir = train_dir, normalize = True)
  train_loader = DataLoader(dataset, batch_size = 8, shuffle=True, num_workers=2, collate_fn= None)
  print('dataloader length (number of batches) -->', len(train_loader))
  print(dataset.img_list)
  batch = next(iter(train_loader))