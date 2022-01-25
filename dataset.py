import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageNet(Dataset):
  def __init__(self, img_dir, normalize = True):
    self.img_dir = img_dir
    self.img_list = os.listdir(img_dir)
    self.normalize = normalize

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
    img_path = os.path.join(self.img_dir, self.img_list[idx])
    image = Image.open(img_path)
    image = np.array(image)/255
    if self.normalize is True:
      image = self.normalize_image(image)
    else:
      image = np.transpose(image,(2,0,1))
    image = torch.tensor(image, dtype=torch.float32, requires_grad=False) 
    return image

  def collate_fn(data):
    """ data: is a list of tuples with (example, label, length) """
    images, targets = zip(*data)
    images = torch.stack(images)
    return (images, targets)

if __name__=='__main__':
    dataset = ImageNet(img_dir = './data/host_images')
    train_loader = DataLoader(dataset, batch_size = 8, shuffle=True, num_workers=2, collate_fn= None)
    print(len(train_loader))