import numpy as np
import matplotlib.pyplot as plt
from dataset import ImageNet
from encrypt import encrypt_batch
from decrypt import decrypt_batch
from torch.utils.data import DataLoader
from torchvision import utils

def plotGrid(images):
  img_grid = utils.make_grid(images, nrow=4)
  img_grig = np.array(img_grid)
  img_grid = np.transpose(img_grid,(1,2,0))
  plt.imshow(img_grid)
  plt.show()

if __name__ == "__main__":
    # dataset = ImageNet(img_dir = './data/host_images')
    dataset = ImageNet(img_dir = '/mnt/c/Users/ASUS/Projects/Imagenet/n01440764')
    train_loader = DataLoader(dataset, batch_size = 8, shuffle=True, num_workers=2, collate_fn= None)
    images = next(iter(train_loader))
    print(type(images))
    print(images)
    encrypted = encrypt_batch(images)
    plotGrid(encrypted)
    decrypted = decrypt_batch(encrypted)
    plotGrid(decrypted)
