import dataset
import model
import os
import argparse
from tqdm import tqdm
from dataset import ImageNet
import itertools
import torchvision
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
import torch.optim as optim
from model import EncryptionModule, DecryptionModule, SegNet, RevealedNetwork

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/steganography')

class Train():
  def __init__(self, encryption_module, decryption_module, O_loader, H_loader):
    self.encryption_module = encryption_module
    self.decryption_module = decryption_module
    self.O_loader = O_loader
    self.H_loader = H_loader

  def lossfunction(self, E, R, H, C):
    l1_loss = torch.norm(H-C) + torch.norm(E-R)
    ce_loss = 0 #according to imagenet labels
    loss = l1_loss + ce_loss
    return loss

  def add_images_to_tensorboard(self, epoch):
    O, H = next(iter(self.O_loader)), next(iter(self.H_loader))
    #print(O.dtype, H.dtype)
    
    C, E = self.encryption_module.encrypt(H = H, O = O)
    S_, R= self.decryption_module.decrypt(C = C)

    img_grid = torchvision.utils.make_grid(O.to(torch.device('cpu')), nrow=4)
    writer.add_image('original image at epoch {}'.format(epoch), img_grid)

    img_grid = torchvision.utils.make_grid(H.to(torch.device('cpu')), nrow=4)
    writer.add_image('host image at epoch {}'.format(epoch), img_grid)

    img_grid = torchvision.utils.make_grid(C.to(torch.device('cpu')), nrow=4)
    writer.add_image('host image at epoch {}'.format(epoch), img_grid)

    img_grid = torchvision.utils.make_grid(torch.tensor(S_).to(torch.device('cpu')), nrow=4)
    writer.add_image('reconstructed secret image at epoch {}'.format(epoch), img_grid)

  def train_one_epoch(self, running_loss):
    #revealed-encrypted, host-container
    for iteration, (O, H) in enumerate(zip(self.O_loader, self.H_loader)):
      optimizer.zero_grad()
      C, E = self.encryption_module.encrypt(H = H, O = O)
      S_, R= self.decryption_module.decrypt(C = C)
      loss = self.lossfunction(E, R, H, C)
      loss.backward(retain_graph=False, create_graph=False)

      writer.add_scalar("Loss", loss, iteration) #add loss to tensorboard

      running_loss = running_loss*0.9 + loss.item()
      # nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01, norm_type=2.0)
      optimizer.step() #update weights
    return running_loss

  def train(self, num_epochs = 300):
    running_loss = 0
    for epoch in tqdm(range(num_epochs)):
      self.add_images_to_tensorboard(epoch)
      running_loss = self.train_one_epoch(running_loss = running_loss)
      #torch.save({'hiding_network_state_dict': self.encryption_module.hiding_network.state_dict(), 
      #'revealing_network_state_dict': self.decryption_module.revealing_network.state_dict(), 
      #'optimizer_state_dict': optimizer.state_dict()}, os.path.join(PATH, 'model_at_epoch_{}.pth'.format(epoch)))
      torch.cuda.empty_cache()

if __name__=='__main__':
    PATH = './checkpoints'
    parser = argparse.ArgumentParser(description='Steganography')
    parser.add_argument('--iterations', type=int, default=10000, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')    
    parser.add_argument('--lr', type=float, default=0.0001, help='lr') 
    parser.add_argument('--host_image_path', type=str, default='./data/host_images', help='path to host images dir')
    parser.add_argument('--original_image_path', type=str, default='./data/original_images', help='path to original images dir')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
    parser.add_argument('--resume_training', type=bool, default=False, help='resume training from a checkpoint')

    args = parser.parse_args()

    iterations = args.iterations
    batch_size = args.batch_size
    lr = args.lr
    host_image_path = args.host_image_path
    original_image_path = args.original_image_path
    beta1 = args.beta1
    beta2 = args.beta2
    resume_training = args.resume_training

    if not os.path.exists('checkpoints'):
      os.makedirs('checkpoints')

    lr = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter('runs/steganography')

    print('Creating Model')

    revealing_network = RevealedNetwork()
    hiding_network = SegNet(num_classes = 3, n_init_features=6, drop_rate=0.5,
                     filter_config=(64, 128, 256, 512, 512))

    hiding_network = hiding_network.to(device)
    revealing_network = revealing_network.to(device)
    
    #create optimizer
    optimizer = torch.optim.Adam(itertools.chain(revealing_network.parameters(), 
      hiding_network.parameters()), lr= lr, betas=(beta1, beta2))

    if resume_training:
      checkpoint = torch.load(os.path.join(PATH, 'model.pth'), map_location = device)
      hiding_network.load_state_dict(checkpoint['hiding_network_state_dict'])
      revealing_network.load_state_dict(checkpoint['revealing_network_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    encryption_module = EncryptionModule(hiding_network, device)
    decryption_module = DecryptionModule(revealing_network, device)

    print('Loading Data')
    H_dataset = ImageNet(img_dir = './data/preprocessed/tiny-imagenet-200/train')
    H_loader = DataLoader(H_dataset, batch_size = 4, shuffle=True, num_workers=2, collate_fn= None)

    O_dataset = ImageNet(img_dir = './data/preprocessed/tiny-imagenet-200/train')
    O_loader = DataLoader(O_dataset, batch_size = 4, shuffle=True, num_workers=2, collate_fn= None)   
    print('Num_Batches: {}'.format(len(O_loader)),"dtype: {}".format(next(iter(H_loader)).dtype))

    print('Starting training')
    trainer = Train(encryption_module=encryption_module, decryption_module=decryption_module, O_loader = O_loader, H_loader = H_loader)
    trainer.train(num_epochs = 300)