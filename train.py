import dataset
import model
import os
import argparse
from tqdm import tqdm
from dataset import dataloader
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/exp_1')

def lossfunction(encrypted_images, revealed_images, host_images, container_images):
  l1_loss = torch.norm(host_images-container_images) + torch.norm(encrypted_images-revealed_images)
  ce_loss = 0 #according to imagenet labels
  loss = l1_loss + ce_loss
  return loss

lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr= lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, min_lr=0.00001, eps=1e-08, verbose=True)

checkpoint = torch.load(os.path.join(PATH, 'model.pth'), map_location = device)
hiding_network.load_state_dict(checkpoint['hiding_network_state_dict'])
revealing_network.load_state_dict(checkpoint['revealing_network_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def add_images_to_tensorboard(epoch, encryption_module, decryption_module, hiding_network, revealing_network):
  original_images, host_images = next(iter(train_dataloader))
  container_images, encrypted_images = encryption_module.encrypt(
      host_images = host_images, original_images = original_images, hiding_network = hiding network
      )
  reconstructed_secret_images, revealed_images = decryption_module.decrypt(
      container_images = container_images, revealing_network = revealing network
      )

  img_grid = torchvision.utils.make_grid(original_images.to(torch.device('cpu')), nrow=4)
  writer.add_image('original image at epoch {}'.format(epoch), img_grid)

  img_grid = torchvision.utils.make_grid(reconstructed_secret_images.to(torch.device('cpu')), nrow=4)
  writer.add_image('reconstructed secret image at epoch {}'.format(epoch), img_grid)

def train_one_epoch(epoch, train_dataloader, hiding_network, revealing_network, running_loss):
  #reveled-encrypted, host-container
  for iteration, images in enumerate(train_dataloader):
    optimizer.zero_grad()
    original_images, host_images = next(iter(train_dataloader))
    container_images, encrypted_images = encryption_module.encrypt(
        host_images = host_images, original_images = original_images, hiding_network = hiding network
        )
    reconstructed_secret_images, revealed_images = decryption_module.decrypt(
        container_images = container_images, revealing_network = revealing network
        )
    loss = lossfunction(encrypted_images, revealed_images, host_images, container_images)
    loss.backward(retain_graph=False, create_graph=False)

    writer.add_scalar("Loss", loss, iteration) #add loss to tensorboard

    running_loss = running_loss*0.9 + loss.item()
    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01, norm_type=2.0)
    optimizer.step() #update weights
  return running_loss

def train(num_epochs = 300, train_dataloader, encryption_module, decryption_module, hiding_network, revealing_network):
  running_loss = 0
  for epoch in tqdm(range(num_epochs)):
    add_images_to_tensorboard(epoch, encryption_module, decryption_module, hiding_network, revealing_network)
    running_loss = train_one_epoch(epoch = epoch, train_dataloader = train_dataloader, hiding_network = hiding_network, revealing_network = revealing_network, running_loss = running_loss)
    torch.save({'hiding_network_state_dict': hiding_network.state_dict(), 'hiding_network_state_dict': hiding_network.state_dict(), 'optimizer_state_dict': optimizer_state_dict()}, os.path.join(PATH, 'model at epoch {}.pth'.format(epoch)))
    torch.cuda.empty_cache()


if __name__=='__main__':
    PATH = './checkpoints'
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--iterations', type=int, default=300000, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size') 
    parser.add_argument('--seq_len', type=int, default=10, help='seq_len')     
    parser.add_argument('--layer_dim', type=int, default=128, help='layer_dim')   
    parser.add_argument('--lr', type=float, default=0.0001, help='lr')   
    parser.add_argument('--D_G_training_ratio', type=int, default=10, help='D_G_training_ratio')
    parser.add_argument('--gradient_penalty_coeff', type=int, default=10, help='gradient_penalty_coeff')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2')

    args = parser.parse_args()

    if not os.path.exists('checkpoints'):
      os.makedirs('checkpoints')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter('runs/steganography')
    iterations = args.iterations
    batch_size = args.batch_size
    seq_len = args.seq_len
    layer_dim = args.layer_dim
    lr = args.lr
    D_G_training_ratio = args.D_G_training_ratio
    gradient_penalty_coeff = args.gradient_penalty_coeff
    beta1 = args.beta1
    beta2 = args.beta2

    print('Creating Model')
    encryption_module = model.EncryptionModule()
    decryption_module = model.DecryptionModule()
    revealing_network = model.RevealedNetwork()
    hiding_network = model.SegNet(num_classes = 3, n_init_features=6, drop_rate=0.5,
                     filter_config=(64, 128, 256, 512, 512))

    hiding_network = hiding_network.to(device)
    revealing_network = revealing_network.to(device)

    #create optimizer
    optimizer = torch.optim.Adam(itertools.chain(revealing_network.parameters(), hiding_network.parameters()), lr= lr, betas=(beta1, beta2))
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, min_lr=0.00001, eps=1e-08, verbose=True)

    print('Loading Data')
    dataloader = dataloader(batch_size = batch_size)
    print('Num_Batches: {}'.format(len(dataloader)))

    print('Starting training')
    train(num_epochs = 300, train_dataloader = dataloader, encryption_module = encryption_module, 
    decryption_module = decryption_module, hiding_network = hiding_network, revealing_network = revealing_network)