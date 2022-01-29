import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import idctn, dctn
from torch.fft import fft2, ifft2

class EncryptionModule():
    def __init__(self, hiding_network, device):
        self.hiding_network = hiding_network
        self.device = device

    def encrypt(self, H, O):
        """ 
        input: original secret images O
        output: container images C
        """
        #S = fft2(O)
        S = dctn(np.array(O), axes=(-1, -2))
        E = torch.tensor(S) #ecc(secret_images)
        images=[]
        for x,y in zip(E, H):
            images.append(torch.cat((x,y)))

        #print(torch.stack(images, dim=0).size())
        images = torch.stack(images, dim=0).to(self.device)
        C = self.hiding_network(images)
        return (C, E)

class DecryptionModule():
    def __init__(self, revealing_network, device):
        self.revealing_network = revealing_network
        self.device = device
    
    def decrypt(self, C):
        """ 
        input: Container images C
        output: Secret images S'
        """
        R = self.revealing_network(C)
        #x = inv_ecc(R.clone().detach().to(torch.device('cpu')))
        x = R.clone().detach().to(torch.device('cpu'))
        S_ = idctn(np.array(x), axes=(-1, -2)) #inverse dct to plot image
        return (S_, R)

class SegNet(nn.Module):
    """SegNet: A Deep Convolutional Encoder-Decoder Architecture for
    Image Segmentation. https://arxiv.org/abs/1511.00561
    See https://github.com/alexgkendall/SegNet-Tutorial for original models.
    Args:
        num_classes (int): number of classes to segment
        n_init_features (int): number of input features in the fist convolution
        drop_rate (float): dropout rate of each encoder/decoder module
        filter_config (list of 5 ints): number of output features at each level
    """
    def __init__(self, num_classes, n_init_features=1, drop_rate=0.5,
                 filter_config=(64, 128, 256, 512, 512)):
        super(SegNet, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (n_init_features,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(0, 5):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i], drop_rate))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i], drop_rate))

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], num_classes, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(0, 5):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 5):
            feat = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])

        return self.classifier(feat)

class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU(inplace=True)]
            if n_blocks == 3:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()

class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
        super(_Decoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_in_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_in_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU(inplace=True)]
            if n_blocks == 3:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)

class RevealedNetwork(nn.Module):
    """
    Revealed network decodes the container image to give the 
    encrypted secret image and host image.
    """
    def __init__(self):
      super(RevealedNetwork, self).__init__()
      self.conv = nn.Sequential(
      nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=3, padding= 'same', bias=False),
      nn.BatchNorm2d(num_features = 64),
      nn.ReLU(inplace=True),

      nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, padding= 'same', bias=False),
      nn.BatchNorm2d(num_features = 128),
      nn.ReLU(inplace=True),

      nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=3, padding= 'same', bias=False),
      nn.BatchNorm2d(num_features = 256),
      nn.ReLU(inplace=True),

      nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size=3, padding= 'same', bias=False),
      nn.BatchNorm2d(num_features = 128),
      nn.ReLU(inplace=True),

      nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size=3, padding= 'same', bias=False),
      nn.BatchNorm2d(num_features = 64),
      nn.ReLU(inplace=True),
      
      nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size=3, padding= 'same', bias=False),
      nn.BatchNorm2d(num_features = 3),
      nn.ReLU(inplace=True),)

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    revealing_network = RevealedNetwork()
    hiding_network = SegNet(num_classes = 3, n_init_features=6, drop_rate=0.5,
                     filter_config=(64, 128, 256, 512, 512))
    
    print("revealing network architecture")
    print("-----------------------------------------------------------------")
    print(revealing_network)
    print("-----------------------------------------------------------------")
    print("hiding network architecture")
    print("-----------------------------------------------------------------")
    print(hiding_network)