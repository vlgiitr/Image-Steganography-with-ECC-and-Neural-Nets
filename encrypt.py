from utils import get_bin
import numpy as np
import imageio
import random
from ecc import key_length, public_key
import ecc as ECC

def load_image(filepath):
  image = imageio.imread(filepath)
  channels = image.shape[-1] if image.ndim == 3 else 1
  length = image.shape[0]
  width = image.shape[1]

  return image, channels, length, width

def extract_channels(image, channels):
  pixels = []
  for channel in range(channels):
    pixels.append(image[:, :, channel].flatten())
  return pixels


def group_pixels(channel, key_length):
  # Randomly add 1 or 2 to the pixel values
  channel = [channel[i]+random.randint(1, 2) for i in range(len(channel))]
  members = key_length//8 - 1
  b = []
  grp = ""
  for i in range(len(channel)):
    grp+=get_bin(channel[i], 8)
    if i%(members+1)==0 and i!=0:
      b.append(grp)
      grp = ""

  b.append(grp.zfill(key_length-8))

  return b


def make_points_and_encrypt(grouped_pixels, public_key):
  # Generate a random number
  # random_key = random.randint(1, ECC.bitcoin_gen.n-1)
  random_key = 18398800208287441760983961865782355938679551141855098527665653562118439265414 # For consistency

  keyPb = random_key*public_key

  PC = []
  for i in range(0, len(grouped_pixels), 2):
    point = ECC.Point(ECC.bitcoin_curve, int(grouped_pixels[i], 2), int(grouped_pixels[i+1], 2))
    point = point + keyPb
    PC.append(point)

  return PC


def get_channel_for_cipher_image(PC, width):
  pixel_values = []
  for point in PC:
    x, y = point.x, point.y
    bins = get_bin(x)
    nums = [int(bins[i:i+8], 2) for i in range(0, len(bins), 8)]
    pixel_values.append(nums)
    bins = get_bin(y)
    nums = [int(bins[i:i+8], 2) for i in range(0, len(bins), 8)]
    pixel_values.append(nums)

  for i in pixel_values:
    while (len(i)!=32):
      i.append(0)

  pixel_values = np.asarray(pixel_values).flatten().reshape(width, -1)

  return(pixel_values)


def encrypt_image(filepath):
  image, channels, length, width = load_image(filepath)
  pixels_list = extract_channels(image, channels)

  secret_channels = []

  for channel_pixel in pixels_list:
    grouped_pixels = group_pixels(channel_pixel, key_length)
    secret_points = make_points_and_encrypt(grouped_pixels, public_key)
    sec_channel = get_channel_for_cipher_image(secret_points, width)
    secret_channels.append(sec_channel)

  return np.dstack((tuple(secret_channels)))


# encrypted_image = encrypt_image('doge.png')
# print(encrypted_image)

# # from matplotlib import pyplot as plt
# # plt.imshow(encrypted_image, interpolation='nearest')
# # plt.show()
