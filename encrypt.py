import numpy as np
import imageio
import random
from torch import stack, tensor
from ecc import key_length, public_key, G, bitcoin_curve
import ecc as ECC

random_key = 18398800208287441760983961865782355938679551141855098527665653562118439265414

def get_bin(x, n=0):
    """
    Get the binary representation of x.

    Parameters
    ----------
    x : int
    n : int
        Minimum number of digits. If x needs less digits in binary, the rest
        is filled with zeros.

    Returns
    -------
    str
    """
    return format(x, 'b').zfill(n)

def load_tensor(image):
  #image = tensor.numpy()
  channels = image.shape[-1] if image.ndim == 3 else 1
  length = image.shape[0]
  width = image.shape[1]

  return image, channels, length, width

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
  # channel = [channel[i]+random.randint(1, 2) for i in range(len(channel))]
  members = key_length//8 - 1
  b = []
  grp = ""
  group_length = 0
  found = []
  for i in range(len(channel)):
    group_length += 1
    found.append(channel[i])
    grp+=get_bin(channel[i], 8)
    if group_length==32:
      b.append(grp)
      grp = ""
      group_length = 0
  return b

def make_points_and_encrypt(grouped_pixels, public_key):
  # Generate a random number
  # random_key = random.randint(1, ECC.bitcoin_gen.n-1)
  # random_key = 18398800208287441760983961865782355938679551141855098527665653562118439265414 # For consistency

  keyPb = random_key*public_key
  # print('keyPb', keyPb)
  keyG = random_key*G

  PC = []
  for i in range(0, len(grouped_pixels)-1, 2):
    point = ECC.Point(ECC.bitcoin_curve, int(grouped_pixels[i], 2), int(grouped_pixels[i+1], 2))
    point = point + keyPb
    PC.append(point)
  return PC

def get_channel_for_cipher_image(PC, width):
  pixel_values = []
  ok = True
  for point in PC:
    x, y = point.x, point.y
    bins = get_bin(x, 256)
    nums = [int(bins[i:i+8], 2) for i in range(0, len(bins), 8)]
    pixel_values.append(nums)
    if (ok):
      ok = False

    bins = get_bin(y, 256)
    nums = [int(bins[i:i+8], 2) for i in range(0, len(bins), 8)]
    pixel_values.append(nums)

  for i in pixel_values:
    while (len(i)!=32):
      i.append(0)

  #print("error here #####################", np.array(pixel_values).shape, "width {}".format(width))
  pixel_values = np.asarray(pixel_values).flatten().reshape(width, -1)
  return pixel_values


def encrypt_image(image_tensor):
  image, channels, length, width = load_tensor(image_tensor)
  pixels_list = extract_channels(image, channels)
  secret_channels = []

  for channel_pixel in pixels_list:
    grouped_pixels = group_pixels(channel_pixel, key_length)
    secret_points = make_points_and_encrypt(grouped_pixels, public_key)
    sec_channel = get_channel_for_cipher_image(secret_points, width)
    secret_channels.append(sec_channel)

  return np.dstack((tuple(secret_channels)))

def encrypt_batch(images):
    print(images.dtype)
    img_list = []
    for image in images:
        img_list.append(tensor(encrypt_image(image)))
    return stack(tensors=img_list, dim=0)
