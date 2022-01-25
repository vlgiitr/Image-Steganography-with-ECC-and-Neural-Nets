import encrypt as encrypt
from encrypt import random_key
import numpy as np
import ecc
from torch import stack, tensor
from ecc import G, key_length, public_key
from ecc import secret_key as private_key
from ecc import bitcoin_curve as BC

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

def group_pixels(channel, key_length):
  members = key_length//8 - 1
  group_length = 0
  b = []
  grp = ""
  found = []
  for i in range(len(channel)):
    group_length+=1
    found.append(channel[i])
    grp+=get_bin(channel[i], 8)
    if group_length==32:
        b.append(int(grp, 2))
        found = []
        grp = ""
        group_length = 0
  return b

def make_points_and_decrypt(grouped_pixels, public_key):
  keyG = random_key*G
  keyPv = private_key*keyG
  PC = []
  for i in range(0, len(grouped_pixels), 2):
    x = grouped_pixels[i]
    y = BC.p - grouped_pixels[i+1]
    point = ecc.Point(ecc.bitcoin_curve, x, y)
    point = point + keyPv
    point.y = BC.p - point.y
    PC.append(point)

  return PC

def get_channel_for_cipher_image(PC, width):
  pixel_values = []
  for point in PC:
    x, y = point.x, point.y
    bins = get_bin(x, 256)
    nums = [int(bins[i:i+8], 2) for i in range(0, len(bins), 8)]
    pixel_values.append(nums)

    bins = get_bin(y, 256)
    nums = [int(bins[i:i+8], 2) for i in range(0, len(bins), 8)]
    pixel_values.append(nums)

  for i in pixel_values:
    while (len(i)!=32):
      i.append(0)

  pixel_values = np.asarray(pixel_values).flatten().reshape(width, -1)
  return(pixel_values)


def decrypt_image(filepath):
  image, channels, length, width = encrypt.load_image(filepath)
  pixels_list = encrypt.extract_channels(image, channels)

  original_channels = []
  for channel_pixel in pixels_list:
    grouped_pixels = group_pixels(channel_pixel, key_length)
    secret_points = make_points_and_decrypt(grouped_pixels, public_key)
    sec_channel = get_channel_for_cipher_image(secret_points, width)
    original_channels.append(sec_channel)
    
  return np.dstack((tuple(original_channels)))

def decrypt_batch(images):
    img_list = []
    for image in images:
        img_list.append(tensor(decrypt_image(image)))
    return stack(tensors=img_list, dim=0)