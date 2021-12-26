from utils import get_bin
import numpy as np
import imageio
import random
from ecc import key_length, public_key, G, bitcoin_curve
import ecc as ECC

random_key = 18398800208287441760983961865782355938679551141855098527665653562118439265414

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
      if (i==31):
        print(found, b)
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
  for i in range(0, len(grouped_pixels), 2):
    point = ECC.Point(ECC.bitcoin_curve, int(grouped_pixels[i], 2), int(grouped_pixels[i+1], 2))
    if (i<3):
      print(point)
      # print("point IS on the curve: ", (point.y**2 - point.x**3 - 7) % bitcoin_curve.p == 0)
    point = point + keyPb
    if (i<3):
      # print("point IS on the curve: ", (point.y**2 - point.x**3 - 7) % bitcoin_curve.p == 0)
      print(point)
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
      print('Final nums and bins')
      print(nums, bins)
      ok = False
    # print(nums, bins)
    # print(len(nums), len(bins))
    bins = get_bin(y, 256)
    nums = [int(bins[i:i+8], 2) for i in range(0, len(bins), 8)]
    pixel_values.append(nums)
    # print(nums, bins)
    # print(len(nums), len(bins))
    # print(pixel_values)

  for i in pixel_values:
    while (len(i)!=32):
      i.append(0)

  # print(len(pixel_values))
  pixel_values = np.asarray(pixel_values).flatten().reshape(width, -1)
  # print(pixel_values.shape)
  # print(pixel_values)

  return(pixel_values)


def encrypt_image(filepath):
  image, channels, length, width = load_image(filepath)
  pixels_list = extract_channels(image, channels)
  print(pixels_list[0][:32])
  secret_channels = []

  for channel_pixel in pixels_list:
    # print(len(channel_pixel))
    grouped_pixels = group_pixels(channel_pixel, key_length)
    # print(len(grouped_pixels))
    secret_points = make_points_and_encrypt(grouped_pixels, public_key)
    # print(secret_points)
    sec_channel = get_channel_for_cipher_image(secret_points, width)
    secret_channels.append(sec_channel)

  return np.dstack((tuple(secret_channels)))

encrypted_image = encrypt_image('lena.png')
# print(encrypted_image)
image, channels, length, width = load_image('lena.png')
# print(*extract_channels(image, channels)[0][:320], sep=", ")

from PIL import Image
im = Image.fromarray((encrypted_image).astype(np.uint8))
im.save('enc.png')

