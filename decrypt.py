import encrypt as encrypt
from encrypt import random_key
import numpy as np
import utils
from ecc import key_length, public_key


def group_pixels(channel, key_length):
  members = key_length//8 - 1
  b = []
  grp = ""
  for i in range(len(channel)):
    grp+=utils.get_bin(channel[i], 8)
    if i%(members+1)==0 and i!=0:
      b.append(int(grp, 2))
      grp = ""

  b.append(int(grp.zfill(key_length-8), 2))
  # print(b)

  return b


def decrypt_image(filepath):
  image, channels, length, width = encrypt.load_image(filepath)
  pixels_list = encrypt.extract_channels(image, channels)

  original_channels = []

  for channel_pixel in pixels_list:
    grouped_pixels = group_pixels(channel_pixel, key_length)
    # secret_points = make_points_and_encrypt(grouped_pixels, public_key)
    # sec_channel = get_channel_for_cipher_image(secret_points, width)
    # secret_channels.append(sec_channel)

  return np.dstack((tuple(original_channels)))


decrypted_image = decrypt_image('enc.png')
print(decrypted_image)

# from matplotlib import pyplot as plt
# # plt.imshow(encrypted_image, interpolation='nearest')
# # plt.show()
# plt.savefig('enc.png')
