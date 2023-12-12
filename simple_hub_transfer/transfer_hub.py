import os
import tensorflow as tf
import tensorflow_hub as hub


# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

class StyleTransfer:

  def __init__(self):

    self.content_path = "content_imgs/"
    self.style_path = "style_imgs/"
    self.hub_model = hub.load('')

  def tensor_to_image(self,tensor):

    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


  def load_img(self,path_to_img):

    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

  def imshow(self,image, title=None):
    if len(image.shape) > 3:
      image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
      plt.title(title)

  def display(self,content,style):

    plt.subplot(1, 2, 1)
    self.imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    self.imshow(style, 'Style Image')

  def transfer(self,content,style):

    stylized_image = self.hub_model(tf.constant(content), tf.constant(style))[0]
    self.tensor_to_image(stylized_image)
    self.imshow(stylized_image)

  def main(self):

    content_image = self.load_img(self.content_path)
    style_image = self.load_img(self.style_path)

    #self.display(content_image,style_image)
    self.transfer(content_image,style_image)

s = StyleTransfer()
s.main()
