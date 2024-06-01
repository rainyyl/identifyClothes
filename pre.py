from __future__ import absolute_import, division, print_function, unicode_literals

# 载入TensorFlow 和 tf.keras
import tensorflow as tf
from tensorflow import keras

# 载入 辅助包
import numpy as np
import matplotlib.pyplot as plt
# 输出当前的tf版本
# print(tf.__version__)
#从网站下载数据=========================================
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
