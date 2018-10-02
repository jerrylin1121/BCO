import tensorflow as tf
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_filename", default="input/demonstration.txt", help="the demonstration inputs")
parser.add_argument("--mode", default="train", choices=["train", "test"], required=True)
parser.add_argument("--model_dir", help="where to save/restore the model")

parser.add_argument("--maxits", type=int, default=1000, help="the number of training iteration")
parser.add_argument("--M", type=int, default=1000, help="the number of post demonstration examples")

parser.add_argument("--batch_size", type=int, default=32, help="number of examples in batch")
parser.add_argument("--lr", type=float, default=0.002, help="initial learning rate for adam SGD")

parser.add_argument("--save_freq", type=int, default=100, help="save model every save_freq iterations, 0 to disable")
parser.add_argument("--print_freq", type=int, default=50, help="print current reward and loss every print_freq iterations, 0 to disable")

args = parser.parse_args()

def weight_initializer():
  return tf.truncated_normal_initializer(stddev=0.1)

def bias_initializer():
  return tf.constant_initializer(0.01)

def get_shuffle_idx(num, batch_size):
  tmp = np.arange(num)
  np.random.shuffle(tmp)
  split_array = []
  cur = 0
  while num > batch_size:
    num -= batch_size
    if(num != 0):
      split_array.append(cur+batch_size)
      cur+=batch_size
  return np.split(tmp, split_array)
