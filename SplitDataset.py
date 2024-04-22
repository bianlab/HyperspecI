import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mp
import scipy.io as scio
import os
import cv2
import time
import random
import shutil
import argparse


parser = argparse.ArgumentParser(description="Training data and validata partition")

#Paths of input and output data
parser.add_argument("--data_folder", type=str, default= './Dataset_Train/HSI_400_1000/HSI_all/', help='Original data folder')

parser.add_argument("--train_folder", type=str, default= './Dataset_Train/HSI_400_1000/Train/', help='Training data folder')
parser.add_argument("--test_folder", type=str, default= './Dataset_Train/HSI_400_1000/Valid/', help='Validata folder')

opt = parser.parse_args()


path_data = os.listdir(opt.data_folder)

random.shuffle(path_data)
train_ratio = 0.9 #the ratio of training data
data_nums = len(path_data)
train_nums = int(data_nums * train_ratio)
train_sample = random.sample(path_data,train_nums)
test_sample = list(set(path_data)-set(train_sample))
print(len(path_data))
print(len(train_sample))
print(len(test_sample))


#Move the original data into the taining data folder
for k in train_sample:
     shutil.move(os.path.join(opt.data_folder,k),os.path.join(opt.train_folder,k))


# Move the original data into the validata folder
for k in test_sample:
    
    shutil.move(os.path.join(opt.data_folder,k),os.path.join(opt.test_folder,k))

