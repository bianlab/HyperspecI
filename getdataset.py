from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch
import random
import os
import hdf5storage
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate
import torch.nn.functional as F
import h5py


class TrainDataset_V1(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_path = data_path
        self.patch_size = patch_size

        data_list = os.listdir(data_path)
        data_list.sort()

        self.data_list = data_list
        self.img_num = len(self.data_list)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):


        f = h5py.File(self.data_path + self.data_list[idx], 'r')
        hsi = f['hsi'][:]
        f.close()

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        random_h = random.randint(0,hsi.shape[1] - patch_size_h -1)
        random_w = random.randint(0,hsi.shape[2] - patch_size_w -1)
        output_hsi = hsi[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        output_hsi = output_hsi.astype(np.float32)
        output_hsi = output_hsi / output_hsi.max()

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num

class ValidDataset_V1(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_paths = []
        self.patch_size = patch_size

        data_list = os.listdir(data_path)
        data_list.sort()
        for i in range(len(data_list)):

            self.data_paths.append(data_path + data_list[i])

        self.img_num = len(self.data_paths)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):

        f = h5py.File(self.data_paths[idx], 'r')
        hsi = f['hsi'][:]
        f.close()

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

       
        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        random_h = random.randint(0, hsi.shape[1] - patch_size_h -1)
        random_w = random.randint(0, hsi.shape[2] - patch_size_w -1)
        output_hsi = hsi[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        output_hsi = output_hsi.astype(np.float32)
        output_hsi = output_hsi / output_hsi.max()

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num






class TrainDataset_V2(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_path = data_path
        self.patch_size = patch_size
        self.select_index = np.concatenate((np.arange(0,61,1), np.arange(62, 132, 2)))
        data_list = os.listdir(data_path)
        data_list.sort()

        self.data_list = data_list
        self.img_num = len(self.data_list)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):


        f = h5py.File(self.data_path + self.data_list[idx], 'r')
        hsi = f['hsi'][:]
        hsi = hsi[self.select_index, :, :]
        f.close()

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]


        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        random_h = random.randint(0,hsi.shape[1] - patch_size_h -1)
        random_w = random.randint(0,hsi.shape[2] - patch_size_w -1)
        output_hsi = hsi[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        output_hsi = output_hsi.astype(np.float32)
        output_hsi = output_hsi / output_hsi.max()

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num

class ValidDataset_V2(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_paths = []
        self.patch_size = patch_size
        
        self.select_index = np.concatenate((np.arange(0,61,1), np.arange(62, 132, 2)))

        data_list = os.listdir(data_path)
        data_list.sort()
        for i in range(len(data_list)):

            self.data_paths.append(data_path + data_list[i])

        self.img_num = len(self.data_paths)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):

        f = h5py.File(self.data_paths[idx], 'r')
        hsi = f['hsi'][:]
        hsi = hsi[self.select_index, :, :]
        f.close()

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

       
        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        random_h = random.randint(0, hsi.shape[1] - patch_size_h -1)
        random_w = random.randint(0, hsi.shape[2] - patch_size_w -1)
        output_hsi = hsi[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        output_hsi = output_hsi.astype(np.float32)
        output_hsi = output_hsi / output_hsi.max()

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num





class TestDataset_MOS(Dataset):
    def __init__(self, data_path, data_list, start_dir, image_size, arg=False):

        self.arg = arg
        self.data_path = data_path

        self.start_dir = start_dir
        self.image_size = image_size

        self.data_list = data_list

        self.MOS_list = []

        for i in range(len(data_list)):

            bmp = cv2.imread(self.data_path + self.data_list[i])[:, :, 0]
            bmp = bmp[self.start_dir[0]:self.start_dir[0]+self.image_size[0], self.start_dir[1]:self.start_dir[1] + self.image_size[1]]
            bmp = bmp / bmp.max()
            bmp = bmp.astype(np.float32)
            mos = np.expand_dims(bmp, axis=0)
            self.MOS_list.append(mos)
            
        self.img_num = len(self.data_list)

    def __getitem__(self, idx):
        mos_name = self.data_list[idx]
        mos = self.MOS_list[idx]

        return np.ascontiguousarray(mos), mos_name

    def __len__(self):
        return self.img_num

