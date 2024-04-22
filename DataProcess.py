import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

class Data_Process(object):
    def __init__(self):
        self.noise_sigma = 0
        self.hsi_max = []

    def add_noise(self, inputs, sigma):
        noise = torch.zeros_like(inputs)
        noise.normal_(0, sigma)
        noisy = inputs + noise
        noisy = torch.clamp(noisy, 0, 1.0)
        return noisy

    #Randomly extract sub-patches required for training from the original patch
    def get_random_mask_patches(self, mask, image_size, patch_size, batch_size):

        masks = []
        for i in range(batch_size):
            random_h = random.randint(0, image_size[0] - patch_size[0] -1)
            random_w = random.randint(0, image_size[1] - patch_size[1] -1)
            mask_patch = mask[:, random_h:random_h + patch_size[0], random_w:random_w + patch_size[1]]
            mask_patch = mask_patch / mask_patch.max()
            masks.append(mask_patch)
            
        mask_patches = torch.stack(masks, dim=0)
        return mask_patches
            
        
    #Forward model of snapshot hyperspectral imaging for generating input synthesized measurements from hyperspectral targets
    def get_mos_hsi(self, hsi, mask, sigma=0, mos_size=2048, hsi_input_size=512, hsi_target_size=512, init_div_rat=10):
        if not hsi_input_size == hsi_target_size:
            hsi_out = self.extend_spatial_resolution(hsi, extend_rate=hsi_target_size / hsi_input_size)
        else:
            hsi_out=hsi

        if not mos_size == hsi_input_size:
            hsi_expand = self.extend_spatial_resolution(hsi, extend_rate=mos_size / hsi_input_size)
        else:
            hsi_expand=hsi

        mos = torch.sum(hsi_expand * mask, dim=1).unsqueeze(1)
        mos_max = torch.max(mos.view(mos.shape[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)

        #normalize the input and target data using the adaptive variable
        output_hsi = hsi_out / mos_max * init_div_rat
        input_mos = mos / mos_max


        if isinstance(sigma, tuple):
            select_noise_sigma = sigma[random.randint(0, len(sigma) - 1)]
        else: 
            select_noise_sigma = sigma

        input_mos = self.add_noise(input_mos, select_noise_sigma)

        return input_mos, output_hsi


    def extend_spatial_resolution(self, hsi, extend_rate):
        hsi_extend = torch.nn.functional.interpolate(hsi, recompute_scale_factor=True, scale_factor=extend_rate)
        return hsi_extend



class Image_Cut(object):
    def __init__(self, image_size, patch_size, stride):
        self.patch_size = patch_size
        self.stride = stride 
        self.image_size = image_size

        self.patch_number = []
        self.hsi_max = []

    def image2patch(self, image):
        '''
        image_size = C, H, W
        '''
        patch_size = self.patch_size
        stride = self.stride

        c, h, w = image.shape
        image = image.unsqueeze(0)
        range_h = np.arange(0, h-patch_size[0], stride)
        range_w = np.arange(0, w-patch_size[1], stride)

        range_h = np.append(range_h, h-patch_size[0])
        range_w = np.append(range_w, w-patch_size[1])
        patches = []
        for m in range_h:
            for n in range_w:
                patches.append(image[:, :, m : m + patch_size[0], n : n + patch_size[1]])

        return torch.cat(patches, 0)
    def patch2image(self, patches):

        patch_size = self.patch_size
        stride = self.stride
        c = patches.shape[1]
        h, w = self.image_size

        res = torch.zeros((c, h, w)).to(patches.device)
        weight = torch.zeros((c, h, w)).to(patches.device)

        range_h = np.arange(0, h-patch_size[0], stride)
        range_w = np.arange(0, w-patch_size[1], stride)


        range_h = np.append(range_h, h-patch_size[0])
        range_w = np.append(range_w, w-patch_size[1])

        index = 0

        for m in range_h:
            for n in range_w:
                res[:, m : m + patch_size[0], n : n + patch_size[1]] = res[:, m : m + patch_size[0], n : n + patch_size[1]] + patches[index, ...]

                weight[:, m : m + patch_size[0], n : n + patch_size[1]] = weight[:, m : m + patch_size[0], n : n + patch_size[1]] + 1
                index = index+1

        image = res / weight
        return image








