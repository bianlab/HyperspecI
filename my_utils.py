from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
import hdf5storage
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label  + 1e-4) / (label + 1e-4)

        mrae = torch.mean(error)
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error))
        return rmse



class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()

    def forward(self, outputs, labels):
        assert outputs.shape == labels.shape
        num = torch.sum(outputs * labels, 1)
        den = torch.sqrt(torch.sum(outputs * outputs, 1)) * torch.sqrt(torch.sum(labels * labels, 1)) 
        sam = torch.arccos((num) / (den)).mean()
        return sam



class Loss_Fidelity(nn.Module):
    def __init__(self):
        super(Loss_Fidelity, self).__init__()

    def forward(self, outputs, labels):
        assert outputs.shape == labels.shape
        num = torch.sum(outputs * labels, 1)
        den = torch.sqrt(torch.sum(outputs * outputs, 1)) * torch.sqrt(torch.sum(labels * labels, 1)) 
        fidelity = ((num) / (den)).mean()
        return fidelity


class Loss_TV(nn.Module):
    def __init__(self, TVLoss_weight: float=1):
        super(Loss_TV, self).__init__()
        self.weight = TVLoss_weight

    def forward(self, outputs, labels):

        _, _, h, w = outputs.shape

        h_tv = torch.abs(outputs[:, :, 1:, :] - labels[:, :, :h-1, :]).mean()
        w_tv = torch.abs(outputs[:, :, :, 1:] - labels[:, :, :, :w-1]).mean()

        loss = self.weight*(h_tv + w_tv)

        return loss



class Loss_MSE(nn.Module):
    def __init__(self):
        super(Loss_MSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        mse = torch.mean(sqrt_error)
        return mse


class Loss_MAE(nn.Module):
    def __init__(self):
        super(Loss_MAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        l1_error = torch.abs(error)
        mae = torch.mean(l1_error)
        return mae


class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=1.0):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range)
        Itrue = Itrue.reshape(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range)
        Ifake = Ifake.reshape(N, C * H * W)

        mse = nn.MSELoss(reduction='none')
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)

        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)



   
#When traning or testing the HyperspecI-V2, we must eliminate the zero in HSIs
class Loss_MRAE_V2(nn.Module):
    def __init__(self):
        super(Loss_MRAE_V2, self).__init__()

    def forward(self, outputs, labels):
        assert outputs.shape == labels.shape
        
        #Remove zero elements from the denominator
        
        b, c, h, w = labels.size()
        labels = labels.permute(0, 2, 3, 1)
        labels = labels.reshape(-1, c)
        
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = outputs.reshape(-1, c)
        column_sum = labels.sum(dim=1)
        non_zero_columns = column_sum != 0
        non_zero_column_indices = torch.nonzero(non_zero_columns).squeeze()
        filtered_labels = labels[non_zero_column_indices, :]
        filtered_outputs = outputs[non_zero_column_indices, :]
        error = torch.abs(filtered_outputs - filtered_labels  + 1e-4) / (filtered_labels + 1e-4)
        mrae = torch.mean(error)  
        return mrae
    
    

class Loss_SAM_V2(nn.Module):
    def __init__(self):
        super(Loss_SAM_V2, self).__init__()

    def forward(self, outputs, labels):
        assert outputs.shape == labels.shape
        
        
        b, c, h, w = outputs.size()
        labels = labels.permute(0, 2, 3, 1)
        labels = labels.reshape(-1, c)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = outputs.reshape(-1, c)
        
        #Remove zero elements from the denominator
        
        column_sum = labels.sum(dim=1)
        non_zero_columns = column_sum != 0
        non_zero_column_indices = torch.nonzero(non_zero_columns).squeeze()
        
        filtered_labels = labels[non_zero_column_indices, :]
        filtered_outputs = outputs[non_zero_column_indices, :]
        
         
        num = torch.sum(filtered_outputs * filtered_labels, 1)
        den = torch.sqrt(torch.sum(filtered_outputs * filtered_outputs, 1)) * torch.sqrt(torch.sum(filtered_labels * filtered_labels, 1)) 
        sam = torch.arccos((num) / (den)).mean()
        return sam


class Loss_Fidelity_V2(nn.Module):
    def __init__(self):
        super(Loss_Fidelity_V2, self).__init__()

    def forward(self, outputs, labels):
        assert outputs.shape == labels.shape
        
        b, c, h, w = outputs.size()
        labels = labels.permute(0, 2, 3, 1)
        labels = labels.reshape(-1, c)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = outputs.reshape(-1, c)
        
        #Remove zero elements from the denominator
        
        column_sum = labels.sum(dim=1)
        
        non_zero_columns = column_sum != 0
        non_zero_column_indices = torch.nonzero(non_zero_columns).squeeze()
        
        filtered_labels = labels[non_zero_column_indices, :]
        filtered_outputs = outputs[non_zero_column_indices, :]
        
        num = torch.sum(filtered_outputs * filtered_labels, 1)
        den = torch.sqrt(torch.sum(filtered_outputs * filtered_outputs, 1)) * torch.sqrt(torch.sum(filtered_labels * filtered_labels, 1)) 
        fidelity = ((num) / (den)).mean()
        return fidelity
    

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
class Loss_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(Loss_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)



def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close

