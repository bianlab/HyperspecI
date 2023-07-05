from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
import hdf5storage

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
        # self.sum += val * n
        # self.count += n

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
        error = torch.abs(outputs - label) / label
        # mrae = torch.mean(error.view(-1))

        # mrae = torch.mean(error.view(-1))

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


# class Loss_PSNR(nn.Module):
#     def __init__(self):
#         super(Loss_PSNR, self).__init__()

#     def forward(self, im_true, im_fake, data_range=1.0):
#         N = im_true.size()[0]
#         C = im_true.size()[1]
#         H = im_true.size()[2]
#         W = im_true.size()[3]
#         Itrue = im_true.clamp(0., 1.).mul_(data_range)
#         Itrue = Itrue.resize(N, C * H * W)
#         Ifake = im_fake.clamp(0., 1.).mul_(data_range)
#         Ifake = Ifake.resize(N, C * H * W)

#         mse = nn.MSELoss(reduce=False)
#         err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)

#         psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
#         return torch.mean(psnr)




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

