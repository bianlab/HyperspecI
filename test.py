import torch
import argparse
import os
import time
import torch.backends.cudnn as cudnn
from my_utils import AverageMeter, initialize_logger, save_checkpoint, Loss_RMSE, Loss_PSNR
import torch.utils.data
from architecture import model_generator
import h5py
import numpy as np
import cv2

parser = argparse.ArgumentParser(description="Mosaic to HSI")
parser.add_argument("--method", type=str, default='srnet', help='Model')
parser.add_argument("--gpu_id", type=str, default='3', help='path log files')


parser.add_argument("--pretrained_model_path", type=str, default='./model_zoo/SRNet_lib.pth', help='path log files')
parser.add_argument("--image_folder", type=str, default='./Test_Source_Lib/Image/', help='path log files')
parser.add_argument("--save_folder", type=str, default='./Test_Source_Lib/Output_HSI/', help='path log files')


# parser.add_argument("--pretrained_model_path", type=str, default='./model_zoo/SRNet_sun.pth', help='path log files')
# parser.add_argument("--image_folder", type=str, default='./Test_Source_Sun/Image/', help='path log files')
# parser.add_argument("--save_folder", type=str, default='./Test_Source_Sun/Output_HSI/', help='path log files')

opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
# srnet_61_noise5_1000_100_Train_0417_overexposure
def data_preprocessing(img, sigma):
    noise = torch.zeros_like(img)
    noise.normal_(0, sigma)
    out = img + noise
    return out

def main():
    cudnn.benchmark = True

    model = model_generator(opt.method, opt.pretrained_model_path)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    if torch.cuda.is_available():
        model.cuda()

    start_x = 0
    start_y = 400
    #scene1, scene2, scene3, scene4, scene5, scene6

    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    
    test_list = os.listdir(opt.image_folder)
    test_list.sort()
    for i in range(len(test_list)):
        strat_time = time.time()
        bmp = cv2.imread(opt.image_folder + test_list[i])[:, :, 0]
        bmp = bmp[start_x:, start_y:]
        print('bmp>>>>>>>>>>',test_list[i], bmp.dtype, bmp.shape, bmp.max(), bmp.mean(), bmp.min())
        bmp = bmp / bmp.max()
        bmp = bmp.astype(np.float32)

        print('bmp', bmp.dtype, bmp.shape, bmp.max(), bmp.mean(), bmp.min())
        mos = torch.from_numpy(bmp)
        print('mos', mos.dtype, mos.shape, mos.max(), mos.mean(), mos.min())
        hsi = Testdate(mos, model)
        print('save_hsi', hsi.dtype, hsi.shape, hsi.max(), hsi.mean(), hsi.min())

        
        f = h5py.File(opt.save_folder + 'HSI_R_' + test_list[i][:-4] + '.h5', 'w')
        f['mos'] = bmp
        f['hsi_R'] = hsi
        f.close()

        # hdf5storage.savemat(save_folder + 'HSI_R_' + test_list[i][:-4] + '.mat', {'mos':bmp, 'hsi_R':hsi})

        end_time = time.time()
        print('平均运行时间：', test_list[i], (end_time - strat_time))


  

def Testdate(mos, model):
    model.eval()
    mos = mos.unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        outputs = model(mos)
        print('outputs>>>>>>>>>>', outputs.shape, outputs.max(), outputs.mean(), outputs.min())
        outputs = outputs / outputs.max()
        outputs = torch.maximum(outputs, torch.tensor(0))
        outputs = outputs.squeeze().cpu().numpy()
        print('outputs', outputs.shape, outputs.max(), outputs.mean(), outputs.min())
        
    return outputs


if __name__ == '__main__':
    main()


