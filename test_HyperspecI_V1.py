import hdf5storage
import torch
import argparse
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from getdataset import TestDataset_MOS
from my_utils import initialize_logger
import torch.utils.data
from architecture import model_generator
import numpy as np
import h5py

parser = argparse.ArgumentParser(description="Reconstruct hypersepctral images from measurements")
parser.add_argument("--method", type=str, default='V1_srnet', help='Model')
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--mask_path", type=str, default='./MASK/Mask_HyperspecI_V1.mat', help='path log files')
parser.add_argument("--start_dir", type=int, default=(0, 0), help="size of test image coordinate")
parser.add_argument("--image_size", type=int, default=(2048, 2048), help="size of test image")
parser.add_argument("--pretrained_model_path", type=str, default='./Model_zoo/SRNet_V1.pth', help='path log files')
parser.add_argument("--image_folder", type=str, default= './Measurements_Test/HyperspecI_V1/', help='path log files')
parser.add_argument("--save_folder", type=str, default= './Measurements_Test/Output_HyperspecI_V1/', help='path log files')


opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():
    cudnn.benchmark = True
    mask_init = hdf5storage.loadmat(opt.mask_path)['mask']
    mask = np.maximum(mask_init, 0)
    mask = mask / mask.max()
    mask = mask.astype(np.float32)
    mask = torch.from_numpy(mask)
    mask = mask.cuda()
    mask = mask.unsqueeze(0)
    model = model_generator(opt.method, opt.pretrained_model_path)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    if torch.cuda.is_available():
        model.cuda()
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    test_list = os.listdir(opt.image_folder)
    test_list.sort()

    test_data = TestDataset_MOS(data_path=opt.image_folder, data_list=test_list, start_dir=opt.start_dir, image_size=opt.image_size, arg=False)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    mask_test = mask.repeat(opt.batch_size, 1, 1, 1)
    model.eval()


    for i, (MOS, mos_name) in enumerate(test_loader):
        
        MOS = MOS.cuda()
        with torch.no_grad():
            outputs = model(MOS, mask_test)
            for k in range(len(mos_name)):
                output_hsi = outputs[k, :, :, :].squeeze()
                output_hsi = torch.maximum(output_hsi, torch.tensor(0))
                output_hsi = output_hsi / output_hsi.max()
                output_hsi = output_hsi.cpu().numpy()
                input_mos =  MOS[k, :, :, :].squeeze()
                input_mos = input_mos.cpu().numpy()
                print('input_mos>>>>>>>>>>', mos_name[k], input_mos.shape, input_mos.max(), input_mos.mean(), input_mos.min())
                print('outputs>>>>>>>>>>', mos_name[k], output_hsi.shape, output_hsi.max(), output_hsi.mean(), output_hsi.min())

                
                f = h5py.File(opt.save_folder + 'HSI_R_' + mos_name[k][:-4] + '.h5', 'w')
                f['mos'] = input_mos
                f['hsi_R'] = output_hsi
                f.close()

if __name__ == '__main__':
    main()


