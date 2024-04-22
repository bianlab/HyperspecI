# A low-cost integrated hyperspectral imaging sensor with full temporal and spatial resolution at VIS-NIR wide range

[Liheng Bian*](https://scholar.google.com/citations?user=66IFMDEAAAAJ&hl=zh-CN&oi=sra), [Zhen Wang*](https://scholar.google.com/citations?hl=zh-CN&user=DexiDloAAAAJ), [Yuzhe Zhang*](https://scholar.google.com/citations?hl=zh-CN&user=rymYR-wAAAAJ), Lianjie Li, Yinuo Zhang, Chen Yang, Wen Fang, Jiajun Zhao, Chunli Zhu, Qinghao Meng, Xuan Peng, and Jun Zhang. (*Equal contributions)



## 1. System requirements

### 1.1 All software dependencies and operating systems

The project has been tested on Windows 10 or Ubuntu 20.04.1.

### 1.2 Versions the software has been tested on

The project has been tested on CUDA 11.4, pytorch 1.11.0, torchvision 0.12.0,  python 3.7.13, opencv-python 4.5.5.64. 



## 2. Installation guide

### 2.1 Instructions

- The code for training and testing can be downloaded at public repository ：https://github.com/bianlab/HyperspecI
- The mask, testing measurements and pre-trained weights can be downloaded from the Google Drive link: https://drive.google.com/drive/folders/1x6nZpcTP9RIsENJL566pV9v83e1e4gpn?usp=sharing
- Due to the massive amount of training dataset, we have packaged it into multiple repositories for storage: https://github.com/bianlab/Hyperspectral-imaging-dataset



## 3. Program description and testing

Download the mask to  `./MASK/HyperspecI_V1.mat` and  `./MASK/HyperspecI_V2.mat` ;

Download the pre-trained weights to  `./model_zoo/SRNet_V1.pth` and   `./model_zoo/SRNet_V2.pth` ;

Download the testing measurements to    `./Measurements_Test/HyperspecI_V1/` and   `./Measurements_Test/HyperspecI_V2/` 

Download the training dataset to    `'./Dataset_Train/HSI_400_1000/HSI_all/'` and   `'./Dataset_Train/HSI_400_1700/HSI_all/'` 

### 3.1 Main program and data description

- The model of hyperspectral images reconstruction:  `./architecture/SRNet.py` 

- Pre-trained weights of SRNet for HyperspecI-V1:   `./model_zoo/SRNet_V1.pth` 

- Pre-trained weights of SRNet for HyperspecI-V2:   `./model_zoo/SRNet_V2.pth` 

- Calibrated sensing matrix of HyperspecI-V1:   `./MASK/HyperspecI_V1.mat` 

- Calibrated sensing matrix of HyperspecI-V2:   `./MASK/HyperspecI_V2.mat` 

- Measurements collected by our HyperspecI-V1:   `./Measurements_Test/HyperspecI_V1/` 

- Measurements collected by our HyperspecI-V2:   `./Measurements_Test/HyperspecI_V2/` 

- The test and training program :    `train_HyperspecI_V1.py` ,`train_HyperspecI_V2.py`   `test_HyperspecI_V1.py` ,`test_HyperspecI_V2.py` 

  

### 3.2 Model Training of SRNet

Run the train program on the collected measurements to reconstruct hyperspectral images in pytorch platform.

● First, download the training dataset of HyperspecI-V1 (400-1000 nm ) into ` ./Dataset_Train/HSI_400_1000/HSI_all/` , and the training dataset of HyperspecI-V2 (400-1700 nm ) into ` ./Dataset_Train/HSI_400_1700/HSI_all/` . 

● Second, run `SplitDataset.py` to partition the training data and validate, with 90% allocated for training and 10% for validation. 

The details  operations for HyperspecI-V1 dataset partition :

```python
python SplitDataset.py --data_folder './Dataset_Train/HSI_400_1000/HSI_all/' --train_folder './Dataset_Train/HSI_400_1000/Train/' --test_folder './Dataset_Train/HSI_400_1000/Valid/' 
```

The details  operations for HyperspecI-V2 dataset partition :

```python
python SplitDataset.py --data_folder './Dataset_Train/HSI_400_1700/HSI_all/' --train_folder './Dataset_Train/HSI_400_1700/Train/' --test_folder './Dataset_Train/HSI_400_1700/Valid/' 
```



● Third, the training programs are executed to train the spectral reconstruction model. 

For training HyperspecI-V1,  execute the following command in the terminal, and the training results will be saved in the ` ./exp/HyperspecI_V1/` folder.

```python
python train_HyperspecI_V1.py 
```

For training HyperspecI-V2,  execute the following command in the terminal, and the training results will be saved in the ` ./exp/HyperspecI_V2/` folder.

```python
python train_HyperspecI_V2.py 
```



### 3.3 Test hyperspectral reconstruction results in real-world scenes

Run the test program on the collected images to reconstruct hyperspectral images in pytorch platform.

(1) When the images were collected using our HyperspecI-V1 imaging sensors,  the hypersepectral images can be reconstructed by run the following program in the terminal.

```python
python test_HyperspecI_V1.py
```

The measurements collected using HyperspecI-V1 from the folder  `'./Measurements_Test/HyperspecI_V1/' `  . And output reconstructed hyperspectral  images  will be saved in  `'./Measurements_Test/Output_HyperspecI_V1/' `  .



(2) When the images were collected using our HyperspecI-V2 imaging sensors, the hypersepectral images can be reconstructed by run the following program in the terminal. 

```python
python test_HyperspecI_V2.py 
```

The measurements collected using HyperspecI-V2 from the folder  `'./Measurements_Test/HyperspecI_V2/' `  . And output reconstructed hyperspectral  images  will be saved in  `'./Measurements_Test/Output_HyperspecI_V2/' `  .
