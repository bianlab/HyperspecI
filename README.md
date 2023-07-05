# A low-cost integrated hyperspectral imaging sensor with full temporal and spatial resolution at VIS-NIR wide range

[Liheng Bian*](https://scholar.google.com/citations?user=66IFMDEAAAAJ&hl=zh-CN&oi=sra), [Zhen Wang*](https://scholar.google.com/citations?hl=zh-CN&user=DexiDloAAAAJ), [Yuzhe Zhang*](https://scholar.google.com/citations?hl=zh-CN&user=rymYR-wAAAAJ), Yinuo Zhang, Chen Yang, Lianjie Li, Wen Fang, Jiajun Zhao, Chunli Zhu, Dezhi Zheng, and Jun Zhang. (*Equal contributions)[[pdf]]([arxiv.org/pdf/2306.11583.pdf](https://arxiv.org/pdf/2306.11583.pdf))  

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2306.11583)



## 1. System requirements

### 1.1 All software dependencies and operating systems

The project has been tested on Windows 10 or Ubuntu 20.04.1.

### 1.2 Versions the software has been tested on

The project has been tested on CUDA 11.4, pytorch 1.11.0, torchvision 0.12.0,  python 3.7.13, opencv-python 4.5.5.64. 



## 2. Installation guide

### 2.1 Instructions

To install the software, clone the repository and run the following command in the terminal:

```
git clone https://github.com/bianlab/HyperspecI.git
```

### 2.2 Typical install time on a "normal" desk top computer 

The installation time is approximately 10 seconds and fluctuates depending on network conditions.



## 3. Program description and testing

### 3.1 Program description

- The model of hyperspectral images reconstruction:  `. /architecture/SRNet.py` 
- Pre-trained weights of SRNet under sunlight:   `. /model_zoo/SRNet_sun.pth` 
- Pre-trained weights of SRNet under laboratory light source (Thorlabs, SLS302) :   `. /model_zoo/SRNet_sun.pth` 
- Images collected by our sensors under sunlight:   `. /Test_Source_Sun/Image/` 
- Images collected by our sensors under laboratory light source:   `. /Test_Source_Lib/Image/` 
- The test program :   `test.py` 

### 3.2 Test hyperspectral reconstruction results in real-world scenes

Run the test program on the collected images to reconstruct hyperspectral images in pytorch platform.

(1) When the images were collected using our sensor under sunlight, hypersepectral images can be reconstructed by run the following program in the terminal.

```python
cd ./HyperspecI
python test.py --pretrained_model_path './model_zoo/SRNet_sun.pth' --image_folder './Test_Source_Sun/Image/' --save_folder './Test_Source_Sun/Output_HSI/'
```

(2) When the images were collected using our sensor under laboratory light source, the hypersepectral images can be reconstructed by run the following program in the terminal.

```python
cd ./HyperspecI
python test.py --pretrained_model_path './model_zoo/SRNet_lib.pth' --image_folder './Test_Source_Lib/Image/' --save_folder './Test_Source_Lib/Output_HSI/'
```

The reconstructed hyperspectral images can be saved in the following folder respectively.

```markdown
./Test_Source_Sun/Output_HSI/
./Test_Source_Lib/Output_HSI/
```



