# Conservative Adaptive Synthetic Sampling(CASS)
When using the code, please cite our paper：
Enhancing Tool Wear State Identification in imbalanced and small sample scenarios through Conservative Adaptive Synthetic Sampling


## Code
This codebase contains example code implementing Conservative Adaptive Synthetic Sampling (CASS).
The first file, CASS.py, contains the CASS implementation code. The second file, SFROR.py, implements the Sample Feature Retention Rate and Offset Rate (SFROR) algorithm. The third file, Data_Preprocessing.py, implements the data preprocessing routine, which comprehensively includes wavelet denoising, feature extraction using EEMD, dimensionality reduction via KPCA, and visualization of the preprocessed data results.

## Dataset
We have publicly released the TTD2022 public turning tool dataset to scholars worldwide, 
and TTD2022 is publicly available for free at:
 http://mad-net.cn:8765.


## Dependencies
The code was written with: Python 3.7.0; Numpy 1.21.2; pandas 1.1.5; Matplotlib 3.4.2;PyEMD；pywt and Scikit learn 0.23.2.

## System components and specification used for the implementation
CPU: 12th Gen Intel(R) Core(TM) i9-12900KF
GPU: NVIDIA GeForce RTX 3090
RAM: 32 GB

##Author Contact
E-mail: gs.ywzhu19@gzu.edu.cn or yw_zhu2022126.com


