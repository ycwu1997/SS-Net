# Exploring Smoothness and Class-Separation for Semi-supervised Medical Image Segmentation
by Yicheng Wu*, Zhonghua Wu, Qianyi Wu, Zongyuan Ge, and Jianfei Cai. 

### News
```
<12.06.2022> We provided our pre-trained models on the LA and ACDC datasets, see './SS-Net/pretrained_pth';
```
```
<09.06.2022> We released the codes;
```
### Introduction
This repository is for our MICCAI 2022 paper: '[Exploring Smoothness and Class-Separation for Semi-supervised Medical Image Segmentation](https://doi.org/10.1007/978-3-031-16443-9_4)'.

### Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.1 and Python 3.8.10. All experiments in our paper were conducted on a single NVIDIA Tesla V100 GPU with an identical experimental setting.

### Usage
1. Clone the repo.;
```
git clone https://github.com/ycwu1997/SS-Net.git
```
2. Put the data in './SS-Net/data';

3. Train the model;
```
cd SS-Net
# e.g., for 5% labels on LA
python ./code/train_ss_3d.py --labelnum 4 --gpu 0
```
4. Test the model;
```
cd MC-Net
# e.g., for 5% labels on LA
python ./code/test_LA.py --labelnum 4
```

### Citation
If our SS-Net model is useful for your research, please consider citing:

      @inproceedings{wu2022exploring,
        title={Exploring Smoothness and Class-Separation for Semi-supervised Medical Image Segmentation},
        author={Wu, Yicheng and Wu, Zhonghua and Wu, Qianyi and Ge, Zongyuan and Cai, Jianfei},
        booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
        pages={34--43},
        volume={13435},
        year={2022},    
        doi={10.1007/978-3-031-16443-9\_4},
        organization={Springer, Cham}
        }

### Acknowledgements:
Our code is origin from [MC-Net](https://github.com/ycwu1997/MC-Net), [SemiSeg-Contrastive](https://github.com/Shathe/SemiSeg-Contrastive), [VAT](https://github.com/lyakaap/VAT-pytorch), and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

### Questions
If any questions, feel free to contact me at 'ycwueli@gmail.com'

