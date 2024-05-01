# Unraveling Complex Aberration Cues with Transformer

## Requirements
* Python == 3.8.18
* pytorch==1.13.1
* torchvision==0.14.1
* tensorboard==2.12.1
* opencv-python==4.2.0.34
* tqdm==4.65.0
* see `requirements.txt` for more detail

## Usage

### 1. Download Datasets
* DDFF-12-Scene Dataset [1]
[ddff-dataset-trainval.h5py](https://vision.in.tum.de/webarchive/hazirbas/ddff12scene/ddff-dataset-trainval.h5)
[ddff-dataset-test.h5py](https://vision.in.tum.de/webarchive/hazirbas/ddff12scene/ddff-dataset-test.h5)

* DefocusNet Dataset [2], 4D Light Field Dataset [3]
  : Please refer to the steps outlined on the official GitHub page of [AiFDepthNet](https://github.com/albert100121/AiFDepthNet) [6].

* FlyingThings3D Dataset [4]
 [FlyingThings3D_FS](https://drive.google.com/file/d/19n3QGhg-IViwt0aqQ4rR8J3sO60PoWgL/view?usp=sharing)

* Middlebury Dataset [5]
[Middlebury_FS](https://drive.google.com/file/d/1FDXf47Qp1-dT_C7bo30ZySvvPAgJf5FU/view?usp=sharing)

* Put the datasets in folders `Datasets`

### 2. Test
* Dataset abbr.
  - ddff (DDFF-12-Scene Dataset)
  - def (DefocusNet Dataset)
  - hci (4D Light Field Dataset)
  - fly (FlyingThings3D Dataset)
###
    python test.py --dataset [Dataset]


## Train
* Dataset abbr.
  - ddff (DDFF-12-Scene Dataset)
  - def (DefocusNet Dataset)
  - hci (4D Light Field Dataset)
  - fly (FlyingThings3D Dataset)
* learning rate: 1e-4 (default)
* training epoch: 3000 (default)
* continue epoch: 0 (default) 
* saved folder: `exp/` (default) 
###
    python train.py --saveroot [saved folder] --dataset [Dataset] --lr [learning rate] --max_epoch [training epochs] --load_epoch [continue epoch]


## References
>[1] Hazirbas, Caner, et al. "Deep depth from focus." Computer Vision–ACCV 2018: 14th Asian Conference on Computer Vision, Perth, Australia, December 2–6, 2018, Revised Selected Papers, Part III 14. Springer International Publishing, 2019.

>[2] Maximov, Maxim, Kevin Galim, and Laura Leal-Taixé. "Focus on defocus: bridging the synthetic to real domain gap for depth estimation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

>[3] Honauer, Katrin, et al. "A dataset and evaluation methodology for depth estimation on 4D light fields." Computer Vision–ACCV 2016: 13th Asian Conference on Computer Vision, Taipei, Taiwan, November 20-24, 2016, Revised Selected Papers, Part III 13. Springer International Publishing, 2017.

>[4] Mayer, Nikolaus, et al. "A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

>[5] Scharstein, Daniel, et al. "High-resolution stereo datasets with subpixel-accurate ground truth." Pattern Recognition: 36th German Conference, GCPR 2014, Münster, Germany, September 2-5, 2014, Proceedings 36. Springer International Publishing, 2014.

>[6] Wang, Ning-Hsu, et al. "Bridging unsupervised and supervised depth from focus via all-in-focus supervision." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

>[7] Won, Changyeon, and Hae-Gon Jeon. "Learning depth from focus in the wild." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

>[8] Liu, Ze, et al. "Video swin transformer." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
