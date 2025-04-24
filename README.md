# VSGNet

This repository is the implementation of [_Vision Foundation Model Guided Multi-modal Fusion Network for Remote Sensing Semantic Segmentation_](https://ieeexplore.ieee.org/document/10909146) .

# Introduction

we propose a novel multi-modal fusion network, VSGNet, that integrates Vision Foundation Models (VFM) into remote sensing semantic segmentation tasks. Unlike existing methods that rely on multiple sensor-specific architectures, our approach utilizes a single VFM-based network that can handle diverse types of remote sensing data. This framework significantly reduces the cost and complexity of remote sensing data acquisition by replacing the need for physical sensors with pre-trained large models, which can generate complementary modality images directly from RGB data. The key innovation lies in the application of VFMs to multi-modal remote sensing, a concept not yet explored in prior work.
![overview](framework/fig.png)

# Forests dataset
Sample images from the Forests dataset used in our paper
![dataset](dataset/dataset.png)

# CheckPoints
Model checkpoints are waiting to be uploaded.

# Acknowledgment
Our implementation is mainly based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation), [DFormer](https://github.com/VCIP-RGBD/DFormer?tab=readme-ov-file) and [SAM](https://github.com/facebookresearch/segment-anything). Thanks for their authors.

# Credits
if you find our work useful, you can cite our work:
```
@ARTICLE{10909146,
  author={Pan, Chen and Fan, Xijian and Tjahjadi, Tardi and Guan, Haiyan and Fu, Liyong and Ye, Qiaolin and Wang, Ruili},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Vision Foundation Model Guided Multimodal Fusion Network for Remote Sensing Semantic Segmentation}, 
  year={2025},
  volume={18},
  number={},
  pages={9409-9431},
  keywords={Semantic segmentation;Sensors;Land surface;Transformers;Vegetation;Remote sensing;Visualization;Semantics;Earth;Computational modeling;Cross-modal fusion;land cover mapping;semantic segmentation;vision foundation model (VFM)},
  doi={10.1109/JSTARS.2025.3547880}}
```