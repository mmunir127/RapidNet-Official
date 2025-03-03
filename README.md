## RapidNet: Multi-Level Dilated Convolution Based Mobile Backbone (WACV 2025)

Mustafa Munir, Md Mostafijur Rahman, and Radu Marculescu

[PDF](https://openaccess.thecvf.com/content/WACV2025/html/Munir_RapidNet_Multi-Level_Dilated_Convolution_Based_Mobile_Backbone_WACV_2025_paper.html) | [Arxiv](https://arxiv.org/abs/2412.10995)

# Overview
This repository contains the source code for RapidNet: Multi-Level Dilated Convolution Based Mobile Backbone


# Pretrained Models

Weights trained on ImageNet-1K, COCO 2017 Object Detection and Instance Segmentation, and ADE20K Semantic Segmentation can be downloaded [here](https://huggingface.co/SLDGroup/RapidNet/tree/main). 

# Usage

## Installation Image Classification

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install mpi4py
pip install -r requirements.txt
```

## Image Classification

### Train image classification:
```
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --use_env main.py --data-path /path/to/imagenet --model RapidNet_Model --output_dir RapidNet_Results
```
For example:
```
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env main.py --data-path Datasets/ILSVRC/Data/CLS-LOC/ --model RapidNet_M --output_dir RapidNet_Results
```
### Test image classification:
```
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --use_env main.py --data-path /path/to/imagenet --model RapidNet_Model --resume pretrained_model --eval
```
For example:
```
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env main.py --data-path Datasets/ILSVRC/Data/CLS-LOC/ --model RapidNet_M --resume pretrained_model --eval
```

## Installation Object Detection and Instance Segmentation
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install timm submitit mmcv-full mmdet==2.28
pip install -U openmim
```

## Object Detection and Instance Segmentation

Detection and instance segmentation on MS COCO 2017 is implemented based on [MMDetection](https://github.com/open-mmlab/mmdetection). We follow settings and hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/segmentation), [PoolFormer](https://github.com/sail-sg/poolformer), and [EfficientFormer](https://github.com/snap-research/EfficientFormer) for comparison. 

All commands for object detection and instance segmentation should be run from the /detection directory.

### Data preparation

Prepare COCO 2017 dataset according to the [instructions in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).

### ImageNet Pretraining
Put ImageNet-1K pretrained weights of backbone as 
```
RapidNet
├── Results
│   ├── model
│   │   ├── model.pth
│   │   ├── ...
```

### Train object detection and instance segmentation:
```
python -m torch.distributed.launch --nproc_per_node num_GPUs --nnodes=num_nodes --node_rank 0 main.py configs/mask_rcnn_rapidnet_model --rapidnet_model RapidNet_Model --work-dir Output_Directory --launcher pytorch > Output_Directory/log_file.txt 
```
For example:
```
python -m torch.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 main.py configs/mask_rcnn_rapidnet_m_fpn_1x_coco.py --rapidnet_model rapidnet_m --work-dir detection_results/ --launcher pytorch > detection_results/RapidNet_M_Detection.txt 
```
### Test object detection and instance segmentation:
```
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --node_rank 0 test.py configs/mask_rcnn_rapidnet_model --checkpoint Pretrained_Model --eval {bbox or segm} --work-dir Output_Directory --launcher pytorch > log_file.txt
```
For example:
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank 0 test.py configs/mask_rcnn_rapidnet_m_fpn_1x_coco.py --checkpoint Pretrained_Model.pth --eval bbox --work-dir detection_results/ --launcher pytorch > detection_results/RapidNet_M_Eval.txt
```

## Installation Semantic Segmentation
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
```
pip install -U openmim
mim install mmengine
mim install mmcv-full
```
```
mim install "mmsegmentation <=0.30.0"
```

### Train semantic segmentation:

Semantic segmentation on ADE20K is implemented based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). We follow settings and hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/segmentation), [PoolFormer](https://github.com/sail-sg/poolformer), and [EfficientFormer](https://github.com/snap-research/EfficientFormer) for comparison. 

```
python -m torch.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 tools/train.py configs/sem_fpn/fpn_rapidnet_m_ade20k_40k.py --rapidnet_model rapidnet_m --work-dir semantic_results/ --launcher pytorch > semantic_results/RapidNet_M_Semantic.txt
```

### Citation

If our code or models help your work, please cite MobileViG (CVPRW 2023) and RapidNet (WACV 2025):

```
@InProceedings{RapidNet_2025_WACV,
    author    = {Munir, Mustafa and Rahman, Md Mostafijur and Marculescu, Radu},
    title     = {RapidNet: Multi-Level Dilated Convolution Based Mobile Backbone},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {8291-8301}
}
```

```
@InProceedings{mobilevig2023,
    author    = {Munir, Mustafa and Avery, William and Marculescu, Radu},
    title     = {MobileViG: Graph-Based Sparse Attention for Mobile Vision Applications},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {2211-2219}
}
```
