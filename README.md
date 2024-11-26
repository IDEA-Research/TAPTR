# TAPTRv2: Attention-based Position Update Improves Tracking Any Point

By [Hongyang Li](https://scholar.google.com.hk/citations?view_op=list_works&hl=zh-CN&user=zdgHNmkAAAAJ&gmla=AMpAcmTJNHoetv6zgfzZkIRcYsFr0UkGGDyl5tAp5etuBqhz3lzYZCQrVDot02xVQ1XTbnMS1fPdAfe0-2--aTXOtewokjyShNLOQQyyhtkolwaz0hvENZpi-pJ-Wg), [Hao Zhang](https://scholar.google.com/citations?user=B8hPxMQAAAAJ&hl=zh-CN), [Shilong Liu](https://scholar.google.com/citations?hl=zh-CN&user=nkSVY3MAAAAJ), [Zhaoyang Zeng](https://scholar.google.com.hk/citations?user=U_cvvUwAAAAJ&hl=zh-CN&oi=sra), [Feng Li](https://scholar.google.com.hk/citations?user=ybRe9GcAAAAJ&hl=zh-CN&oi=sra), [Tianhe Ren](https://scholar.google.com.hk/citations?user=cW4ILs0AAAAJ&hl=zh-CN&oi=sra), [Bohan Li](https://scholar.google.com.hk/citations?hl=zh-CN&user=V-YdQiAAAAAJ), and [Lei Zhang](https://scholar.google.com/citations?hl=zh-CN&user=fIlGZToAAAAJ) <sup>:email:</sup>.

### [Project Page](https://taptr.github.io) | [TAPTR](https://arxiv.org/abs/2403.13042) | [TAPTRv2](https://arxiv.org/abs/2403.13042) | [BibTeX](#citing-taptr)

# :scroll: Abstract
In this paper, we present TAPTRv2, a Transformer-based approach built upon TAPTR for solving the Tracking Any Point (TAP) task. TAPTR borrows designs from DEtection TRansformer (DETR) and formulates each tracking point as a point query, making it possible to leverage well-studied operations in DETR-like algorithms. TAPTRv2 improves TAPTR by addressing a critical issue regarding its reliance on cost-volume, which contaminates the point query’s content feature and negatively impacts both visibility prediction and cost-volume computation. In TAPTRv2, we propose a novel attention-based position update (APU) operation and use key-aware deformable attention to realize. For each query, this operation uses key-aware attention weights to combine their corresponding deformable sampling positions to predict a new query position. This design is based on the observation that local attention is essentially the same as cost-volume, both of which are computed by dot-production between a query and its surrounding features. By introducing this new operation, TAPTRv2 not only removes the extra burden of cost-volume computation, but also leads to a substantial performance improvement. TAPTRv2 surpasses TAPTR and achieves state-of-the-art performance on many challenging datasets, demonstrating the superiority.

___TLDR___

1) According to our experiments and analysis, we find that __TAPTR’s reliance on the source-consuming cost-volume stems from the domain gap between training and evaluation data__.

2) Cost-Volume-Aggregation module in TAPTR not only introduces __extra burden of cost-volume computation__ (especially when the number of points, the length of videos, or the video resolution increases), but also __contaminate__ the point query's content feature. 

3) We find that the __attention weights and cost-volume are essentially the same__. Based on the observation we propose the __Attention-based Position Update (APU)__ module. 

4) TAPTRv2 no longer requires the source-consuming cost-volume, and its point queries are no longer contaminated, making it __simpler yet stronger__.


# :hammer_and_wrench: Method
## Comparison with previous methods
<img src="assets/comparison.png">


## Overview
<img src="assets/overview.png">


# Performance
<img src="assets/performance.png">

# :gear: Usage 
We develop and test our method under ```python=3.8.18,pytorch=1.13.0+cu117,cuda=11.7```. Other versions might be available as well.

## Prepare datasets
Construct the dataset as in [CoTracker](https://github.com/facebookresearch/co-tracker), and put it at 
```
kubric data (for training): ./datas/kubric_movif/
tapvid data (for evaluation): 
    ./datas/tapvid/tapvid_davis
    ./datas/tapvid/tapvid_kinetics
```

## Installation
```sh
git https://github.com/IDEA-Research/TAPTR.git
cd TAPTR
pip install -r requirements.txt
cd models/dino/ops
python setup.py install # This compilation requires nvcc, please make sure you have installed CUDA correctly. CUDA11.7 is tested.
```

## Eval our trained models
Download our provided [checkpoint](https://drive.google.com/file/d/1N639qnzmh9qUQoYckvBilCGpKLwmU5Vu/view?usp=share_link), and put it at "logs/TAPTRv2/taptrv2.pth"
```sh
# Select the dataset you want to evaluate in evaluate.sh manually. 
bash evaluate.sh
```

## Demos
```sh
# Point Trajectory Demo
CUDA_VISIBLE_DEVICES=0 python demo_inter.py -c config/TAPTR.py --path_ckpt logs/TAPTR/taptr.pth
# Video Editing Demo
CUDA_VISIBLE_DEVICES=0 python demo_inter_video_editing.py -c config/TAPTR.py --path_ckpt logs/TAPTR/taptr.pth
```

## Train the models
```sh
bash dist_train.sh
```

# :herb: Acknowledgments
We would like to thank [TAP-Vid](https://github.com/google-deepmind/tapnet) and [Co-Tracker](https://github.com/facebookresearch/co-tracker) for publicly releasing their code and data. 

# :black_nib: Citation

```
@article{li2024taptrv3,
  title={{TAPTRv2: Attention-based Position Update Improves Tracking Any Point}},
  author={Hongyang Li, Hao Zhang, Shilong Liu, Zhaoyang Zeng, Feng Li, Tianhe Ren, Bohan Li, Lei Zhang},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```
