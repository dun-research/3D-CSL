# 3D-CSL: Self-supervised 3D Context Similarity Learning for Near-Duplicate Video Retrieval

This is an official pytorch implementation of paper ["3D-CSL: Self-supervised 3D Context Similarity Learning for Near-Duplicate Video Retrieval"](https://arxiv.org/abs/2211.05352). This repository provides code for calculating similarities between the query and database videos. Also, the evaluation code and pretrained weights are available to facilitate the reproduction of the paper's results.



## Installation
Requirements
 + python 3.6+
 + torch==1.7.1
 + (optional) mmcv==1.5.2 and mmaction2==0.24.0

You can install all the dependencies by 
```
    pip install -r requirements.txt
```

We also provide a distributed script for evaluation, which requires additional installation of mmcv and mmaction.
Please follow the official instruction to install, [mmcv](https://github.com/open-mmlab/mmcv) and [mmaction2](https://github.com/open-mmlab/mmaction2)

## Evaluation
We provide code and pretrained weights to reproduce the experiments in the paper.

### Dataset Preparation

+ **Video Download**

  + In order to download the datasets, you can follow the official guidelines. 
  + Currently supported datasets are: [FIVR-200K](https://github.com/MKLab-ITI/FIVR-200K) and [CC_WEB_VIDEO](http://vireo.cs.cityu.edu.hk/webvideo/).

+ **Annotation File**
  + We use three files to organize the video relationships for each dataset following [Visil](https://github.com/MKLab-ITI/visil).
  + query_file: paths to file that contains the query videos, see example in [fivr-5k-queries.txt](data/fivr-5k-queries.txt)
  + database_file: paths to file that contains the database videos, see example in [fivr-5k-database.txt](data/fivr-5k-database.txt)
  + ann_file: relationship of query and database video, see example in [fivr.pkl](data/fivr.pkl)
  + please modified the default paths in [eval_dataset.py](datasets/eval_dataset.py)
  

### Run Evaluation
+ **Pretrained Weights**
  + You can download the pretrained model on [Google Drive](https://drive.google.com/drive/folders/1Zd01H2dwewE3FzjF46Kc80icHEbcArCI?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1kwfxqmZmwnlY_WagR58KlA)(password: jmnb).
  + After setup data and weights, you can use the following scripts to evaluate fivr-5k dataset:
  + To reproduce the reported results of paper, please switch to "fivr-200k" or "cc_web_video" dataset. The annotation files of these datasets can be found on [Google Drive](https://drive.google.com/drive/folders/1Zd01H2dwewE3FzjF46Kc80icHEbcArCI?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1kwfxqmZmwnlY_WagR58KlA)(password: jmnb).

```
    # run model on single gpu
    python run_eval.py \
        --model base \
        --weights checkpoints/best_model_base_224_16x16_rgb.pth \
        --dataset fivr-5k  \
        --out_file outputs/similarity.json \
        --topk-cs

    # run model on multiple gpus, requires mmcv and mmaction2
    python -m torch.distributed.launch --nproc_per_node 8  \
          run_eval_dist.py \ 
          --model base  \
          --weights checkpoints/best_model_base_224_16x16_rgb.pth   \
          --dataset fivr-5k  \
          --out  outputs/test_3dcsl_base -la pytorch \
          --topk-cs 

```


## Use 3D-CSL to extract video features
3D-CSL is a well-trained video feature extractor that provides computation-effective clip-level features for video retrieval. Below is a toy example that extracts features for any videos.

```python
    import torch

    model = SimilarityRecognizer(model_type="base", batch_size=8)
    model.load_pretrained_weights(checkpoint)
    model.eval()

    imgs = torch.rand((1, 3, 8, 224, 224))
    features = model.extract_features(imgs)
```



## Citation
If this code is helpful for your work, please cite our paper.
```
    @article{deng20223d,
    title={3D-CSL: self-supervised 3D context similarity learning for Near-Duplicate Video Retrieval},
    author={Rui Deng and Qian Wu and Yuke Li},
    journal={arXiv preprint arXiv:2211.05352},
    year={2022}
    }
```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact us for further details
The code for training is not included in this repository. We can not release the training code publicly for IP reasons. 

If you need the training code or have any questions, please contact us with the following email address:
[dengrui01@corp.netease.com]()
