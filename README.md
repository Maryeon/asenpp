# Fine-Grained Fashion Similarity Prediction by Attribute-Specific Embedding Learning (ASEN++)

This repository is a [PyTorch](https://pytorch.org/) implementation for paper Fine-Grained Fashion Similarity Prediction by Attribute-Specific Embedding Learning. This work extends our [previous work](https://github.com/maryeon/asen) and obtains much better performance.

### Network Structure

![network structure](imgs/framework.png)

### Dependencies

We conduct our experiments with python 3.6 and CUDA 10.1. Install dependent packages by

```sh
pip install -r requirements.txt
```

### Dataset

#### Data Split

To perform attribute-specific fashion retrieval, [these files](https://drive.google.com/file/d/1_Cyo-IkHYU977bneTXaMC_f63e3vLfSA/view?usp=sharing) are needed. It contains split annotations and meta data for three datasets, i.e., FashionAI, DARN, DeepFashion. Related files for each is included in a directory.

#### FashionAI Dataset

As the full FashionAI has not been publicly released, we utilize its early version for the [FashionAI Global Challenge 2018](https://tianchi.aliyun.com/competition/entrance/231671/introduction?spm=5176.12281949.1003.9.493e3eafCXLQGm). You can first sign up and download the data. Once done, you should uncompress them into the corresponding directory.

#### DARN Dataset

As some images’ URLs have been broken, only 214,619 images are obtained for our experiments. We provide with a series of [URLs](https://drive.google.com/file/d/10jpHsFI2njzEGl7kdACXbvstz6tXyE0R/view?usp=sharing) for the images.

#### DeepFashion Dataset

[DeepFashion](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf) is a large dataset which consists of four benchmarks for various tasks in the field of clothing including [category and attribute prediction](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) which we use for our experiments, in-shop clothes retrieval, fashion landmark detection and consumer-to-shop clothes retrieval.

#### Configuration

The behavior of our codes is controlled by configuration files under the `config` directory. Be sure to correctly configure `root path` and `dataset` according to your working environment.

### Training

```python
python main.py --cfg config/<Dataset>/<Dataset>.yaml config/<Dataset>/s1.yaml
python main.py --cfg config/<Dataset>/<Dataset>.yaml config/<Dataset>/s2.yaml --resume runs/<Dataset>_s1/checkpoint.pth.tar
```

### Evaluation

```python
python main.py --cfg config/<Dataset>/<Dataset>.yaml config/<Dataset>/s2.yaml --resume runs/<Dataset>_s2/model_best.pth.tar --test TEST
```

### Citation

If it's of any help to your research, consider citing our work:

```latex
@inproceedings{dong2021fine,
  title={Fine-Grained Fashion Similarity Prediction by Attribute-Specific Embedding Learning},
  author={Dong, Jianfeng and Ma, Zhe and Mao, Xiaofeng and Yang, Xun and He, Yuan and Hong, Richang and Ji, Shouling},
  year = {2021}
}
```

