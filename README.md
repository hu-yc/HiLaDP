# A PyTorch Implementation of HiLaDP

```bibtex
@inproceedings{hu2025hierarchy,
  title = {Hierarchy-Aware Generative Modeling for Enhanced Label Differential Privacy},
  booktitle = {Proceedings of the 25th {IEEE} International Conference of Data Mining},
  author = {Hu, Yichen and Wan, Changsheng},
  year = 2025,
  pages = {1283--1292},
  address = {Washington, D.C.}
}

```

## Requirements
the file environment.yml records the requirement packages for this project.

To install the requirement packages, please run the following command:

```bash
conda env create -f environment.yml -n hiladp
```

Then, the environment can be activated by using the command
```bash
conda activate hiladp
```


## Datasets
The datasets used in this project could be accessed with `torchvision.datasets` or the following link:
- [Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
- [Eye Disease Diagnosis](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data)

## Demo
Firstly, you need to generate the dependence-aware prior by running the following command:
```bash
python prior_generator.py --ds CIFAR10 --nr_fea 512 --nr_class 10 --epochs 50 --lr 1e-3 --eta 10 --trunc 20
```
Then, you can run the demo by executing the following command:
```bash
python demo.py --ds CIFAR10 --nr_class 10 --epochs 50 --lr 2e-4 --eps 4.0
```

**N.B.**: This package is only free for academic usage.
