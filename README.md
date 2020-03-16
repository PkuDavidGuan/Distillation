# Distillation

## Overview

Distillation reproduces some state-of-the-art knowledge distillation methods. Knowledge distillation techniques could boost the performance of a miniaturized student network with the supervision of the output distribution and feature maps from a sophisticated teacher network. Generally, Knowledge distillation methods can be divided into two categories: **output distillation** and **feature distillation**. In output distillation methods, only the last output layer of teacher and student networks are leveraged for knowledge distillation. While in the feature distillation methods, several distillation positions are selected.

![Output distillation and feature distillation methods.](overview.png)

## Model zoo and baselines

### Models

- `wrn`: "[Wide Residual Networks](https://arxiv.org/abs/1605.07146)"
- `shufflenetV2`: "[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)"

### Datasets

- `cifar10`: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- `cifar100`: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- `imagenet`:[ILSVRC2012](http://image-net.org/challenges/LSVRC/2012/index)
- `cinic10`: [CINIC-10: CINIC-10 Is Not Imagenet or CIFAR-10](https://github.com/BayesWatch/cinic-10)

### Baselines

- kd: "[Distilling the Knowledge in a Neural Network](http://arxiv.org/abs/1503.02531)"
- at: "[Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](http://arxiv.org/abs/1612.03928)"
- sp: "[Similarity-Preserving Knowledge Distillation](http://openaccess.thecvf.com/content_ICCV_2019/html/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.html)"
- margin_ReLU: "[A Comprehensive Overhaul of Feature Distillation](<http://openaccess.thecvf.com/content_ICCV_2019/html/Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.html>)"

## Quick Start

### Train a simple network

Distillation support the vanilla training of a network. For example, if we want to train a teacher network `wrn_28_4` on dataset `CIFAR-100`, we could run: 

```shell
source scripts/cifar100.teacher.wrn_28_4.sh
```

The script `cifar100.teacher.wrn_28_4.sh` contains: 

```shell
python simple_train.py \
    --teacher_name wrn-28-4 \                  # the wrn_28_4 network 
    --dataset cifar100 \                       # train on CIFAR-100
    --name teacher/cifar100_wrn_28_4_e200 \    # save the trained model here.
    --tensorboard                              # log the training statistics with tensorboard
```



### Train the student network with knowledge distillation methods

For example, if we want to implement the `kd` method with `wrn_28_4` as the teacher network and `wrn_16_2` as the student network, we could run:

```shell
source scripts/cifar100.wrn_28_4_16_2.kd.sh
```

The content of the script  `cifar100.wrn_28_4_16_2.kd.sh` is:

```shell
python3 train.py \
    --dataset cifar100 \ # select the CIFAR-100 as the experimental dataset.
    --name cifar100.wrn_28_4_16_2.kd \ # save the training logs here.
    --teacher_name wrn-28-4 \  # select wrn_28_4 as the teacher
    --student_name wrn-16-2 \  # select wrn_16_2 as the student
    --epochs 200 \             # total training epoches
    --kd_method kd \           # select kd as the knowledge distilation method
    --alpha 0.8 \              # hyperparamters of the method kd
    --temperature 16 \         # hyperparamters of the method kd
    --teacher_model runs/teacher/cifar100_wrn_28_4_e200/model_best.pth.tar \ # the location that saves the teacher network
    --tensorboard              # log the training statistics with tensorboard
```



## Expermental Results

### Results on CIFAR100

| Teacher  | Student  | Teacher acc | Student acc | kd    | at    | sp    | margin_ReLU |
| -------- | -------- | ----------- | ----------- | ----- | ----- | ----- | :---------: |
| WRN-28-4 | WRN-16-2 | 79.17       | 73.42       | 74.02 | 73.1  | 74.12 |    75.41    |
| WRN-28-4 | WRN-16-4 | 79.17       | 77.24       | 78.56 | 77.43 | 78.67 |    79.39    |
| WRN-28-4 | WRN-28-2 | 79.17       | 75.78       | 76.14 | 75.41 | 77    |    77.86    |

### Results on CINIC10

| Teacher      | Student      | Teacher acc | Student acc | kd    | at    | sp    | margin_ReLU |
| ------------ | ------------ | ----------- | ----------- | ----- | ----- | ----- | :---------: |
| Sh.NetV2-2.0 | Sh.NetV2-1.0 | 81.5        | 78.11       | 85.78 | 84.71 | 85.32 |    85.29    |
| Sh.NetV2-2.0 | Sh.NetV2-0.5 | 81.5        | 73.82       | 80.34 | 79.06 | 79.15 |    78.79    |
| Sh.NetV2-1.0 | Sh.NetV2-0.5 | 78.11       | 73.82       | 80.29 | 78.42 | 79.02 |    79.69    |

Note: Sh.NetV2 means shufflenetV2.

### Grid Search for KD

We make a grid search for the method `kd` on CIFAR100 with the teacher `wrn_28_4` and student `wrn_16_2`. Experimental result shows a fluctuation on accuracy with different hyperparamters, which reveals the necessity of a grid search when implementing `kd` into a certain scenario.

|      | 2     | 4     | 8     | 16    |
| ---- | ----- | ----- | ----- | ----- |
| 0.2  | 74.65 | 74.88 | 74.74 | 74.45 |
| 0.4  | 74.2  | 74.32 | 74.79 | 74.81 |
| 0.6  | 73.77 | 74.9  | 74.5  | 74.28 |
| 0.8  | 73.27 | 73.99 | 74.31 | 73.89 |

Note: 1) Row: $\alpha$, the weight of kd loss defined in the original paper. 2) Column: $T$, the temperature defined in the original paper.

## Add your own knowledge distillation methods

### Add a new dataset

If you want to add a new dataset A, you should create a class (subclass of torch.utils.data.dataset) to load A. You should also design the data preprocess. Just mimic `framework/datasets/cifar.py` to create your dataset, and **do not forget** to register it into `framework/datasets/__init__.py`.

### Add a new model

If you want to add a new model, you should put it into `framework/modeling/backbone/` and register it into `framework/modeling/backbone/__init__.py`.

You should modify the `forward` function defined in your new model. The forward should return: 1. the output logits. 2. All feature maps that may be used for knowledge distillation. 

Specifically, if you use the method `margin_ReLU`, you have to consider many other things such as the margin, the pre-ReLU feature maps... See  "[A Comprehensive Overhaul of Feature Distillation](<http://openaccess.thecvf.com/content_ICCV_2019/html/Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.html>)" for details.

### Add a new knowledge distillation method

We really hope our framework can help researchers to build up their novel distillation methods. You should do the following two steps:

First, you should select the appropriate model builder:

1. If you only need the output layer of the teacher and student network, you can just use the `NaiveModelBuilder` class defined in `framework/modeling/model_builder.py`.
2. If you need feature maps in different stages of network, you can use the `ModelBuilder` class defined in `framework/modeling/model_builder.py`.
3. Otherwise, you may need to create your own model builder.

Second, you should define your loss function, and add it into `framework/losses` . Do not forget to register the new loss function into `framework/losses/__init__.py`.