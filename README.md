# Robustness, Privacy, and Generalization of Adversarial Training

This repository contains the PyTorch source code for technical report, "Robustness, Privacy, and Generalization of Adversarial Training," by Fengxiang He, Shaopeng Fu, Bohan Wang, and Dacheng Tao. 

## Code Structures

```
|----./
    |---- models/
        |---- __init__.py
        |---- resnet.py
    |---- utils/
        |---- datasets/
            |---- __init__.py
            |---- torchvision_datasets.py
        |---- __init__.py
        |---- argument.py
        |---- attack.py
        |---- data.py
        |---- generic.py
    |---- eval.py
    |---- eval_grad.py
    |---- train.py
```

## Requirements

- Python 3.7.2
- pytorch 1.6.0
- torchvision 0.7.0
- numpy 1.19.1

## Instructions

The following four scripts are the example scripts for adversarial training, evaluation, and calculating gradient noises:

```
template_train.sh
template_eval.sh
template_eval_grad_mu.sh
template_eval_grad_noise.sh
```

We then introduce the usages of these scripts as examples. Noting that you should modify these scripts for your own use.

### Training

Run the example script as follow:

```
bash template_train.sh
```

After finishing training, the log will be saved in `./temp/log.txt`, the data records will be saved in `./temp/ckpts/ckpt-80000-log.pkl`, and the trained model will be saved in `./temp/ckpts/ckpt-80000-model.pkl`.

Some of the key hyperparameters in `template_train.sh` is introduced in the following table:

|  Hyperparameter   |                         Description                          |                     Type                     |
| :---------------: | :----------------------------------------------------------: | :------------------------------------------: |
|    `--dataset`    |                    The choice of dataset                     | `string`, should be: `cifar10` or `cifar100` |
| `--calc-mg-freq`  | The number of interval steps of calculating max gradient norm for robustified intensity |                    `int`                     |
|  `--pgd-radius`   |                   The radius $\rho$ in PGD                   |                   `float`                    |
|   `--pgd-steps`   |         The number of iterative modification in PGD          |                    `int`                     |
| `--pgd-step-size` |                The step size $\alpha$ in PGD                 |                   `float`                    |
| `--pgd-norm-type` |                  The norm of metric in PGD                   |    `string`, should be: `l-infty` or `l2`    |

### Evaluation

Run the example script as follow:

```
bash template_eval.sh
```

The evaluation results will be saved in `./temp/eval/eval-rad-10-ts-80000.pkl`, which includes the accuracies on training / test / adversarial training / adversarial test sets, and the membership inference attack accuracy.

### Calculation of Gradient Noises

##### Step 1: Calculate the gradient on the full training set

Run the example script as follow:

```
bash template_eval_grad_mu.sh
```

The gradient on the full training set will be saved in `./temp/eval-grad/ckpt-80000-mu.pkl`.

##### Step 2: Calculate the gradient noises

Run the example script as follow:

```
bash template_eval_grad_noise.sh
```

The collected gradient noises will be saved in `./temp/eval-grad/ckpt-80000-noise.pkl`.

## Citation

```
@article{he2020robustness,
  title={Robustness, Privacy, and Generalization of Adversarial Training},
  author={He, Fengxiang and Fu, Shaopeng and Wang, Bohan and Tao, Dacheng},
  journal={arXiv preprint arXiv:2012.13573},
  year={2020}
}
```

## Contact

For any issues please kindly contact

Fengxiang He, [fengxiang.f.he@gmail.com](mailto:fengxiang.f.he@gmail.com)   
Shaopeng Fu, [fshp971@gmail.com](mailto:fshp971@gmail.com)   
Bohan Wang, [bhwangfy@gmail.com](mailto:bhwangfy@gmail.com)   
Dacheng Tao, [dacheng.tao@gmail.com](mailto:dacheng.tao@gmail.com)   

--

Last update: Mon 29 Dec 2020 AEDT
