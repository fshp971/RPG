save_dir='./temp'

python train.py \
    --dataset cifar10 \
    --arch wrn \
    --optim sgd \
    --weight-decay 2e-4 \
    --momentum 0.9 \
    --batch-size 128 \
    --train-steps 80000 \
    --calc-mg-freq 100 \
    --fix-random-seed \
    --np-seed 19260817 \
    --torch-seed 998244353 \
    --lr 0.1 \
    --lr-decay-rate 0.1 \
    --lr-decay-freq 30000 \
    --pgd-radius 6 \
    --pgd-steps 8 \
    --pgd-step-size 1.5 \
    --pgd-random-start \
    --pgd-norm-type l-infty \
    --report-freq 100 \
    --save-freq 5000 \
    --save-dir $save_dir