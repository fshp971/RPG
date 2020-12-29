resume_path='./temp/ckpts/ckpt-80000-model.pkl'
save_dir='./temp/eval'
save_name='eval-rad-10'

python eval.py \
    --dataset cifar10 \
    --arch wrn \
    --batch-size 128 \
    --pgd-radius 10 \
    --pgd-steps 8 \
    --pgd-step-size 2.5 \
    --pgd-norm-type l-infty \
    --resume-path $resume_path \
    --save-dir $save_dir \
    --eval-save-name $save_name