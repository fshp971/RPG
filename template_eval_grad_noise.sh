resume_path='./temp/ckpts/ckpt-80000-model.pkl'
resume_grad_mu_path='./temp/eval-grad/ckpt-80000-mu.pkl'
save_dir='./temp/eval-grad'
save_name='ckpt-80000'

python eval_grad.py \
    --dataset cifar10 \
    --arch wrn \
    --batch-size 128 \
    --pgd-radius 0 \
    --pgd-steps 0 \
    --pgd-step-size 0 \
    --pgd-norm-type l-infty \
    --grad-opt get-noise \
    --grad-rep-T 500 \
    --grad-samp-T 20000 \
    --resume-path $resume_path \
    --resume-grad-mu-path $resume_grad_mu_path \
    --save-dir $save_dir \
    --eval-save-name $save_name
