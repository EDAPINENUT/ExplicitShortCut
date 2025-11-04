# run shortcut diffusion with classifier-free guidance (SCD-cfg)
    accelerate launch --multi_gpu train.py \
    --exp-name "shortcut-b2-cfg" \
    --output-dir "exp" \
    --data-dir "../imagenet_vq/train" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 256 \
    --checkpointing-step 40000 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240 \
    --path-type "linear" \
    --loss-type "l2" \
    --time-sampler "uniform" \
    --ratio-r-not-equal-t 0.25 \
    --cfg-omega 1.5 \
    --no-debug \
    --tgt-decay 0.0 \
    --ema-decay 0.9999 \
    --discrete-time-steps 128

# run shortcut diffusion with class condition (SCD-cnd)
    accelerate launch --multi_gpu train.py \
    --exp-name "shortcut-b2-cnd" \
    --output-dir "exp" \
    --data-dir "../imagenet_vq/train" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 256 \
    --checkpointing-step 40000 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240 \
    --path-type "linear" \
    --loss-type "l2" \
    --time-sampler "uniform" \
    --ratio-r-not-equal-t 0.25 \
    --cfg-omega 1.0 \
    --no-debug \
    --tgt-decay 0.0 \
    --ema-decay 0.9999 \
    --discrete-time-steps 128

# run meanflow with classifier-free guidance (MF-cfg)
    accelerate launch --multi_gpu train.py \
    --exp-name "meanflow-b2-cfg" \
    --output-dir "exp" \
    --data-dir "../imagenet_vq/train" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240\
    --path-type "linear" \
    --loss-type "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 1.0 \
    --cfg-kappa 0.5 \
    --cfg-min-t 0.0 \
    --cfg-max-t 1.0 \
    --no-debug

# run meanflow with class condition (MF-cnd) by setting cfg-min-t larger than cfg-max-t
    accelerate launch --multi_gpu train.py \
    --exp-name "meanflow-b2-cnd" \
    --output-dir "exp" \
    --data-dir "../imagenet_vq/train" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240 \
    --path-type "linear" \
    --loss-type "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 1.0 \
    --cfg-kappa 0.5 \
    --cfg-min-t 1.0 \
    --cfg-max-t 0.0 \
    --no-debug

# run scm-linear with classifier-free guidance (SCM-linear-cfg)
    accelerate launch --multi_gpu train.py \
    --exp-name "scmlinear-b2-cfg" \
    --output-dir "exp" \
    --data-dir "../imagenet_vq/train" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240 \
    --path-type "linear" \
    --loss-type "l2" \
    --time-sampler "logit_normal" \
    --time-mu -1.0 \
    --time-sigma 1.4 \
    --ratio-r-not-equal-t 0.25 \
    --cfg-omega 1.0 \
    --cfg-kappa 0.5 \
    --cfg-min-t 1.0 \
    --cfg-max-t 0.0 \
    --variational-adaptive-weight \
    --no-debug 

# run scm-cosine with class condition (SCM-cosine-cnd)
    accelerate launch --multi_gpu train.py \
    --exp-name "scmcosine-b2-cnd" \
    --output-dir "exp" \
    --data-dir "../imagenet_vq/train" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240 \
    --path-type "cosine" \
    --loss-type "l2" \
    --time-sampler "logit_normal" \
    --time-mu -1.0 \
    --time-sigma 1.4 \
    --variational-adaptive-weight \
    --no-debug 

# run imm with class condition (IMM-cnd)
    accelerate launch --multi_gpu train.py \
    --exp-name "imm-b2-cnd" \
    --output-dir "exp" \
    --data-dir "../imagenet_vq/train" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 2048 \
    --allow-tf32 \
    --mixed-precision "fp16" \
    --epochs 240 \
    --path-type "linear" \
    --time-sampler "logit_normal" \
    --group-size 4 \
    --gamma 12 \
    --no-debug