## Elucidating Explicit&Easy Shortcut Model
Code base of "On the Design of One-step Diffusion via Shortcutting Flow Paths"

### DATA PREPARATION
This implementation utilizes LMDB datasets with VAE-encoded latent representations for efficient training. The preprocessing pipeline is reimplementation from the [MAR](https://github.com/LTH14/mar/blob/main/main_cache.py). 
Once the ImageNet is downloaded in "YOUR/IMAGNET/PATH", 
run the following for create the LMDB datasets:
```bash
torchrun preprocess_scripts/main_cache_imagenet.py \
--folder_dir "YOUR/IMAGNET/PATH/train"
--target_lmdb "YOUR/DESTINATION/LMDB/PATH"
```


### TRAINING FROM SCRATCH
Training ESC from scratch with SiT-B/2 with class-consistent mini-batching, run the following
```bash
accelerate launch --multi_gpu \
    train.py \
    --exp-name "esc-b2-cc" \
    --output-dir "exp" \
    --data-dir "YOUR/DESTINATION/LMDB/PATH" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 512 \
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
    --cfg-min-t 0.0 \
    --cfg-max-t 1.0 \
    --variational-adaptive-weight \
    --grad-warmup-steps 0 \
    --use-vplug \
    --vplug-prob 0.5 \
    --term-zero-steps 20000 \
    --class-consist \
    --no-debug
```

Or without class-consistent mini-batching:
```bash
accelerate launch --multi_gpu \
    train.py \
    --exp-name "esc-b2-nocc" \
    --output-dir "exp" \
    --data-dir "YOUR/DESTINATION/LMDB/PATH" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 512 \
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
    --cfg-min-t 0.0 \
    --cfg-max-t 1.0 \
    --variational-adaptive-weight \
    --grad-warmup-steps 0 \
    --use-vplug \
    --vplug-prob 0.5 \
    --term-zero-steps 20000 \
    --no-class-consist \
    --no-debug
```

Training ESC from scratch with SiT-XL/2 with class-consistent mini-batching, run the following
```bash
accelerate launch --multi_gpu \
    train.py \
    --exp-name "esc-xl-cc" \
    --output-dir "exp" \
    --data-dir "YOUR/DESTINATION/LMDB/PATH" \
    --model "SiT-XL/2" \
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
    --cfg-omega 0.2 \
    --cfg-kappa 0.92 \
    --cfg-min-t 0.0 \
    --cfg-max-t 0.75 \
    --variational-adaptive-weight \
    --grad-warmup-steps 0 \
    --use-vplug \
    --vplug-prob 0.2 \
    --term-zero-steps 20000 \
    --class-consist \
    --no-debug
```

Or without class-consistent mini-batching:
```bash
accelerate launch --multi_gpu \
    train.py \
    --exp-name "esc-xl-nocc" \
    --output-dir "exp" \
    --data-dir "YOUR/DESTINATION/LMDB/PATH" \
    --model "SiT-XL/2" \
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
    --cfg-omega 0.2 \
    --cfg-kappa 0.92 \
    --cfg-min-t 0.0 \
    --cfg-max-t 0.75 \
    --variational-adaptive-weight \
    --grad-warmup-steps 0 \
    --use-vplug \
    --vplug-prob 0.2 \
    --term-zero-steps 20000 \
    --no-class-consist \
    --no-debug
```


### EVALUATION
For large-scale sampling and quantitative evaluation (FID, IS), we provide a distributed evaluation framework:

```bash
torchrun --nproc_per_node=8 --nnodes=1 evaluate.py \
    --ckpt "/PATH/TO/THE/CHECKPOINTS" \
    --model "SiT-B/2" \
    --resolution 256 \
    --cfg-scale 1.0 \
    --per-proc-batch-size 128 \
    --num-fid-samples 50000 \
    --sample-dir "./fid_dir" \
    --compute-metrics \
    --num-steps 1 \
    --fid-statistics-file "./fid_stats/adm_in256_stats.npz" \
    --adapt-model
```

If there is any data type problem, it means that the numpy or torch version is not correct, you can run the following instead:
```bash
torchrun --nnodes=1 evaluate.py \
    --ckpt "/PATH/TO/THE/CHECKPOINTS" \
    --model "SiT-B/2" \
    --resolution 256 \
    --cfg-scale 1.0 \
    --per-proc-batch-size 128 \
    --num-fid-samples 50000 \
    --sample-dir "./fid_dir" \
    --compute-metrics \
    --num-steps 1 \
    --fid-statistics-file "./fid_stats/adm_float32_in256_stats.npz" \
    --adapt-model
```
