## Training ESC from Scratch
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
