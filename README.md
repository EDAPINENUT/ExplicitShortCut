## On the Design of One-step Diffusion via Shortcutting Flow Paths  *(ESC: ExplicitShortCut)*

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](./demon/elucidating_shortcut__arxiv_.pdf)
[![Model](https://img.shields.io/badge/🤗%20Model-ESC--XL/2-blue)](https://huggingface.co/Delcher/ESC-XL2/tree/main)
[![Model](https://img.shields.io/badge/🤗%20Model-ESC--B/2-blue)](https://huggingface.co/Delcher/ESC-B2)
<div align="center">
  <a href="https://https://edapinenut.github.io/" target="_blank">Haitao&nbsp;Lin</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://peiyannn.github.io" target="_blank">Peiyan&nbsp;Hu</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://openreview.net/profile?id=~Minsi_Ren1" target="_blank">Minsi&nbsp;Ren</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://cn.linkedin.com/in/zhifeng-gao-30070088" target="_blank">Zhifeng&nbsp;Gao</a><sup>3</sup>
  <br>
  <a href="http://homepage.amss.ac.cn/research/homePage/8eb59241e2e74d828fb84eec0efadba5/myHomePage.html" target="_blank">Zhi-Ming&nbsp;Ma</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
   <a href="https://guolinke.github.io" target="_blank">Guolin&nbsp;Ke</a><sup>3</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://tailin.org" target="_blank">Tailin&nbsp;Wu</a><sup>1</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://en.westlake.edu.cn/faculty/stan-zq-li.html" target="_blank">Stan Z.&nbsp;Li</a><sup>1</sup><br>
  <sup>1</sup> Westlake University &emsp; <sup>2</sup> Chinese Academy of Sciences &emsp; <sup>3</sup>DP Technology &emsp
</div>

---
<p align="center">
  <img src="./demon/shortcut_demon.jpg" alt="ESC Overview" width="80%">
</p>

<b>Summary</b>: We propose Explicit ShortCut (ESC), a framework provides
theoretical justification for validity of shortcut models and disentangles concrete component-level choices, thereby enabling systematic identification of improvements.
With our proposed improvements, the resulting one-step model achieves a new state-of-the-art FID50k of 2.85 on ImageNet-256×256 under the classifier-free guidance setting with no pre-training, distillation, or curriculum learning.


---



### Data Preparation
This implementation utilizes LMDB datasets with VAE-encoded latent representations for efficient training. The preprocessing pipeline is reimplementation from the [MAR](https://github.com/LTH14/mar/blob/main/main_cache.py). 
Once the ImageNet is downloaded in "YOUR/IMAGNET/PATH", 
run the following for create the LMDB datasets:
```bash
torchrun preprocess_scripts/main_cache_imagenet.py \
--folder_dir "YOUR/IMAGNET/PATH/train"
--target_lmdb "YOUR/DESTINATION/LMDB/PATH"
```


### Training from Scratch
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

### Download from the Checkpoints

We provide pretrained checkpoints for models trained with class-consistent minibatching:

| Model | Checkpoint |
|-------|------------|
| ESC-XL/2 | [Hugging Face](https://huggingface.co/Delcher/ESC-XL2/tree/main) |
| ESC-B/2 | [Hugging Face](https://huggingface.co/Delcher/ESC-B2) |

### Training the Baselines
See [./scripts/run_baseline.sh](./scripts/run_baseline.sh)


### Evaluation
For the trained checkpoints, or the downloaded ones (.pt file), we provide a distributed evaluation scripts for large-scale sampling and quantitative evaluation (FID, IS):

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

### Acknowledgements

This codebase is built upon [REPA](https://github.com/sihyun-yu/REPA). We thank the authors for their excellent work and open-source contribution.

We also thank  the original MeanFlow implementation: [Gsunshine/MeanFlow](https://github.com/Gsunshine/meanflow), [Gsunshine/py-meanflow](https://github.com/Gsunshine/py-meanflow), and [zhuyu-cs/MeanFlow](https://github.com/zhuyu-cs/MeanFlow) for their PyTorch reimplementation, which helped with early code restructuring.

For [IMM](https://github.com/lumalabs/imm), [sCT](https://github.com/xandergos/sCM-mnist), and [CM](https://github.com/openai/consistency_models), we thanks their (re-)implementation for our further remodularizing.

**If you find our work is helpful to your research, please cite the following:**
```
TBD
```