import argparse

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="MeanFlow Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, default="debug")
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-swanlab", action=argparse.BooleanOptionalAction, default=False)

    # model
    parser.add_argument("--model", type=str, default="SiT-B/4")
    parser.add_argument("--num-classes", type=int, default=1000)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../esc/imagenet_vq/train")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=32)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # model
    parser.add_argument("--model-name", type=str, default="esc", choices=["esc", "meanflow", "scm", "scd", "imm"])

    # optimization
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--checkpointing-steps", type=int, default=40000)
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # basic loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--loss-type", default="l2", type=str, choices=["l2", "adaptive"], help="Loss weighting type")
    
    # MeanFlow specific parameters
    parser.add_argument("--time-sampler", type=str, default="uniform", choices=["uniform", "logit_normal"], 
                        help="Time sampling strategy")
    parser.add_argument("--time-mu", type=float, default=-0.4, help="Mean parameter for logit_normal distribution")
    parser.add_argument("--time-sigma", type=float, default=1.0, help="Std parameter for logit_normal distribution")
    parser.add_argument("--ratio-r-not-equal-t", type=float, default=0.75, help="Ratio of samples where r≠t")
    parser.add_argument("--adaptive-p", type=float, default=1.0, help="Power param for adaptive weighting")
    parser.add_argument("--cfg-omega", type=float, default=1.0, help="CFG omega param, default 1.0 means no CFG")
    parser.add_argument("--cfg-kappa", type=float, default=0.0, help="CFG kappa param for mixing")
    parser.add_argument("--cfg-min-t", type=float, default=0.0, help="Minum time for cfg trigger")
    parser.add_argument("--cfg-max-t", type=float, default=1.0, help="Maxium time for cfg trigger")

    # SCM specific parameters
    parser.add_argument("--variational-adaptive-weight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad-warmup-steps", type=int, default=0, help="Tagent warmup steps")

    # SCD specific parameters
    parser.add_argument("--discrete-time-steps", type=int, default=128, help="Total discretization steps")
    
    # IMM specific parameters
    parser.add_argument("--group-size", type=int, default=4, help="Group size in kernel function")
    parser.add_argument("--gamma", type=int, default=12, help="Gamma as the power in time sampling")

    # ESC specific parameters
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--tgt-decay", type=float, default=0.0, help="Target model decay rate")
    parser.add_argument("--use-vplug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vplug-prob", type=float, default=0.2)
    parser.add_argument("--term-zero-steps", type=int, default=20000, help="Term zero steps")
    parser.add_argument("--class-consist", action=argparse.BooleanOptionalAction, default=False)
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args