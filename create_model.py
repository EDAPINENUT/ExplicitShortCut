from loss.esc_loss import ESCLoss
from loss.meanflow_loss import MeanFlowLoss
from loss.scm_cos_loss import SCMCosineLoss
from loss.scm_linear_loss import SCMLinearLoss
from loss.scd_loss import SCDLoss
from loss.imm_loss import IMMLoss

def create_model(args):
    if args.model_name == "esc":
        loss_fn = ESCLoss(
            path_type=args.path_type, 
            # Add MeanFlow specific parameters
            time_sampler=args.time_sampler,
            time_mu=args.time_mu,
            time_sigma=args.time_sigma,
            ratio_r_not_equal_t=args.ratio_r_not_equal_t,
            adaptive_p=args.adaptive_p,
            loss_type=args.loss_type,
            label_dropout_prob=args.cfg_prob,
            cfg_omega=args.cfg_omega,
            cfg_kappa=args.cfg_kappa,
            cfg_min_t=args.cfg_min_t,
            cfg_max_t=args.cfg_max_t,
            # Add ESC specific parameters
            use_vplug=args.use_vplug,
            vplug_select_prob=args.vplug_prob,
            variational_adaptive_weight=args.variational_adaptive_weight,
            grad_warmup_steps=args.grad_warmup_steps,
        )
    elif args.model_name == "meanflow":
        loss_fn = MeanFlowLoss(
            path_type=args.path_type, 
            # Add MeanFlow specific parameters
            time_sampler=args.time_sampler,
            time_mu=args.time_mu,
            time_sigma=args.time_sigma,
            ratio_r_not_equal_t=args.ratio_r_not_equal_t,
            adaptive_p=args.adaptive_p,
            loss_type=args.loss_type,
            label_dropout_prob=args.cfg_prob,
            cfg_omega=args.cfg_omega,
            cfg_kappa=args.cfg_kappa,
            cfg_min_t=args.cfg_min_t,
            cfg_max_t=args.cfg_max_t,
        )
    
    elif args.model_name == "scm" and args.path_type == "linear":
        loss_fn = SCMLinearLoss(
            path_type=args.path_type, 
            # Add SCM-linear specific parameters
            time_sampler=args.time_sampler,
            time_mu=args.time_mu,
            time_sigma=args.time_sigma,
            ratio_r_not_equal_t=args.ratio_r_not_equal_t,
            adaptive_p=args.adaptive_p,
            loss_type=args.loss_type,
            label_dropout_prob=args.cfg_prob,
            cfg_omega=args.cfg_omega,
            cfg_kappa=args.cfg_kappa,
            cfg_min_t=args.cfg_min_t,
            cfg_max_t=args.cfg_max_t,
            variational_adaptive_weight=args.variational_adaptive_weight,
            grad_warmup_steps=args.grad_warmup_steps,
        )
    
    elif args.model_name == "scm" and args.path_type == "cosine":
        loss_fn = SCMCosineLoss(
            path_type=args.path_type, 
            # Add SCM-cosine specific parameters
            time_sampler=args.time_sampler,
            time_mu=args.time_mu,
            time_sigma=args.time_sigma,
            loss_type=args.loss_type,
            label_dropout_prob=args.cfg_prob,
            variational_adaptive_weight=args.variational_adaptive_weight,
            adaptive_p=args.adaptive_p,
            grad_warmup_steps=args.grad_warmup_steps,
        )
    
    elif args.model_name == "scd":
        loss_fn = SCDLoss(
            path_type=args.path_type, 
            # Add SCD specific parameters
            ratio_r_not_equal_t=args.ratio_r_not_equal_t,
            adaptive_p=args.adaptive_p,
            loss_type=args.loss_type,
            discrete_time_step=args.discrete_time_steps
        )
    
    elif args.model_name == "imm":
        loss_fn = IMMLoss(
            path_type=args.path_type, 
            # Add IMM specific parameters
            time_sampler=args.time_sampler,
            label_dropout_prob=args.cfg_prob,
            group_size=args.group_size,
            gamma=args.gamma,
        )
    return loss_fn