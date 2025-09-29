from loss.esc_loss import ESCLoss
from loss.meanflow_loss import MeanFlowLoss
# from loss.scm_loss import SCMLoss
from loss.scd_loss import SCDLoss


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
            variational_adaptive_weight=args.variational_adaptive_weight
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
    
    elif args.model_name == "scm":
        loss_fn = SCMLoss(
            path_type=args.path_type, 
            # Add MeanFlow specific parameters
            time_sampler=args.time_sampler,
            time_mu=args.time_mu,
            time_sigma=args.time_sigma,
            ratio_r_not_equal_t=args.ratio_r_not_equal_t,
            adaptive_p=args.adaptive_p,
            loss_type=args.loss_type
        )
    
    elif args.model_name == "scd":
        loss_fn = SCDLoss(
            path_type=args.path_type, 
            # Add MeanFlow specific parameters
            ratio_r_not_equal_t=args.ratio_r_not_equal_t,
            adaptive_p=args.adaptive_p,
            loss_type=args.loss_type,
            discrete_time_step=args.discrete_time_steps
        )
    return loss_fn