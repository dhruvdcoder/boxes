#!/bin/env python
from boxes.objective import *
import argparse
import os
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import dummy_minimize, gp_minimize, forest_minimize, gbrt_minimize
import sqlite3
import copy
from pprint import pprint

# np.seterr(all='raise')

##########################################
# Hyperparameter Sampling
##########################################
learning_params = (
    Integer(5, 50, name="dims"),
    Integer(20, 50, name="num_batches"),
    Real(1e-4, 1e-0, "log-uniform", name="learning_rate"),
)

loss_weights = (
    Real(1e-2, 1e4, "log-uniform", name="mean_unit_cube_loss"),
    Real(1e-6, 1e4, "log-uniform", name="mean_unary_kl_loss"),
    Real(1e-1, 1e4, "log-uniform", name="mean_pull_loss"),
    Real(1e3, 1e8, "log-uniform", name="mean_push_loss"),
    Real(1e-1, 1e4, "log-uniform", name="mean_surrogate_loss"),
)


#######################################
# Default Settings
#######################################
defaults = {
    "BoxParamType": None,
    "extra_callbacks": [],
    "gradient_clipping": (-1,1),
    "universe_box": None,
    "unit_cube_loss_func": (mean_unit_cube_loss,),
    "projected_gradient_descent": False,
    "vol_func": clamp_volume,
    "plateau_loss_funcs": [],
}

# Box Parametrizations
box_param_options = {
    "Boxes": {"BoxParamType": Boxes, "extra_callbacks": [MinBoxSize()]},
    "MinMaxBoxes": {"BoxParamType": MinMaxBoxes},
    "DeltaBoxes": {"BoxParamType": DeltaBoxes, "gradient_clipping": (-1000, 1000)},
    "SigmoidBoxes": {"BoxParamType": SigmoidBoxes, "gradient_clipping": (-1000, 1000)},
    "MinMaxSigmoidBoxes": {"BoxParamType": MinMaxSigmoidBoxes, "gradient_clipping": (-1000, 1000)},
}

# Universe
universe_options = {
    "scaled": {"universe_box": smallest_containing_box, "unit_cube_loss_func": (), "gradient_clipping": (-1000, 1000)},
    # Note: the gradient clipping here ^^^ will overwrite any gradient clipping from above.
    # This is OK because:
    #   - DeltaBoxes' gradient_clipping is the same as this (currently)
    #   - It doesn't really make sense to use SigmoidBoxes with a scaled universe
    "scaled_penalty": {"universe_box": smallest_containing_box_outside_unit_cube},
    "hard_penalty": {},
    # "hard_projected": {"projected_gradient_descent": True},
    "hard": {"unit_cube_loss_func": ()},
}

# Methods for fixing plateaus in the loss landscape
plateau_options = {
    "softbox": {"vol_func": soft_volume},
    "constraint": {"plateau_loss_funcs": [mean_pull_loss, mean_push_loss]},
    # "hardbox": {"plateau_loss_funcs": (mean_surrogate_loss)},
    "none": {},
}

# Hyperparameter Optimization Methods
hyperparameter_optimization_methods = {
    "random": dummy_minimize,
    "forest": forest_minimize,
    "gp": gp_minimize,
    "gbrt": gbrt_minimize,
}

########################################
# Parser
########################################
parser = argparse.ArgumentParser()
parser.add_argument("--box_param", choices=box_param_options, help="Type of box parametrization", required=True)
parser.add_argument("--universe", choices=universe_options, help="Volume function", required=True)
parser.add_argument("--plateau", choices=plateau_options, help="Method for pulling boxes together", required=True)
parser.add_argument("--path", help="Folder with training data", required=True)
parser.add_argument("--hyper_opt_method", choices=hyperparameter_optimization_methods, help="Optimization method to choose hyperparameters", default="gp")
parser.add_argument("--epochs", help="Number of epochs to train", type=int, default=10)
parser.add_argument("--hyper_calls", help="Number of times to call the hyperparameter optimizer", type=int, default=100)
parser.add_argument("--init_min_vol", help="Minimum volumes for boxes at initialization", type=float, default=torch.finfo(torch.float32).tiny)
parser.add_argument("--device", help="Specify the device to train on (eg. cpu, cuda)", type=str, default="cuda")
args = parser.parse_args()
os.chdir(args.path)


##########################################
# Update Settings
##########################################
settings = copy.deepcopy(defaults)
settings.update(box_param_options[args.box_param])
settings.update(universe_options[args.universe])
settings.update(plateau_options[args.plateau])
settings["epochs"] = args.epochs
settings["init_min_vol"] = args.init_min_vol
settings["device"] = args.device

# Pick random seed for eventual hyperparameter training
# (This is passed to the skopt function)
random_seed = np.random.randint(2**32 - 1)

###################################################
# SETUP SQL
###################################################
sql_hyperparameter_optimization_instance_columns = [
    "[id] INTEGER PRIMARY KEY",
    "[box_param] TEXT",
    "[universe] TEXT",
    "[plateau] TEXT",
    "[init_min_vol] REAL",
    "[hyper_opt_method] TEXT",
    "[epochs] INTEGER",
    "[hyper_calls] INTEGER",
    "[gradient_clip_min] REAL",
    "[gradient_clip_max] REAL",
    "[device] TEXT",
    "[random_seed] INTEGER",
    ]

sql_training_instance_columns = [
    "[id] INTEGER PRIMARY KEY",
    "[hyper_opt_id] INTEGER NOT NULL",
    "[hyper_opt_iteration] INTEGER",
    "[dims] INTEGER",
    "[num_batches] INTEGER",
    "[learning_rate] REAL",
    "[torch_random_seed] INTEGER",
    ]
sql_training_instance_columns += [f"[{z.name}] REAL" for z in loss_weights]
# Note: FOREIGN KEY definitions have to occur after the rest of the columns.
sql_training_instance_columns += ["FOREIGN KEY(hyper_opt_id) REFERENCES Hyperparameter_Optimization_Instances(id)"]
#####################################################
# Write SQL for Hyperparameter Optimization Instance
#####################################################
hyperparam_optimization_instance_values = {
    **vars(args),
    "gradient_clip_min": settings["gradient_clipping"][0],
    "gradient_clip_max": settings["gradient_clipping"][1],
    "random_seed": random_seed
}
hyperparam_optimization_instance_values.pop("path", None)

sql_conn = sqlite3.connect("train.db")
sql_logging.create_or_update_table_and_cols_(sql_conn, "Hyperparameter_Optimization_Instances", sql_hyperparameter_optimization_instance_columns)
hyper_opt_id = sql_logging.write_dict_(sql_conn, "Hyperparameter_Optimization_Instances", hyperparam_optimization_instance_values)
sql_logging.create_or_update_table_and_cols_(sql_conn, "Training_Instances", sql_training_instance_columns)
##############################
# Load Data
##############################
unary_probs = torch.from_numpy(np.loadtxt("train_tc_unary.tsv")).float().to(args.device)
train = Probs.load_from_julia("", 'train_tc_pos.tsv', 'train_neg.tsv', ratio_neg = 1).to(args.device)
dev = Probs.load_from_julia("", 'dev_pos.tsv', 'dev_neg.tsv', ratio_neg = 1).to(args.device)
# test = PairwiseProbs.load_from_julia(PATH, 'test_pos.tsv', 'test_neg.tsv', ratio_neg = 1).to(args.device)

#############################
# Prepare Objective Function
#############################
loss_funcs = [
    mean_cond_kl_loss,
    mean_unary_kl_loss(unary_probs),
    *settings.pop("unit_cube_loss_func", []),
    *settings.pop("plateau_loss_funcs", []),
]

# Reduce our parameter space, depending on the arguments chosen.
loss_func_names = [f.__name__ for f in loss_funcs]
loss_weights = [z for z in loss_weights if z.name in loss_func_names]
parameter_space = (*learning_params, *loss_weights)


o = Objective(
    loss_funcs = loss_funcs,
    unary_probs = unary_probs,
    train = train,
    dev = dev,
    obj_func_to_min = obj_mean_cond_kl_min,
    hyper_opt_id = hyper_opt_id,
    sql_conn = sql_conn,
    **settings
)

@use_named_args(parameter_space)
def objective(**hyperparameters):
    return o.objective(**hyperparameters)

##############################
# Hyperparameter Optimize!
##############################
try:
    res_gp = hyperparameter_optimization_methods[args.hyper_opt_method](
        objective, parameter_space, n_calls=args.hyper_calls, random_state=random_seed)
    pprint(res_gp.fun)
    pprint(res_gp.specs)
    pprint(res_gp.x)
except KeyboardInterrupt:
    print("Stopped hyperparamter optimzation due to keyboard interrupt.\nThe in-progress run was not saved to the database.")
