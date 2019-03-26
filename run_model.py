#!/bin/env python
from boxes import *
import math
import argparse
import os
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import dummy_minimize, gp_minimize, forest_minimize, gbrt_minimize
import sqlite3


@dataclass
class RecorderCollection:
    learn:Recorder = field(default_factory=Recorder)
    train:Recorder = field(default_factory=Recorder)
    dev:Recorder = field(default_factory=Recorder)

def save_record_to_sql(conn, rec: Recorder, table_name: str, training_instance_id: int):
    cols = [f"[{col}] REAL" for col in rec._data.columns]
    cols += ["[epochs] REAL"]
    cols += ["[training_instance_id] INT"]
    rec._data["training_instance_id"] = training_instance_id

    c = conn.cursor()
    c.execute(f"create table if not exists {table_name} ({','.join(cols)})")
    for col in cols:
        try:
            c.execute(f"alter table {table_name} add column {col}")
        except:
            pass
    rec._data.to_sql(table_name, conn, if_exists="append", index_label="epochs")
    conn.commit()

learning_params = (
    Integer(5, 50, name="dims"),
    Integer(20, 200, name="num_batches"),
    Real(1e-5, 1e-1, "log-uniform", name="learning_rate"),
)

loss_weights = (
    Real(1e-2, 1e4, "log-uniform", name="mean_unit_cube_loss"),
    Real(1e-6, 1e4, "log-uniform", name="mean_unary_kl_loss"),
    Real(1e-1, 1e4, "log-uniform", name="mean_pull_loss"),
    Real(1e3, 1e8, "log-uniform", name="mean_push_loss"),
    Real(1e-1, 1e4, "log-uniform", name="mean_surrogate_loss"),
)

defaults = {
    "BoxParamType": None,
    "extra_callbacks": (),
    "gradient_clipping": (-1,1),
    "universe_box": None,
    "unit_cube_loss_func": (mean_unit_cube_loss,),
    "projected_gradient_descent": False,
    "vol_func": clamp_volume,
    "plateau_loss_funcs": (),
}

# Box Parametrizations
box_param_options = {
    "UnitBoxes": {"BoxParamType": UnitBoxes, "extra_callbacks": (MinBoxSize())},
    "MinMaxUnitBoxes": {"BoxParamType": MinMaxUnitBoxes},
    "DeltaBoxes": {"BoxParamType": DeltaBoxes},
    "SigmoidBoxes": {"BoxParamType": SigmoidBoxes},
}

# Universe
universe_options = {
    "scaled": {"universe_box": smallest_containing_box, "unit_cube_loss_func": (), "gradient_clipping": (-100, 100)},
    "scaled_penalty": {"universe_box": smallest_containing_box},
    "hard_penalty": {},
    # "hard_projected": {"projected_gradient_descent": True},
}
# TODO: Anything other than "scaled" should probably also initialize parameters in the unit cube.
# This probably isn't too much of an issue, however, as unit cube penalties will quickly make the boxes move to the unit cube.

# Pull Methods
plateau_options = {
    "softbox": {"vol_func": soft_volume},
    "constraint": {"plateau_loss_funcs": (mean_pull_loss, mean_push_loss)},
    # "hardbox": {"plateau_loss_funcs": (mean_surrogate_loss)},
    "none": {},
}

# Hyperparameter Optimization
hyperparameter_optimization = {
    "random": dummy_minimize,
    "forest": forest_minimize,
    "gp": gp_minimize,
    "gbrt": gbrt_minimize,
}

parser = argparse.ArgumentParser()
parser.add_argument("--box_param", choices=box_param_options, help="Type of box parametrization", required=True)
parser.add_argument("--universe", choices=universe_options, help="Volume function", required=True)
parser.add_argument("--plateau", choices=plateau_options, help="Method for pulling boxes together", required=True)
parser.add_argument("--path", help="Folder with serialized training data", required=True)
parser.add_argument("--hyperopt_method", choices=hyperparameter_optimization, help="Optimization method to choose hyperparameters", default="gp")
parser.add_argument("--epochs", help="Number of epochs to train", type=int, default=10)
parser.add_argument("--hyper_calls", help="Number of times to call the hyperparameter optimizer", type=int, default=100)

sql_columns = [
    "[auto_id] INTEGER PRIMARY KEY",
    "[box_param] text",
    "[universe] text",
    "[plateau] text",
    "[hyperopt_method] text",
    "[epochs] integer",
    "[hyper_calls] integer",
    "[random_seed] integer",
    "[dims] integer",
    "[num_batches] integer",
    "[learning_rate] real"
    ]

sql_columns += [f"[{z.name}] real" for z in loss_weights]

args = parser.parse_args()

os.chdir(args.path)

unary_prob = torch.from_numpy(np.loadtxt("train_tc_unary.tsv")).float().to("cuda")
# Data in the rest of these tsvs should be in the form
# A    B    P(B | A)
# where A and B are the line indices from the unary.tsv file.
train = Probs.load_from_julia("", 'train_tc_pos.tsv', 'train_neg.tsv', ratio_neg = 1).to("cuda")
dev = Probs.load_from_julia("", 'dev_pos.tsv', 'dev_neg.tsv', ratio_neg = 1).to("cuda")
# test = PairwiseProbs.load_from_julia(PATH, 'test_pos.tsv', 'test_neg.tsv', ratio_neg = 1, device=default_device)

model = pd.DataFrame()


defaults.update(box_param_options[args.box_param])
defaults.update(universe_options[args.universe])
defaults.update(plateau_options[args.plateau])


num_boxes = unary_prob.shape[0]
BoxParamType = defaults["BoxParamType"]
vol_func = defaults["vol_func"]

loss_funcs = [mean_cond_kl_loss, mean_unary_kl_loss(unary_prob), *defaults["unit_cube_loss_func"], *defaults["plateau_loss_funcs"]]
loss_func_names = [f.__name__ for f in loss_funcs]
loss_weights = [z for z in loss_weights if z.name in loss_func_names]

parameter_space = (*learning_params, *loss_weights)

random_seed = np.random.randint(2**32 - 1)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

args_to_save = vars(args)
args_to_save.pop("path", None)

sql_conn = sqlite3.connect("train.db")

@use_named_args(parameter_space)
def objective(**hyperparameters):
    print(hyperparameters)


    hyperparam_to_save = {**hyperparameters, **args_to_save, "random_seed": random_seed}
    for k, v in hyperparam_to_save.items():
        if hasattr(v, 'dtype'):
            hyperparam_to_save[k] = v.item()

    c = sql_conn.cursor()
    c.execute(f"insert into Training_Instances ({','.join(hyperparam_to_save.keys())}) values ({':' +', :'.join(hyperparam_to_save.keys())})", hyperparam_to_save)
    sql_conn.commit()
    training_instance_id = c.lastrowid # this is guaranteed to be the ID we want,

    b = BoxModel(1, num_boxes, hyperparameters["dims"], BoxParamType, vol_func)
    b.to("cuda")

    train_dl = DataLoader(train, batch_size=math.ceil(train.ids.shape[0]/hyperparameters["num_batches"]), shuffle=True)
    opt = torch.optim.Adam(b.parameters(), lr=hyperparameters["learning_rate"]) #, betas=(0.9, 0.99))

    weighted_loss_funcs = []
    for loss_func in loss_funcs:
        if loss_func.__name__ in hyperparameters:
            weighted_loss_funcs.append((hyperparameters[loss_func.__name__], loss_func))
        else:
            weighted_loss_funcs.append(loss_func)

    loss_func = LossPieces(*weighted_loss_funcs)

    metrics = [metric_num_needing_push, metric_num_needing_pull, metric_hard_accuracy]

    rec_col = RecorderCollection()

    callbacks = CallbackCollection(
        LossCallback(rec_col.train, train, weighted=False),
        LossCallback(rec_col.dev, dev, weighted=False),
        *(MetricCallback(rec_col.dev, dev, m) for m in metrics),
        *(MetricCallback(rec_col.train, train, m) for m in metrics),
        MetricCallback(rec_col.dev, dev, metric_pearson_r),
        MetricCallback(rec_col.dev, dev, metric_spearman_r),
        GradientClipping(*defaults["gradient_clipping"]),
        StopAtMaxLoss(10),
        *defaults["extra_callbacks"]
    )

    l = Learner(train_dl, b, loss_func.loss_func, opt, callbacks, recorder = rec_col.learn)
    l.train(args.epochs)
    try:
        mean_cond_kl_loss_vals = rec_col.dev["mean_cond_kl_loss"]
        mean_cond_kl_min = np.min(mean_cond_kl_loss_vals[~np.isnan(mean_cond_kl_loss_vals)])
    except KeyError:
        mean_cond_kl_min = 100
    print(f"Mean KL Min: {mean_cond_kl_min}\n\n")

    save_record_to_sql(sql_conn, rec_col.train, "Train", training_instance_id)
    save_record_to_sql(sql_conn, rec_col.dev, "Dev", training_instance_id)

    return mean_cond_kl_min

c = sql_conn.cursor()
c.execute(f"create table if not exists Training_Instances ({','.join(sql_columns)})")
res_gp = hyperparameter_optimization[args.hyperopt_method](objective, parameter_space, n_calls=args.hyper_calls, random_state=random_seed)

print(res_gp.fun)
print(res_gp.specs)
print(res_gp.x)
