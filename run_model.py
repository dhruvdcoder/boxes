#!/bin/env python
from boxes import *
import math
import argparse
import os
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize, gbrt_minimize


@dataclass
class RecorderCollection:
    learn:Recorder = field(default_factory=Recorder)
    train:Recorder = field(default_factory=Recorder)
    dev:Recorder = field(default_factory=Recorder)


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
    "universe_box": None,
    "unit_cube_loss_func": (mean_unit_cube_loss,),
    "projected_gradient_descent": False,
    "vol_func": clamp_volume,
    "plateau_loss_funcs": (),
}

# Box Parametrizations
box_param_options = {
    "UnitBoxes": {"BoxParamType": UnitBoxes},
    "MinMaxUnitBoxes": {"BoxParamType": MinMaxUnitBoxes},
    "DeltaBoxes": {"BoxParamType": DeltaBoxes},
    "SigmoidBoxes": {"BoxParamType": SigmoidBoxes},
}

# Universe
universe_options = {
    "scaled": {"universe_box": smallest_containing_box, "unit_cube_loss_func": ()},
    "scaled_penalty": {"universe_box": smallest_containing_box},
    "hard_penalty": {},
    # "hard_projected": {"projected_gradient_descent": True},
}
# TODO: Anything other than "scaled" should probably also initialize parameters in the unit cube, but maybe not.

# Pull Methods
plateau_options = {
    "softbox": {"vol_func": soft_volume},
    "constraint": {"plateau_loss_funcs": (mean_pull_loss, mean_push_loss)},
    # "hardbox": {"plateau_loss_funcs": (mean_surrogate_loss)},
    "none": {},
}

parser = argparse.ArgumentParser()
parser.add_argument("--box_param", choices=box_param_options, help="Type of box parametrization", required=True)
parser.add_argument("--universe", choices=universe_options, help="Volume function", required=True)
parser.add_argument("--plateau", choices=plateau_options, help="Method for pulling boxes together", required=True)
parser.add_argument("--path", help="Folder with serialized training data", required=True)

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

@use_named_args(parameter_space)
def objective(**hyperparameters):
    print(hyperparameters)

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
        LossCallback(rec_col.train, train),
        LossCallback(rec_col.dev, dev),
        *(MetricCallback(rec_col.dev, dev, m) for m in metrics),
        *(MetricCallback(rec_col.train, train, m) for m in metrics),
        MetricCallback(rec_col.dev, dev, metric_pearson_r),
        MetricCallback(rec_col.dev, dev, metric_spearman_r),
        GradientClipping(-1,1),
        #     MinBoxSize(),
    )

    l = Learner(train_dl, b, loss_func.loss_func, opt, callbacks, recorder = rec_col.learn)
    l.train(10)
    try:
        mean_cond_kl_loss_vals = rec_col.dev["mean_cond_kl_loss"]
        mean_cond_kl_min = np.min(mean_cond_kl_loss_vals[~np.isnan(mean_cond_kl_loss_vals)])
    except KeyError:
        mean_cond_kl_min = 100
    print(f"Mean KL Min: {mean_cond_kl_min}\n\n")
    return mean_cond_kl_min

res_gp = gbrt_minimize(objective, parameter_space, n_calls=10, random_state=0, n_jobs=-1)
