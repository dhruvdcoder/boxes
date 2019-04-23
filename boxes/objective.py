from . import *
from learner import *
from pprint import pprint
from sqlite3 import Connection
from dataclasses import dataclass, field
import math
from sacred import Experiment


@dataclass
class Objective:
    BoxParamType: type
    vol_func: Callable
    loss_funcs: Collection[Callable]
    unary_probs: Tensor
    train: Probs
    dev: Probs
    init_min_vol: float
    universe_box: Callable
    epochs: int
    device: str
    obj_func_to_min: Callable
    extra_callbacks: Collection[Callable] = field(default_factory=list)
    gradient_clipping: Tuple[Optional[float], Optional[float]] = (None, None)
    projected_gradient_descent: bool = False
    hyper_opt_id: Optional[int] = None
    sql_conn: Optional[Connection] = None
    properties: dict = field(default_factory=dict)
    ex: Experiment = field(default_factory=Experiment)

    def __post_init__(self):
        self.num_boxes = self.unary_probs.shape[0]
        self.hyper_opt_iteration = 0
        if self.gradient_clipping != (None, None):
            self.extra_callbacks.append(GradientClipping(*self.gradient_clipping))

    def objective(self, ex, **hyperparameters):
        # Fix random seed
        random_seed = np.random.randint(2**32 - 1)
        torch.manual_seed(random_seed)

        self.hyper_opt_iteration += 1
        print(f"\n[{self.hyper_opt_iteration}] (Hyperoptimization Iteration)")
        pprint(hyperparameters)

        ex.add_config(
            {
                **self.properties,
                **hyperparameters,
                "torch_random_seed": random_seed,
            }
        )

        b = BoxModel(
            BoxParamType = self.BoxParamType,
            vol_func = self.vol_func,
            num_models = 1,
            num_boxes = self.num_boxes,
            dims = hyperparameters["dims"],
            init_min_vol = self.init_min_vol,
            universe_box = self.universe_box
        ).to(self.device)

        train_dl = DataLoader(
            self.train,
            batch_size=math.ceil(self.train.ids.shape[0] / hyperparameters["num_batches"]),
            shuffle=True
        )
        opt = torch.optim.Adam(b.parameters(), lr=hyperparameters["learning_rate"])

        weighted_loss_funcs = []
        for loss_func in self.loss_funcs:
            if loss_func.__name__ in hyperparameters:
                weighted_loss_funcs.append((hyperparameters[loss_func.__name__], loss_func))
            else:
                weighted_loss_funcs.append(loss_func)

        loss_func = LossPieces(*weighted_loss_funcs)

        metrics = [metric_num_needing_push, metric_num_needing_pull, metric_hard_accuracy, metric_hard_f1]

        rec_col = RecorderCollection()

        callbacks = CallbackCollection(
            LossCallback(rec_col.train, self.train, weighted=False),
            LossCallback(rec_col.dev, self.dev, weighted=False),
            *(MetricCallback(rec_col.dev, self.dev, m) for m in metrics),
            *(MetricCallback(rec_col.train, self.train, m) for m in metrics),
            MetricCallback(rec_col.dev, self.dev, metric_pearson_r),
            MetricCallback(rec_col.dev, self.dev, metric_spearman_r),
            PercentIncreaseEarlyStopping(rec_col.dev, "mean_cond_kl_loss", 0.25, 10),
            PercentIncreaseEarlyStopping(rec_col.dev, "mean_cond_kl_loss", 0.5),
            # SacredCallback(rec_col.dev, ex, "VALID"),
            # SacredCallback(rec_col.train, ex, "TRAIN"),
            StopIfNaN(),
            *self.extra_callbacks,
        )

        l = Learner(train_dl, b, loss_func, opt, callbacks, recorder=rec_col.learn, reraise_keyboard_interrupt=True)
        l.train(self.epochs, progress_bar=False)

        obj_to_min = self.obj_func_to_min(rec_col)

        # if self.sql_conn is not None:


        #     training_instance_id = sqlite_logger.write_dict_(self.sql_conn, "Training_Instances", hyperparam_to_save)
        #     sqlite_logger.save_recorder_to_sql_(self.sql_conn, rec_col.train, "Train", training_instance_id)
        #     sqlite_logger.save_recorder_to_sql_(self.sql_conn, rec_col.dev, "Dev", training_instance_id)
        #     print(f"Saved: Training Instance [{training_instance_id}]\n")

        return obj_to_min


def obj_mean_cond_kl_min(rec_col):
        mean_cond_kl_loss_vals = rec_col.dev["mean_cond_kl_loss"]
        mean_cond_kl_loss_min = np.min(mean_cond_kl_loss_vals[~np.isnan(mean_cond_kl_loss_vals)])
        print(f"Mean KL Min: {mean_cond_kl_loss_min}")
        return mean_cond_kl_loss_min
