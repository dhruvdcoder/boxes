import torch
import torch.nn.functional as F
import torch.optim as optim
from .delta_boxes import *
from .unit_boxes import *
from .callbacks import *
from .learner import *
from .probability_dataset import *
from .loss_functions import *
from .box_operations import *
import numpy as np
import pandas as pd

# This is to handle this bug in tqdm, which is fixed in Jupyter but not in JupyterLab:
#   https://github.com/tqdm/tqdm/issues/433
# Workaround:
#   https://github.com/bstriner/keras-tqdm/issues/21
from IPython.core.display import HTML, display
display(HTML("""
<style>
.p-Widget.jp-OutputPrompt.jp-OutputArea-prompt:empty {
  padding: 0;
  border: 0;
}
</style>
"""))
