# EgoExoFitness, full method (ego + 3 exo views), K400-pretrained backbone features.
# Edit the paths below to point at your own pre-extracted feature files.

dataset = "egoexofitness"
num_classes = 5
seed = 8898
multi_gpu = "1"
logname = "1"
ego_only = False
exo_only = False

bs_train = 64
bs_test = 64
num_workers = 0
pretrain = "k400"
from_command = False

# presave_path points at the pretrain root; it must contain
# <root>_ego/, <root>_exol/, <root>_exom/, <root>_exor/ subdirectories
# (see dataloader/egoexofitness.py).
if pretrain == "k400":
    presave_path = "./data/egoexofitness/k400"
elif pretrain == "k600":
    presave_path = "./data/egoexofitness/k600"
elif pretrain == "howto100":
    presave_path = "./data/egoexofitness/howto100m"
    d_model = 768

# EgoExoFitness dataset builds train/test file lists directly from
# presave_path, so no separate annotation file is required.
train_anno_path = "/"
test_anno_path = "/"

# --- network ---
head_type = "simple"
if ego_only:
    neck_type = "TQN_ego"
elif exo_only:
    neck_type = "TQN_exo"
else:
    neck_type = "TQN"

# neck (TQN) settings
if pretrain != "howto100":
    d_model = 1024
q_number = 48
query_var = 1
pe = "query_pe"
max_len = 48
num_head = 2
num_layers_ego = 3
num_layers_exo = 1
weight_layer = 2
weight_head = 4
dropout = 0.5
tau = 0.95
topk = 2

loss = "ordinal"

# --- auxiliary losses ---
VIBGB = True
vib_dim = 512
head_in_dim = d_model
vib_rate = 0.001

VIB = True
CVML = False
cvml_rate = 0.01

scheduler_step = 40
epoch_num = 100
lr = 1e-4
