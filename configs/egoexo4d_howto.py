# Ego-Exo4D, full method (ego + 4 exo views), HowTo100M-pretrained backbone features.
# Edit the paths below to point at your own pre-extracted feature / annotation files.

dataset = "egoexo4d"
num_classes = 4
seed = 8898
multi_gpu = "1"
logname = "1"
ego_only = False
exo_only = False

bs_train = 64
bs_test = 64
num_workers = 0
pretrain = "howto100"
from_command = False

# TODO: point these at your own pre-extracted-feature / annotation directories.
if pretrain == "k400":
    presave_path = "/path/to/presave/384_ego_rgb_10sample_swin_base_patch244_window877_kinetics400_22k_name_8clip"
elif pretrain == "k600":
    presave_path = "/path/to/presave/384_ego_rgb_10sample_swin_base_patch244_window877_kinetics600_22k_name_8clip"
elif pretrain == "howto100":
    presave_path = "/path/to/presave/384_ego_rgb_10sample_TimeSformer_divST_8x32_224_HowTo100M_name_8clip"
    d_model = 768

train_anno_path = "/path/to/demonstrator_simple_train.json"
test_anno_path = "/path/to/demonstrator_simple_test.json"

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
weight_head = 2
dropout = 0.5
tau = 0.95
topk = 2

loss = "ordinal"

# --- auxiliary losses ---
VIBGB = True
vib_dim = 512
head_in_dim = d_model
vib_rate = 0.01

VIB = True
CVML = False
cvml_rate = 0.01

scheduler_step = 40
epoch_num = 40
lr = 1e-5
