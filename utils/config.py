from yacs.config import CfgNode as CN


def log_msg(msg, mode="INFO"):
    color_map = {"INFO": 36, "TRAIN": 32, "EVAL": 31}
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def show_cfg(cfg):
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.MODEL = cfg.MODEL
    dump_cfg.SOLVER = cfg.SOLVER
    dump_cfg.LOG = cfg.LOG
    print(log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))


def save_cfg(cfg, path):
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.MODEL = cfg.MODEL
    dump_cfg.SOLVER = cfg.SOLVER
    dump_cfg.LOG = cfg.LOG
    with open(path, "w") as f:
        f.write("CONFIG:\n{}".format(dump_cfg.dump()))


CFG = CN()

# Result dir structure
# - folder name: Log_{time}_{tag}
# - folder structure:
#   - checkpoints folder
#   - worklog.txt
#   - config.yaml

# Experiment
CFG.EXPERIMENT = CN(new_allowed=True)
CFG.EXPERIMENT.PROJECT = ""
CFG.EXPERIMENT.NAME = ""
CFG.EXPERIMENT.TAG = "default"
CFG.EXPERIMENT.SEED = (
    42  # Random number seed, which is beneficial to the repeatability of the experiment.
)
CFG.EXPERIMENT.TASK = "train"  # train, test, pretext
CFG.EXPERIMENT.DEBUG = False  # Debug mode
CFG.EXPERIMENT.GPU_IDS = "0"  # List of GPUs used
CFG.EXPERIMENT.WORLD_SIZE = 2  # Number of GPUs used
CFG.EXPERIMENT.REPETITION_NUM = 5  # Number of repetition times
CFG.EXPERIMENT.RESUME = False  # Resume training
CFG.EXPERIMENT.CHECKPOINT = ""  # 'Log_2020-03-19_19-53-27'
CFG.EXPERIMENT.CHKP_IDX = (
    None  # Choose index of checkpoint to start from. If None, uses the latest chkp
)
CFG.EXPERIMENT.CHECKPOINT_GAP = 50

# Dataset
CFG.DATASET = CN()
CFG.DATASET.TYPE = "CamCAN"
CFG.DATASET.ROOT = "/home/song/datasets/MEG_datasets/"
CFG.DATASET.CHANNELS = 204
CFG.DATASET.POINTS = 100
CFG.DATASET.NUM_CLASSES = 2
CFG.DATASET.NUM_WORKERS = 2
CFG.DATASET.TEST = CN()
CFG.DATASET.TEST.BATCH_SIZE = 1024
CFG.DATASET.MEAN = -0.1158333
CFG.DATASET.STD = 20.67235566

# Model
CFG.MODEL = CN()
CFG.MODEL.TYPE = ""
CFG.MODEL.ARGS = CN(new_allowed=True)
CFG.MODEL.ARGS.SIAMESE = False
CFG.MODEL.CRITERION = CN(new_allowed=True)

# Solver
CFG.SOLVER = CN(new_allowed=True)
CFG.SOLVER.TRAINER = "base"
CFG.SOLVER.BATCH_SIZE = 1024  # Grid search
CFG.SOLVER.EPOCHS = 100
CFG.SOLVER.LR = 0.003
# CFG.SOLVER.LR_DECAY_STAGES = [150, 180, 210]
# CFG.SOLVER.LR_DECAY_RATE = 0.1
CFG.SOLVER.WEIGHT_DECAY = 0.000
CFG.SOLVER.MOMENTUM = 0.9
CFG.SOLVER.LAMBDA_L1 = 0.0
CFG.SOLVER.TYPE = "SGD"
CFG.SOLVER.SCHEDULER = CN(new_allowed=True)

CFG.EVAL_LINEAR = CN(new_allowed=True)
CFG.EVAL_LINEAR.CLASSIFIER= "LinearClassifier"
CFG.EVAL_LINEAR.CRITERION="CE"
CFG.EVAL_LINEAR.BATCH_SIZE= 512
CFG.EVAL_LINEAR.LR=0.01
CFG.EVAL_LINEAR.SCHEDULER=CN(new_allowed=True)
CFG.EVAL_LINEAR.SCHEDULER.TYPE="ExponentialLR"
CFG.EVAL_LINEAR.SCHEDULER.GAMMA=0.99
CFG.EVAL_LINEAR.SCHEDULER.MIN_LR=0.0001

CFG.EVAL_SEMI = CN(new_allowed=True)
CFG.EVAL_SEMI.EPOCHS = 50
CFG.EVAL_SEMI.TYPE = "SGD"
CFG.EVAL_SEMI.DATA_PATH=""
CFG.EVAL_SEMI.CLASSIFIER= "LinearClassifier"
CFG.EVAL_SEMI.BATCH_SIZE= 512
CFG.EVAL_SEMI.WEIGHT_DECAY = 0.000
CFG.EVAL_SEMI.MOMENTUM = 0.9
CFG.EVAL_SEMI.LR=0.01
CFG.EVAL_SEMI.SCHEDULER=CN(new_allowed=True)
CFG.EVAL_SEMI.SCHEDULER.TYPE="ExponentialLR"
CFG.EVAL_SEMI.SCHEDULER.GAMMA=0.99
CFG.EVAL_SEMI.SCHEDULER.MIN_LR=0.0001
CFG.EVAL_SEMI.SAVE_CKPT=False
CFG.EVAL_SEMI.SAVE_CKPT_GAP=10
# Log
CFG.LOG = CN()
# CFG.LOG.SAVE_CHECKPOINT_FREQ = 20
CFG.LOG.PREFIX = "/home/song/code/current/meg_classification/ssl/results/"
