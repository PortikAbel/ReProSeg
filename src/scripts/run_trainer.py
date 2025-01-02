import random
import numpy as np
import torch

from train.trainer import train_model
from model.util.args import ModelTrainerArgumentParser
from model.util.log import Log


model_trainer_argument_parser = ModelTrainerArgumentParser()
model_args = model_trainer_argument_parser.get_args()

# Create a logger
log = Log(model_args.log_dir, __name__)

# Log the run arguments
model_trainer_argument_parser.save_args(log.metadata_dir)

torch.manual_seed(model_args.seed)
torch.cuda.manual_seed_all(model_args.seed)
random.seed(model_args.seed)
np.random.seed(model_args.seed)

try:
    train_model(log, model_args)
except Exception as e:
    log.exception(e)
log.close()
