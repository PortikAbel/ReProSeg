from data.dataloader import DataLoader, DoubleAugmentDataLoader, PanopticPartsDataLoader
from model.model import ReProSeg
from utils.args import ModelTrainerArgumentParser
from utils.log import Log


model_trainer_argument_parser = ModelTrainerArgumentParser()
model_args = model_trainer_argument_parser.get_args()

# Create a logger
log = Log(model_args.log_dir, __name__)

# Log the run arguments
model_trainer_argument_parser.save_args(log.metadata_dir)

# Log which device was actually used
log.info(
    f"Device used: {model_args.device} {f'with id {model_args.device_ids}' if len(model_args.device_ids) > 0 else ''}",
)

# Create the dataloaders
train_loader = DoubleAugmentDataLoader(model_args)
test_loader = DataLoader("test", model_args)
train_loader_visualization = DataLoader("train", model_args)
panoptic_parts_loader = PanopticPartsDataLoader("train", model_args)


# Create a ReProSeg model
net = ReProSeg(args=model_args, log=log)
net = net.to(device=model_args.device)

if not model_args.skip_training:
    from train.trainer import train_model

    try:
        train_model(net, train_loader, test_loader, log, model_args)
    except Exception as e:
        log.exception(e)

if model_args.visualize_prototypes:
    from visualize.visualizer import ModelVisualizer

    visualizer = ModelVisualizer(net, model_args, log, k=model_args.visualize_top_k)
    visualizer.visualize_prototypes(train_loader_visualization)

if model_args.consistency_score:
    from visualize.interpretability import ModelInterpretability
    interpretability = ModelInterpretability(net, model_args, log, consistency_threshold=model_args.consistency_threshold)
    interpretability.compute_prototype_consistency_score(panoptic_parts_loader)    

log.close()
