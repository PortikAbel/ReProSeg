from data.dataloaders import get_dataloaders
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
    f"Device used: {model_args.device} "
    f"{f'with id {model_args.device_ids}' if len(model_args.device_ids) > 0 else ''}",
)

# Obtain the dataloaders
(
    train_loader,
    test_loader,
    train_loader_visualization,
) = get_dataloaders(log, model_args)


# Create a ReProSeg model
net = ReProSeg(args=model_args, log=log)
net = net.to(device=model_args.device)

if not model_args.skip_training:
    from train.trainer import train_model
    try:
        train_model(net, train_loader, test_loader, log, model_args)
    except Exception as e:
        log.exception(e)

if model_args.visualize_prototypes or model_args.visualize_predictions:
    from visualize.visualizer import ModelVisualizer
    visualizer = ModelVisualizer(net, model_args, log)

    if model_args.visualize_prototypes:
        visualizer.visualize_prototypes(train_loader_visualization, k=model_args.visualize_top_k)
    if model_args.visualize_predictions:
        visualizer.visualize_predictions(test_loader)

log.close()
