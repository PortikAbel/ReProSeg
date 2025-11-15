# Python
import torch
import proto_segmentation.model
from proto_segmentation.model import construct_PPNet


torch.serialization.safe_globals([proto_segmentation.model.PPNet])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ppnet = construct_PPNet().to(device)

ckpt = torch.load(
    '/home/annamari/ProtoSeg-checkpoints/cityscapes_no_kld_imnet_4_16/checkpoints/push_best.pth',
    map_location=device,
    weights_only=False, 
)

print(type(ckpt)) 


ppnet.eval()

