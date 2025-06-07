import argparse
import random
from collections import defaultdict
from queue import PriorityQueue

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw as D
from tqdm import tqdm


from model.model import ReProSeg
from utils.log import Log
from data.config import DATASETS


class ModelVisualizer:
    def __init__(self, net: ReProSeg, args: argparse.Namespace, log: Log):
        self.net = net
        self.args = args
        self.log = log
        self.image_shape = DATASETS[args.dataset]["img_shape"]

    @torch.no_grad()
    def visualize_prototypes(
        self,
        train_loader_visualization,
        k=10,
    ):
        self.log.info(f"Visualizing top {k} prototypes for each class...")

        self.net.eval()
        classification_weights = self.net.layers.classification_layer.weight
        # classification_weights.shape = (num_class, num_prototypes, 1, 1)

        prototypes_not_used = []
        topks: dict[int, PriorityQueue] = defaultdict(lambda: PriorityQueue(maxsize=k))

        MIN_CLASSIFICATION_WEIGHT = 1e-3
        MIN_ACTIVATION_SCORE = 0.1

        # Iterate through the dataset to collect top-k activations from images of each prototype
        img_iter = tqdm(
            enumerate(train_loader_visualization),
            total=len(train_loader_visualization),
            mininterval=100.0,
            desc=f"Searching for top {k} prototype activations",
            ncols=0,
            file=self.log.tqdm_file,
        )
        for i, (xs, ys) in img_iter:
            xs, ys = xs.to(self.args.device), ys.to(self.args.device)
            _aspp, aspp_maxpooled, _out = self.net(xs)

            aspp_maxpooled = aspp_maxpooled.squeeze(0)
            aspp_maxpooled_sum = aspp_maxpooled.sum(dim=(1,2))

            for p in range(self.net.num_prototypes):
                # iterating over each prototype's sum
                if torch.max(classification_weights[:, p]) < MIN_CLASSIFICATION_WEIGHT:
                    # ignore prototypes that are not relevant to any class
                    continue

                if not topks[p].full():
                    topks[p].put((aspp_maxpooled_sum[p].item(), i))
                elif topks[p].queue[0][0] < aspp_maxpooled_sum[p].item():
                    # equal scores. randomly chose one
                    # (since dataset is not shuffled so latter images
                    # with same scores can now also get in topk).
                    replace_choice = random.choice([0, 1])
                    if replace_choice > 0:
                        topks[p].get()
                        topks[p].put((aspp_maxpooled_sum[p].item(), i))
        self.log.info(
            f"{len(prototypes_not_used)} prototypes do not have"
            f"any class connection > {MIN_CLASSIFICATION_WEIGHT}. "
            "Will be ignored in visualisation.",
        )
        already_ignored_count = len(prototypes_not_used)
    
        i_to_p: dict = defaultdict(list)
        for p in topks.keys():
            scores, img_idxs = zip(*topks[p].queue)
            if any(np.array(scores) > MIN_ACTIVATION_SCORE):
                for i in img_idxs:
                    i_to_p[i].append(p)
            else:
                prototypes_not_used.append(p)
        self.log.info(
            f"{len(prototypes_not_used)-already_ignored_count} prototypes do not have"
            f"any similarity score > {MIN_ACTIVATION_SCORE}. "
            "Will be ignored in visualisation.",
        )

        tensors_per_prototype: dict[int, list] = defaultdict(list)
        resize_image = transforms.Resize(size=tuple(self.image_shape))
        pil_to_tensor = transforms.ToTensor()

        img_iter = tqdm(
            enumerate(train_loader_visualization),
            total=len(train_loader_visualization),
            mininterval=100.0,
            desc=f"Collecting top {k} activations for each prototype",
            ncols=0,
            file=self.log.tqdm_file,
        )
        for i, (xs, ys) in img_iter:
            if i not in i_to_p.keys():
                # skip images that do not have any top-k prototype activations
                continue

            # open original image
            img_to_open = train_loader_visualization.dataset.images[i]
            image = pil_to_tensor(Image.open(img_to_open).convert("RGB")).to(self.args.device)
            image = resize_image(image)

            xs, ys = xs.to(self.args.device), ys.to(self.args.device)
            prototype_activations = self.net.interpolate_prototype_activations(xs)

            for p in i_to_p[i]:
                alpha = prototype_activations[p]
                prototype_img = torch.cat((image, alpha), 0)
                tensors_per_prototype[p].append(prototype_img)

        all_tensors = []
        prototype_iter = tqdm(
            tensors_per_prototype.items(),
            total=len(tensors_per_prototype),
            mininterval=100.0,
            desc=f"Visualizing top {k} activations of prototypes",
            ncols=0,
            file=self.log.tqdm_file,
        )
        for p, prototype_tensors in prototype_iter:
            # add text next to each topk-grid, to easily see which prototype it is
            txt_image = Image.new("RGBA", self.image_shape[::-1], (0, 0, 0))
            draw = D.Draw(txt_image)
            draw.text(
                tuple(s // 2 for s in self.image_shape[::-1]),
                f"Prototype {p}",
                anchor="mm",
                fill="white",
            )
            txt_tensor = pil_to_tensor(txt_image).to(self.args.device)
            prototype_tensors.append(txt_tensor)
            # save top-k image patches in grid
            grid = torchvision.utils.make_grid(prototype_tensors, nrow=k + 1, padding=1)
            torchvision.utils.save_image(grid, self.log.prototypes_dir / f"grid_top_{k}_activations_of_prototype_{p}.png")
            all_tensors += prototype_tensors
        
        if len(all_tensors) > 0:
            grid = torchvision.utils.make_grid(all_tensors, nrow=k + 1, padding=1)
            torchvision.utils.save_image(grid, self.log.prototypes_dir / f"grid_top_{k}_prototype_activations.png")
        else:
            self.log.warning("Pretrained prototypes not visualized. Try to pretrain longer.")
