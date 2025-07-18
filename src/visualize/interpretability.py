import argparse
from collections import defaultdict
import heapq
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


from model.model import ReProSeg
from utils.log import Log
from data.config import DATASETS
from .utils import activations_to_alpha, prototype_text, draw_activation_minmax_text_on_image


class ModelInterpretability:
    topks: dict[int, list]
    i_to_p: dict
    tensors_per_prototype: dict[int, list]

    MIN_CLASSIFICATION_WEIGHT = 10
    MIN_ACTIVATION_SCORE = 0.1

    def __init__(self, net: ReProSeg, args: argparse.Namespace, log: Log, k: int = 10):
        self.net = net
        self.args = args
        self.log = log
        self.image_shape = DATASETS[args.dataset]["img_shape"]
        self.k = k

    def collect_topk_prototype_activations(self, train_loader_visualization):
        topks_path = self.log.prototypes_dir / f"topks_k{self.k}.pkl"
        if os.path.exists(topks_path):
            self.log.info(f"Loading top {self.k} prototype activations from {topks_path}")
            with open(topks_path, 'rb') as f:
                self.topks = pickle.load(f)
            return

        self.log.info(f"Collecting top {self.k} prototype activations for each class...")
        classification_weights = self.net.layers.classification_layer.weight
        prototypes_not_used = []
        self.topks = defaultdict(list)

        img_iter = tqdm(
            enumerate(train_loader_visualization),
            total=len(train_loader_visualization),
            mininterval=100.0,
            desc=f"Searching for top {self.k} prototype activations",
            ncols=0,
            file=self.log.tqdm_file,
        )
        for i, (xs, ys) in img_iter:
            xs, ys = xs.to(self.args.device), ys.to(self.args.device)
            _aspp, aspp_maxpooled, _out = self.net(xs)
            aspp_maxpooled = aspp_maxpooled.squeeze(0)
            aspp_maxpooled_sum = aspp_maxpooled.sum(dim=(1,2))
            for p in range(self.net.num_prototypes):
                if torch.max(classification_weights[:, p]) < self.MIN_CLASSIFICATION_WEIGHT:
                    if p not in prototypes_not_used:
                        prototypes_not_used.append(p)
                    continue
                score = aspp_maxpooled_sum[p].item()
                if len(self.topks[p]) < self.k:
                    heapq.heappush(self.topks[p], (score, i))
                else:
                    heapq.heappushpop(self.topks[p], (score, i))
        self.log.info(
            f"{len(prototypes_not_used)} prototypes do not have"
            f" any class connection > {self.MIN_CLASSIFICATION_WEIGHT}. "
            "Will be ignored in visualisation."
        )
        # Save to file
        with open(topks_path, 'wb') as f:
            pickle.dump(self.topks, f)

    def map_images_to_prototypes(self):
        i_to_p_path = self.log.prototypes_dir / "i_to_p.pkl"
        if os.path.exists(i_to_p_path):
            self.log.info(f"Loading image to prototype mapping from {i_to_p_path}")
            with open(i_to_p_path, 'rb') as f:
                self.i_to_p = pickle.load(f)
            return 
        
        self.log.info("Mapping images to prototypes based on topk activations...")
        prototypes_not_activated = []
        self.i_to_p = defaultdict(list)
        for p in self.topks.keys():
            scores, img_idxs = zip(*self.topks[p], strict=True)
            if any(np.array(scores) > self.MIN_ACTIVATION_SCORE):
                for i in img_idxs:
                    self.i_to_p[i].append(p)
            else:
                prototypes_not_activated.append(p)
        self.log.info(
            f"{len(prototypes_not_activated)} prototypes do not have"
            f" any similarity score > {self.MIN_ACTIVATION_SCORE}. "
            "Will be ignored in visualisation."
        )
        with open(i_to_p_path, 'wb') as f:
            pickle.dump(self.i_to_p, f)

    def collect_prototype_tensors(self, train_loader_visualization):
        proto_dir = self.log.prototypes_dir
        tensors_path = proto_dir / f"tensors_per_prototype_k{self.k}.pkl"
        if os.path.exists(tensors_path):
            self.log.info(f"Loading prototype tensors from {tensors_path}")
            with open(tensors_path, 'rb') as f:
                self.tensors_per_prototype = pickle.load(f)
            return
        
        self.log.info(f"Collecting prototype tensors for top {self.k} activations...")
        self.tensors_per_prototype = defaultdict(list)
        resize_image = transforms.Resize(size=tuple(self.image_shape))
        pil_to_tensor = transforms.ToTensor()
        img_iter = tqdm(
            enumerate(train_loader_visualization),
            total=len(train_loader_visualization),
            mininterval=100.0,
            desc=f"Collecting top {self.k} activations for each prototype",
            ncols=0,
            file=self.log.tqdm_file,
        )
        for i, (xs, ys) in img_iter:
            if i not in self.i_to_p.keys():
                continue
            img_to_open = train_loader_visualization.dataset.images[i]
            log.info(f"Processing image {i} ({img_to_open}) with prototypes {self.i_to_p[i]}")
            image = pil_to_tensor(Image.open(img_to_open).convert("RGB"))
            image = resize_image(image)
            xs, ys = xs.to(self.args.device), ys.to(self.args.device)
            prototype_activations = self.net.interpolate_prototype_activations(xs).to(image.device)
            for p in self.i_to_p[i]:
                alpha = activations_to_alpha(prototype_activations[p])
                prototype_img = torch.cat((image, alpha), 0)
                prototype_img = draw_activation_minmax_text_on_image(
                    prototype_img,
                    prototype_activations[p],
                )
                self.tensors_per_prototype[p].append(prototype_img)
        with open(tensors_path, 'wb') as f:
            pickle.dump(self.tensors_per_prototype, f)

    def render_prototype_activations(self):
        self.log.info(f"Saving top {self.k} prototype activations to images...")
        all_tensors = []
        prototype_iter = tqdm(
            self.tensors_per_prototype.items(),
            total=len(self.tensors_per_prototype),
            mininterval=100.0,
            desc=f"Visualizing top {self.k} activations of prototypes",
            ncols=0,
            file=self.log.tqdm_file,
        )
        for p, prototype_tensors in prototype_iter:
            txt_tensor = prototype_text(p, self.image_shape[::-1])
            prototype_tensors.append(txt_tensor)
            grid = torchvision.utils.make_grid(prototype_tensors, nrow=self.k + 1, padding=1)
            torchvision.utils.save_image(
                grid,
                self.log.prototypes_dir / f"grid_top_{self.k}_activations_of_prototype_{p}.png"
            )
            all_tensors += prototype_tensors
        if len(all_tensors) > 0:
            grid = torchvision.utils.make_grid(all_tensors, nrow=self.k + 1, padding=1)
            torchvision.utils.save_image(grid, self.log.prototypes_dir / f"grid_top_{self.k}_prototype_activations.png")
        else:
            self.log.warning("Pretrained prototypes not visualized. Try to pretrain longer.")

    @torch.no_grad()
    def visualize_prototypes(self, train_loader_visualization):
        self.log.info(f"Visualizing top {self.k} prototypes for each class...")
        self.net.eval()
        self.collect_topk_prototype_activations(train_loader_visualization)
        self.map_images_to_prototypes()
        self.collect_prototype_tensors(train_loader_visualization)
        self.render_prototype_activations()

    @torch.no_grad()
    def calculate_consistency_score(self, train_loader_visualization):
        self.log.info("Calculating consistency score for each prototype...")
        self.net.eval()
        img_iter = tqdm(
            enumerate(train_loader_visualization),
            total=len(train_loader_visualization),
            mininterval=100.0,
            desc=f"Collecting top {self.k} activations for each prototype",
            ncols=0,
            file=self.log.tqdm_file,
        )

        object_part_ids = set()
        for i, (xs, ys) in img_iter:
            img_to_open = train_loader_visualization.dataset.images[i]
            # self.log.info(f"Processing image {i} ({img_to_open})")
            # Replace 'leftImg8bit' in the directory path to find panoptic labels
            parts = list(Path(img_to_open).parts)
            parts[parts.index("leftImg8bit")] = "gtFinePanopticParts"

            # Create the new path object with updated directory
            path_to_panoptic_labels = Path(*parts)

            # Replace 'leftImg8bit' in the filename and change extension to .tif
            new_filename = path_to_panoptic_labels.name.replace("leftImg8bit", "gtFinePanopticParts").replace(".png", ".tif")

            # Combine updated directory with new filename
            panoptic_labels = path_to_panoptic_labels.with_name(new_filename)
            if panoptic_labels.exists():
                # self.log.info(f"panoptic labels found: {panoptic_labels}")

                # panoptic labels format:
                # 10^5 * semantic_id + 10^2 * instance_id + part_id
                panoptic_labels_image = Image.open(panoptic_labels)
                panoptic_labels_array = np.array(panoptic_labels_image)
                panoptic_labels_array_filtered = ((panoptic_labels_array//100000)*100000 + 
                                                  panoptic_labels_array%100).astype(np.int32)

                # Flatten and get unique values with their counts
                unique_values, counts = np.unique(panoptic_labels_array_filtered, return_counts=True)
                # Filter out values that belong to semantic class void (i.e., those less than or equal to 100000)
                object_part_ids.update({x for x in unique_values if x > 100000})

        # print(object_part_ids)                

