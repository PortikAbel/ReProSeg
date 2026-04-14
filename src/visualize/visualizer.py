import heapq
import os
import pickle
from collections import defaultdict

from data.dataloader import DataLoader
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from config import ReProSegConfig
from model.model import ReProSeg
from utils.log import Log

from .utils import activations_to_alpha, draw_activation_minmax_text_on_image, prototype_text


class ModelVisualizer:
    topks_of_concept: dict[int, list[tuple[float, int]]]
    """
    dict mapping concept index to list of (activation score, image index) tuples for the top `k` activations of that concept.
    """
    image_to_concepts: dict[int, list[int]]
    """
    dict mapping image index to list of concept indices that has a top `k` activation (the prototypes) for that image.
    """
    tensors_per_concept: dict[int, list[torch.Tensor]]
    """
    dict mapping concept index to list of tensors representing the top `k` activations (the prototypes) for that concept.
    """

    MIN_ACTIVATION_SCORE = 0.1

    def __init__(self, net: ReProSeg, cfg: ReProSegConfig, log: Log):
        self.net = net
        self.device = cfg.env.device
        self.log = log
        self.image_shape = cfg.data.img_shape
        self.k = cfg.visualization.top_k

    def collect_topk_concept_activations(self, train_loader_visualization: DataLoader):
        topks_path = self.log.prototypes_dir / f"topks_k{self.k}.pkl"
        if os.path.exists(topks_path):
            self.log.info(f"Loading top {self.k} concept activations from {topks_path}")
            with open(topks_path, "rb") as f:
                self.topks_of_concept = pickle.load(f)
            return

        self.log.info(f"Collecting top {self.k} activations for each concept...")
        self.topks_of_concept = defaultdict(list)

        used_concepts = self.net.layers.classification_layer.used_concepts.cpu().tolist()

        img_iter = tqdm(
            enumerate(train_loader_visualization),
            total=len(train_loader_visualization),
            mininterval=100.0,
            desc=f"Searching for top {self.k} concept activations",
            ncols=0,
            file=self.log.tqdm_file,
        )
        for batch_idx, (xs, _ys) in img_iter:
            xs = xs.to(self.device)
            _aspp, aspp_maxpooled, _out = self.net(xs)
            aspp_maxpooled_sums = aspp_maxpooled.sum(dim=(2, 3)).cpu().numpy()
            for i, aspp_maxpooled_sum in enumerate(aspp_maxpooled_sums):
                img_idx = batch_idx * train_loader_visualization.batch_size + i
                for concept in used_concepts:
                    insertion_method = heapq.heappush if len(self.topks_of_concept[concept]) < self.k else heapq.heappushpop
                    insertion_method(self.topks_of_concept[concept], (aspp_maxpooled_sum[concept], img_idx))
        # Save to file
        with open(topks_path, "wb") as f:
            pickle.dump(self.topks_of_concept, f)

    def map_images_to_prototypes(self):
        i_to_p_path = self.log.prototypes_dir / "i_to_p.pkl"
        if os.path.exists(i_to_p_path):
            self.log.info(f"Loading image to prototype mapping from {i_to_p_path}")
            with open(i_to_p_path, "rb") as f:
                self.image_to_concepts = pickle.load(f)
            return

        self.log.info("Mapping images to prototypes based on topk activations...")
        concepts_not_activated = []
        self.image_to_concepts = defaultdict(list)
        for p in self.topks_of_concept.keys():
            scores, img_idxs = zip(*self.topks_of_concept[p], strict=True)
            if any(np.array(scores) > self.MIN_ACTIVATION_SCORE):
                for i in img_idxs:
                    self.image_to_concepts[i].append(p)
            else:
                concepts_not_activated.append(p)
        self.log.info(
            f"{len(concepts_not_activated)} concepts do not have"
            f" any similarity score > {self.MIN_ACTIVATION_SCORE}. "
            "Will be ignored in visualisation."
        )
        with open(i_to_p_path, "wb") as f:
            pickle.dump(self.image_to_concepts, f)

    def collect_prototype_tensors(self, train_loader_visualization: DataLoader):
        proto_dir = self.log.prototypes_dir
        tensors_path = proto_dir / f"tensors_per_prototype_k{self.k}.pkl"
        if os.path.exists(tensors_path):
            self.log.info(f"Loading prototype tensors from {tensors_path}")
            with open(tensors_path, "rb") as f:
                self.tensors_per_concept = pickle.load(f)
            return

        self.log.info(f"Collecting prototype tensors for top {self.k} activations...")
        self.tensors_per_concept = defaultdict(list)
        resize_image = transforms.Resize(size=tuple(self.image_shape))
        pil_to_tensor = transforms.ToTensor()
        img_iter = tqdm(
            enumerate(train_loader_visualization),
            total=len(train_loader_visualization),
            mininterval=100.0,
            desc=f"Collecting top {self.k} activations for each concept",
            ncols=0,
            file=self.log.tqdm_file,
        )
        for i, (xs, ys) in img_iter:
            if i not in self.image_to_concepts.keys():
                continue
            img_to_open = train_loader_visualization.dataset.images[i]
            image = pil_to_tensor(Image.open(img_to_open).convert("RGB"))
            image = resize_image(image)
            xs, ys = xs.to(self.device), ys.to(self.device)
            concept_activations = self.net.interpolate_concept_activations(xs).to(image.device)
            for p in self.image_to_concepts[i]:
                alpha = activations_to_alpha(concept_activations[p])
                prototype_img = torch.cat((image, alpha), 0)
                prototype_img = draw_activation_minmax_text_on_image(
                    prototype_img,
                    concept_activations[p],
                )
                self.tensors_per_concept[p].append(prototype_img)
        with open(tensors_path, "wb") as f:
            pickle.dump(self.tensors_per_concept, f)

    def render_prototypes(self):
        self.log.info(f"Saving top {self.k} prototypes to images...")
        all_tensors = []
        prototype_iter = tqdm(
            self.tensors_per_concept.items(),
            total=len(self.tensors_per_concept),
            mininterval=100.0,
            desc=f"Visualizing top {self.k} activations of concepts",
            ncols=0,
            file=self.log.tqdm_file,
        )
        for p, prototype_tensors in prototype_iter:
            txt_tensor = prototype_text(p, self.image_shape[::-1])
            prototype_tensors.append(txt_tensor)
            grid = torchvision.utils.make_grid(prototype_tensors, nrow=self.k + 1, padding=1)
            torchvision.utils.save_image(
                grid, self.log.prototypes_dir / f"grid_top_{self.k}_activations_of_prototype_{p}.png"
            )
            all_tensors += prototype_tensors
        if len(all_tensors) > 0:
            grid = torchvision.utils.make_grid(all_tensors, nrow=self.k + 1, padding=1)
            torchvision.utils.save_image(grid, self.log.prototypes_dir / f"grid_top_{self.k}_prototype_activations.png")
        else:
            self.log.warning("No concepts to visualize with prototypes.")

    @torch.no_grad()
    def visualize_prototypes(self, train_loader_visualization: DataLoader):
        self.log.info(f"Visualizing top {self.k} prototypes for each concept...")
        self.log.prototypes_dir.mkdir(parents=True, exist_ok=True)
        self.net.eval()
        self.collect_topk_concept_activations(train_loader_visualization)
        self.map_images_to_prototypes()
        self.collect_prototype_tensors(train_loader_visualization)
        self.render_prototypes()
