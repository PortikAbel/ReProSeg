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
            _aspp, aspp_maxpooled, _out = self.net(xs, inference=True)
            # b here is 1 (dataloader loads 1 image at a time)
            # aspp.shape           = (b, num_prototypes, channels, w, h)
            # aspp_maxpooled.shape = (b, num_prototypes, w, h)
            # out.shape            = (b, num_class, input_w, input_h)

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
            aspp, _aspp_maxpooled, _out = self.net(xs, inference=True)
            aspp = aspp.squeeze(0)

            for p in i_to_p[i]:
                alpha = self._interpolate_prototypes(aspp[p]).unsqueeze(0)
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

    def _interpolate_prototypes(self, activations: torch.Tensor) -> torch.Tensor:
        max_scale = torch.argmax(activations, dim=(0))
        scales = activations.shape[0]
        
        interpolated_activations = torch.zeros(self.image_shape, device=activations.device)
        upscale = transforms.Resize(size=self.image_shape, interpolation=transforms.InterpolationMode.NEAREST_EXACT)
        for scale in range(scales):
            activations[scale, max_scale!=scale]=0
            padding = scale + 1
            max_pool = torch.nn.MaxPool2d(kernel_size=2*padding+1, padding=padding, stride=1)
            scaled_activations = upscale(max_pool(activations[scale].unsqueeze(0))).squeeze(0)
            interpolated_activations = torch.maximum(interpolated_activations, scaled_activations)

        interpolated_activations = interpolated_activations.where(
            interpolated_activations >= interpolated_activations.mean(), 0)
        interpolated_activations /= interpolated_activations.max()

        return interpolated_activations


    def visualize_predictions(
        self,
        projectloader,
    ):
        print("Visualizing prototypes...", flush=True)
        result_dir = args.log_dir / foldername
        result_dir.mkdir(parents=True, exist_ok=True)

        near_imgs_dirs = dict()
        seen_max = dict()
        saved = dict()
        saved_ys = dict()
        tensors_per_prototype = dict()
        abstainedimgs = set()
        notabstainedimgs = set()

        for p in range(net._num_prototypes):
            near_imgs_dir = result_dir / str(p)
            near_imgs_dirs[p] = near_imgs_dir
            seen_max[p] = 0.0
            saved[p] = 0
            saved_ys[p] = []
            tensors_per_prototype[p] = []

        imgs = projectloader.dataset.imgs

        # Make sure the model is in evaluation mode
        net.eval()
        classification_weights = net._classification.weight
        # Show progress on progress bar
        img_iter = tqdm(
            enumerate(projectloader),
            total=len(projectloader),
            mininterval=100.0,
            desc="Visualizing",
            ncols=0,
            file=log.tqdm_file,
        )

        # Iterate through the data
        images_seen_before = 0
        for i, (xs, ys) in img_iter:  # shuffle is false so should lead to same order as in imgs
            xs, ys = xs.to(device), ys.to(device)
            # Use the model to classify this batch of input data
            with torch.no_grad():
                softmaxes, _, out = net(xs, inference=True)

            max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
            # In PyTorch, images are represented as [channels, height, width]
            max_per_prototype_h, max_idx_per_prototype_h = torch.max(
                max_per_prototype, dim=1
            )
            max_per_prototype_w, max_idx_per_prototype_w = torch.max(
                max_per_prototype_h, dim=1
            )
            for p in range(0, net._num_prototypes):
                c_weight = torch.max(
                    classification_weights[:, p]
                )  # ignore prototypes that are not relevant to any class
                if c_weight > 0:
                    h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                    w_idx = max_idx_per_prototype_w[p]
                    idx_to_select = max_idx_per_prototype[p, h_idx, w_idx].item()
                    found_max = max_per_prototype[p, h_idx, w_idx].item()

                    img_name = imgs[images_seen_before + idx_to_select]
                    if out.max() < 1e-8:
                        abstainedimgs.add(img_name)
                    else:
                        notabstainedimgs.add(img_name)

                    if found_max > seen_max[p]:
                        seen_max[p] = found_max

                    if found_max > 0.5:
                        img_to_open = imgs[images_seen_before + idx_to_select]
                        if isinstance(img_to_open, tuple) or isinstance(
                            img_to_open, list
                        ):  # dataset contains tuples of (img,label)
                            img_label = img_to_open[1]
                            img_to_open = img_to_open[0]

                        (
                            image,
                            img_tensor_patch,
                            h_coord_max,
                            h_coord_min,
                            w_coord_max,
                            w_coord_min,
                        ) = get_patch(img_to_open, args, h_idx, w_idx, softmaxes)
                        saved[p] += 1
                        tensors_per_prototype[p].append((img_tensor_patch, found_max))

                        save_path = result_dir / f"prototype_{p}"
                        save_path.mkdir(parents=True, exist_ok=True)

                        draw = D.Draw(image)
                        draw.rectangle(
                            ((w_coord_min, h_coord_min), (w_coord_max, h_coord_max)),
                            outline="yellow",
                            width=2,
                        )
                        image.save(
                            save_path
                            / f"p{p}_{img_label}_{round(found_max, 2)}_{img_to_open.stem}"
                            f"_rect.png"
                        )

            images_seen_before += len(ys)

        print("num images abstained: ", len(abstainedimgs), flush=True)
        print("num images not abstained: ", len(notabstainedimgs), flush=True)
        for p in range(net._num_prototypes):
            if saved[p] > 0:
                try:
                    sorted_by_second = sorted(
                        tensors_per_prototype[p],
                        key=lambda tup: tup[1],
                        reverse=True,
                    )
                    sorted_ps = [i[0] for i in sorted_by_second]
                    grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                    torchvision.utils.save_image(grid, result_dir / f"grid_{p}.png")
                except RuntimeError:
                    pass
