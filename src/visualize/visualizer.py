import argparse
import random
from collections import defaultdict

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
from visualize import get_patch_size


class ModelVisualizer:
    def __init__(self, net: ReProSeg, args: argparse.Namespace, log: Log):
        self.net = net
        self.args = args
        self.log = log
        self.image_shape = DATASETS[args.dataset]["img_shape"]
        self.patch_size = get_patch_size(self.net, self.args)

    @torch.no_grad()
    def visualize_prototypes(
        self,
        train_loader_visualization,
        k=10,
    ):
        self.log.info(f"Visualizing top {k} prototypes for each class...")

        seen_max = dict()
        saved = dict()
        saved_ys = dict()
        tensors_per_prototype = dict()

        for p in range(self.net.num_prototypes):
            seen_max[p] = 0.0
            saved[p] = 0
            saved_ys[p] = []
            tensors_per_prototype[p] = []

        self.net.eval()
        classification_weights = self.net.layers.classification_layer.weight
        # classification_weights.shape = (num_class, num_prototypes, 1, 1)


        # Show progress on progress bar
        img_iter = tqdm(
            enumerate(train_loader_visualization),
            total=len(train_loader_visualization),
            mininterval=100.0,
            desc=f"Collecting top {k} prototypes for each class",
            ncols=0,
            file=self.log.tqdm_file,
        )

        # Iterate through the data
        images_seen = 0
        topks = dict()
        # Iterate through the training set
        for i, (xs, ys) in img_iter:
            images_seen += 1
            xs, ys = xs.to(self.args.device), ys.to(self.args.device)

            with torch.no_grad():
                # Use the model to classify this batch of input data
                aspp, aspp_maxpooled, out = self.net(xs, inference=True)

                # b here is 1 (dataloader loads 1 image at a time)
                # print(f"aspp output shape before maxpool: {aspp.shape}") # (b, num_prototypes, channels, w, h)
                # print(f"aspp output shape after maxpool: {aspp.shape}") # (b, num_prototypes, w, h)
                # print(f"out shape: {out.shape}") # (b, num_class, input_w, input_h)

                aspp_maxpooled = aspp_maxpooled.squeeze(0)
                aspp = aspp.squeeze(0)

                aspp_maxpooled_sum = aspp_maxpooled.sum(dim=(1,2))


                for p in range(aspp_maxpooled_sum.shape[0]):  # iterating over each prototype's sum
                    c_weight = torch.max(classification_weights[:, p])
                    if c_weight > 1e-3:
                        # ignore prototypes that are not relevant to any class
                        if p not in topks.keys():
                            topks[p] = []

                        if len(topks[p]) < k:
                            topks[p].append((i, aspp_maxpooled_sum[p].item()))
                        else:
                            topks[p] = sorted(
                                topks[p], key=lambda tup: tup[1], reverse=True
                            )
                            if topks[p][-1][1] < aspp_maxpooled_sum[p].item():
                                topks[p][-1] = (i, aspp_maxpooled_sum[p].item())
                            if topks[p][-1][1] == aspp_maxpooled_sum[p].item():
                                # equal scores. randomly chose one
                                # (since dataset is not shuffled so latter images
                                # with same scores can now also get in topk).
                                replace_choice = random.choice([0, 1])
                                if replace_choice > 0:
                                    topks[p][-1] = (i, aspp_maxpooled_sum[p].item())

        prototypes_not_used = []
        i_to_p: dict = defaultdict(list)
        for p in topks.keys():
            img_idxs, scores = zip(*topks[p])
            if any(np.array(scores) > 0.1):
                for i in img_idxs:
                    i_to_p[i].append(p)
            else:
                prototypes_not_used.append(p)

        print(
            len(prototypes_not_used),
            "prototypes do not have any similarity score > 0.1. "
            "Will be ignored in visualisation.",
        )
        abstained = 0
        # Show progress on progress bar
        img_iter = tqdm(
            enumerate(train_loader_visualization),
            total=len(train_loader_visualization),
            mininterval=50.0,
            desc="Visualizing topk",
            ncols=0,
            file=self.log.tqdm_file,
        )
        for i, (xs, ys) in img_iter:
            # shuffle is false so should lead to same order as in imgs
            xs, ys = xs.to(self.args.device), ys.to(self.args.device)
            # Use the model to classify this batch of input data
            with torch.no_grad():
                aspp, aspp_maxpooled, out = self.net(
                    xs, inference=True
                )  # softmaxes has shape (1, num_prototypes, W, H)

                # shape ([1]) because batch size of dataloader is 1
                outmax = torch.amax(out)
                # print(f"max. output probability: {outmax.item()}")
                if outmax.item() == 0.0:
                    abstained += 1

            for p in i_to_p[i]:
                c_weight = torch.max(
                    classification_weights[:, p]
                )  # ignore prototypes that are not relevant to any class
                if (c_weight > 1e-10) or ("pretrain" in folder_name):
                    # get the image patches for all prototype points
                    for h_idx in range(aspp.shape[3]):
                        for w_idx in range(aspp.shape[4]):
                            # # get the h and w index of the max prototype from the p slice
                            # proto_slice = softmaxes[0, p, :, :]
                            # h_idx, w_idx = (proto_slice.max() == proto_slice).nonzero(as_tuple=True)
                            # h_idx, w_idx = h_idx[0], w_idx[0]
                            #img_to_open = imgs[i]
                            img_to_open = train_loader_visualization.dataset.images[i]  # getting the path to the ith image
                            # img_to_open = train_loader_visualization.dataset[i]
                            if isinstance(img_to_open, tuple) or isinstance(
                                img_to_open, list
                            ):  # dataset contains tuples of (img,label)
                                img_to_open = img_to_open[0]

                            # print(img_to_open)

                            image = transforms.Resize(size=tuple(image_shape))(
                                Image.open(img_to_open).convert("RGB")
                            )

                            saved[p] += 1
                            tensors_per_prototype[p].append(image)

        print("Abstained: ", abstained, flush=True)
        all_tensors = []
        for p in range(self.net._num_prototypes):
            if saved[p] > 0:
                # add text next to each topk-grid, to easily see which prototype it is
                text = "P " + str(p)
                txt_image = Image.new(
                    "RGB",
                    (img_tensor_patch.shape[1], img_tensor_patch.shape[2]),
                    (0, 0, 0),
                )
                draw = D.Draw(txt_image)
                draw.text(
                    (
                        img_tensor_patch.shape[0] // 2,
                        img_tensor_patch.shape[1] // 2,
                    ),
                    text,
                    anchor="mm",
                    fill="white",
                )
                txt_tensor = transforms.ToTensor()(txt_image)
                tensors_per_prototype[p].append(txt_tensor)
                # save top-k image patches in grid
                try:
                    grid = torchvision.utils.make_grid(
                        tensors_per_prototype[p], nrow=k + 1, padding=1
                    )
                    torchvision.utils.save_image(grid, self.log.prototypes_dir / f"grid_top_{k}_prototypes_for_class_{p}.png")
                    if saved[p] >= k:
                        all_tensors += tensors_per_prototype[p]
                except Exception:
                    pass
        if len(all_tensors) > 0:
            grid = torchvision.utils.make_grid(all_tensors, nrow=k + 1, padding=1)
            torchvision.utils.save_image(grid, self.log.prototypes_dir / "grid_top_{k}_prototypes.png")
        else:
            print(
                "Pretrained prototypes not visualized. Try to pretrain longer.",
                flush=True,
            )
        return topks


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
