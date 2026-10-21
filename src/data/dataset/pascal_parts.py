"""VOC semantic masks paired with Pascal Panoptic Parts v2 TIFF annotations."""

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as TF

from config.schema.data import DataConfig, DatasetType
from data.data_split import DataSplit
from data.dataset.factory import DatasetFactory
from data.dataset.label_mapping import VOC_CLASSES


def resolve_voc_root(path: Path) -> Path:
    """Accept a torchvision data root, VOCdevkit directory, or VOC2012 directory."""
    for candidate in (path / "VOCdevkit/VOC2012", path / "VOC2012", path):
        if (candidate / "ImageSets/Segmentation").is_dir() and (candidate / "JPEGImages").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find VOC2012/JPEGImages and ImageSets/Segmentation under {path}.")


class PascalPartsDataset(Dataset):
    """Evaluate the intersection of a VOC split and its part annotations.

    Semantic IDs stay in VOC's 0..20 numbering. Void pixels and parts whose
    semantic class disagrees with VOC are ignored. Countable part IDs are
    folded according to the supplied PPP v2 dataset specification, then
    encoded as semantic_id * 100 + part_id, as in PanopticPartsDataset.
    """

    def __init__(
        self,
        path: Path,
        *,
        parts_path: Path | None = None,
        parts_spec: Path | None = None,
        image_shape: tuple[int, int] | None = None,
        official_parts_only: bool = False,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.voc_root = resolve_voc_root(Path(path))
        self.parts_path = Path(parts_path) if parts_path is not None else self.voc_root / "labels/val"
        self.parts_spec = Path(parts_spec) if parts_spec is not None else self.voc_root / "parts.yaml"
        if not self.parts_path.is_dir():
            raise FileNotFoundError(f"Missing Pascal parts directory: {self.parts_path}. Pass --parts-path.")
        if not self.parts_spec.is_file():
            raise FileNotFoundError(f"Missing Pascal parts specification: {self.parts_spec}. Pass --parts-spec.")
        self.image_shape = image_shape
        self.mean, self.std = mean, std
        self.official_parts_only = official_parts_only
        cfg = DataConfig(dataset=DatasetType.VOC_SEGMENTATION, path=self.voc_root.parent.parent)
        self.dataset = DatasetFactory.create(cfg, split=DataSplit.VAL)
        self.indices = [
            index
            for index, image in enumerate(self.dataset.images)
            if (self.parts_path / f"{Path(image).stem}.tif").is_file()
        ]
        self.image_ids = [Path(self.dataset.images[index]).stem for index in self.indices]
        self.num_missing_parts = len(self.dataset) - len(self.indices)
        if not self.indices:
            raise ValueError(f"No VOC validation images have matching part TIFFs in {self.parts_path}.")

        spec = OmegaConf.to_container(OmegaConf.load(self.parts_spec))
        if not isinstance(spec, dict) or not isinstance(spec.get("scene_class2part_classes"), dict):
            raise ValueError("Pascal parts specification must contain a scene_class2part_classes mapping.")
        classes = spec["scene_class2part_classes"]
        groups = spec.get("countable_pids_groupings", {})
        if not isinstance(groups, dict):
            raise ValueError("countable_pids_groupings must be a mapping.")
        names = ["diningtable" if name == "table" else name for name in list(classes)[:20]]
        if names != [entry.name for entry in VOC_CLASSES[1:]]:
            raise ValueError("Pascal parts specification must list the 20 VOC foreground classes in VOC ID order.")
        self.part_names = {}
        self.part_lookup = torch.zeros((21, 100), dtype=torch.long)
        for sid, (name, parts) in enumerate(classes.items(), start=1):
            if sid > 20:
                break
            for pid, part in enumerate(parts, start=1):
                self.part_names[(sid, pid)] = part
                for raw_pid in groups.get(name, {}).get(part, [pid]):
                    self.part_lookup[sid, raw_pid] = pid

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        image, target = self.dataset[self.indices[index]]
        image = TF.to_dtype(TF.to_image(image), torch.float32, scale=True)
        target = torch.from_numpy(np.array(target, dtype=np.int64)).unsqueeze(0)
        with Image.open(self.parts_path / f"{self.image_ids[index]}.tif") as file:
            uids = torch.from_numpy(np.array(file, dtype=np.int64)).unsqueeze(0)
        if uids.shape != target.shape or image.shape[-2:] != target.shape[-2:]:
            raise ValueError(f"Image/semantic/part dimensions differ for {self.image_ids[index]}.")

        sids, raw_pids = uids // 100_000, uids % 100
        valid = (uids >= 100_000) & (sids >= 1) & (sids <= 20) & (raw_pids > 0) & (sids == target)
        pids = self.part_lookup[sids.clamp(0, 20), raw_pids]
        if self.official_parts_only:
            valid &= pids > 0
        else:
            # Retain unknown positive IDs, while folding all documented IDs.
            pids = torch.where(pids > 0, pids, raw_pids)
        parts = (sids * 100 + pids).masked_fill(~valid, 0)
        target = target.masked_fill(target == 255, 0)
        if self.image_shape is not None:
            image = TF.center_crop(image, self.image_shape)
            target = TF.center_crop(target, self.image_shape)
            parts = TF.center_crop(parts, self.image_shape)
        image = TF.normalize(image, self.mean, self.std)
        return image, target, parts
