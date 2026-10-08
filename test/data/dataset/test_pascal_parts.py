import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data.dataset.pascal_parts import PascalPartsDataset


@pytest.fixture
def voc_root(tmp_path):
    root = tmp_path / "VOCdevkit/VOC2012"
    for folder in ("JPEGImages", "SegmentationClass", "ImageSets/Segmentation", "labels/val"):
        (root / folder).mkdir(parents=True)
    (root / "ImageSets/Segmentation/val.txt").write_text("with_parts\nmissing_parts\n")
    for name in ("with_parts", "missing_parts"):
        Image.fromarray(np.full((2, 5, 3), 128, dtype=np.uint8)).save(root / "JPEGImages" / f"{name}.jpg")
        target = np.array([[7, 7, 7, 7, 7], [7, 7, 7, 255, 0]], dtype=np.uint8)
        Image.fromarray(target).save(root / "SegmentationClass" / f"{name}.png")
    # Two countable wheels fold to the same part. Missing parts, unsupported
    # parts, wrong semantic class, void and background must be handled separately.
    uids = np.array([[700120, 700229, 700101, 700199, 700100], [7001, 7, 600120, 700120, 700120]], dtype=np.int32)
    Image.fromarray(uids).save(root / "labels/val/with_parts.tif")
    # Preserve VOC's class ordering, as the real PPP specification does.
    (root / "parts.yaml").write_text(
        "scene_class2part_classes:\n"
        "  aeroplane: []\n  bicycle: []\n  bird: []\n  boat: []\n"
        "  bottle: []\n  bus: []\n  car: [body, wheel]\n"
        "  cat: []\n  chair: []\n  cow: []\n  table: []\n  dog: []\n  horse: []\n"
        "  motorbike: []\n  person: []\n  pottedplant: []\n  sheep: []\n  sofa: []\n"
        "  train: []\n  tvmonitor: []\n"
        "countable_pids_groupings:\n  car:\n    wheel: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]\n"
    )
    return root


@pytest.mark.parametrize("root_level", [0, 1, 2])
@pytest.mark.parametrize("official_only", [False, True])
def test_voc_pairing_and_uid_folding(voc_root, root_level, official_only):
    root = voc_root if root_level == 0 else voc_root.parents[root_level - 1]
    dataset = PascalPartsDataset(root, official_parts_only=official_only)
    assert len(dataset) == 1
    assert dataset.num_missing_parts == 1
    assert dataset.image_ids == ["with_parts"]
    image, semantic, parts = dataset[0]
    assert image.shape == (3, 2, 5)
    assert image.dtype == torch.float32
    assert semantic.dtype == parts.dtype == torch.int64
    assert semantic[0, 1, 3] == 0
    assert parts.tolist() == [[[702, 702, 701, 0 if official_only else 799, 0], [0, 0, 0, 0, 0]]]


def test_pascal_center_crop_applies_to_all_three_tensors(voc_root):
    original = PascalPartsDataset(voc_root)[0]
    cropped = PascalPartsDataset(voc_root, image_shape=(2, 3))[0]
    for full, crop in zip(original, cropped, strict=True):
        assert torch.equal(full[..., 1:4], crop)


def test_pascal_missing_annotations_fail_clearly(voc_root):
    empty = voc_root / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No VOC validation images"):
        PascalPartsDataset(voc_root, parts_path=empty)
    with pytest.raises(FileNotFoundError, match="specification"):
        PascalPartsDataset(voc_root, parts_spec=Path("missing.yaml"))


def test_pascal_rejects_wrong_class_order(voc_root):
    spec = voc_root / "parts.yaml"
    spec.write_text(spec.read_text().replace("aeroplane: []", "wrong_class: []"))
    with pytest.raises(ValueError, match="VOC ID order"):
        PascalPartsDataset(voc_root)


@pytest.mark.parametrize("official_only", [False, True])
def test_pascal_cli_writes_scores_and_image_manifest(voc_root, tmp_path, monkeypatch, official_only):
    from test.proto_segmentation.test_consistency import DummyPPNet
    from visualize import consistency

    model = DummyPPNet()
    model.prototype_class_identity = torch.zeros(2, 21)
    model.prototype_class_identity[0, 7] = 1
    monkeypatch.setattr(consistency, "_load_supported_model", lambda _: model)
    output = tmp_path / "results"
    argv = [
        "consistency",
        "pascal.pth",
        "--dataset",
        "pascal_voc",
        "--data-path",
        str(voc_root),
        "--output-dir",
        str(output),
        "--device",
        "cpu",
    ]
    if official_only:
        argv.append("--official-parts-only")
    monkeypatch.setattr(sys, "argv", argv)
    consistency.main()

    if official_only:
        output = output / "official_parts"
    summary = json.loads((output / "consistency_summary_th_0.8_qt_0.8.json").read_text())
    assert summary["num_evaluated_prototypes"] == 1
    assert summary["prototype_consistency"][0]["class_id"] == 7
    assert (output / "image_ids.txt").read_text() == "with_parts\n"
    metadata = json.loads((output / "pascal_evaluation.json").read_text())
    assert metadata["num_images"] == metadata["num_missing_parts"] == 1
    assert metadata["semantic_class_offset"] == 0
    assert metadata["official_parts_only"] is official_only
