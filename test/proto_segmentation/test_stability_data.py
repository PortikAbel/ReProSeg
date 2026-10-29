import numpy as np
import pytest
import torch
from PIL import Image

from config import EvaluateConfig
from config.schema.data import DatasetType
from evaluate.stability import create_stability_dataloader
from test.data.dataset import test_pascal_parts as pascal_test_data
from test.proto_segmentation.test_consistency import DummyPPNet


@pytest.fixture
def stability_voc_root(tmp_path):
    return pascal_test_data.voc_root.__wrapped__(tmp_path)


@pytest.fixture
def cityscapes_config(tmp_path):
    for directory in ("leftImg8bit", "gtFine", "gtFinePanopticParts"):
        (tmp_path / directory / "val" / "city").mkdir(parents=True)
    semantics = np.full((1024, 2048), 24, dtype=np.uint8)
    parts = np.zeros((1024, 2048), dtype=np.int32)
    parts[10, 20:23] = [2_400_101, 2_400_105, 24_001]
    for name in ("city_000002_000001", "city_000001_000001"):
        Image.new("RGB", (2048, 1024), (64, 128, 192)).save(
            tmp_path / "leftImg8bit/val/city" / f"{name}_leftImg8bit.png"
        )
        Image.fromarray(semantics).save(tmp_path / "gtFine/val/city" / f"{name}_gtFine_labelIds.png")
        Image.fromarray(parts).save(tmp_path / "gtFinePanopticParts/val/city" / f"{name}_gtFinePanopticParts.tif")

    cfg = EvaluateConfig()
    cfg.data.path = tmp_path
    cfg.data.img_shape = (16, 32)  # Training dimensions must not crop evaluation.
    cfg.data.validation_size = 0.2  # Use the official val split, not a train split.
    cfg.data.mean = (0.485, 0.456, 0.406)
    cfg.data.std = (0.229, 0.224, 0.225)
    cfg.env.device = torch.device("cpu")
    return cfg


def test_real_cityscapes_loading_preserves_normalization_alignment_and_part_modes(cityscapes_config):
    cfg = cityscapes_config
    all_loader = create_stability_dataloader(cfg)
    assert all_loader.batch_size == 1
    assert len(all_loader.dataset) == 2
    paths = [all_loader.dataset._get_image_path(i) for i in range(2)]
    assert paths == sorted(paths)
    images, semantics, parts = next(iter(all_loader))
    assert images.shape == (1, 3, 1024, 2048)
    assert semantics.shape == parts.shape == (1, 1, 1024, 2048)
    expected = (torch.tensor([64, 128, 192]) / 255 - torch.tensor(cfg.data.mean)) / torch.tensor(cfg.data.std)
    torch.testing.assert_close(images[0, :, 10, 20], expected)
    assert semantics[0, 0, 10, 20:23].tolist() == [12, 12, 12]
    assert parts[0, 0, 10, 20:23].tolist() == [2401, 2405, 0]

    cfg.evaluate.stability.official_parts_only = True
    official_loader = create_stability_dataloader(cfg)
    official_images, official_semantics, official_parts = next(iter(official_loader))
    assert [official_loader.dataset._get_image_path(i) for i in range(2)] == paths
    assert torch.equal(images, official_images)
    assert torch.equal(semantics, official_semantics)
    assert official_parts[0, 0, 10, 20:23].tolist() == [2401, 0, 0]
    assert cfg.data.img_shape == (16, 32)


def test_loader_honors_batch_size_keeps_final_batch_and_does_not_consume_global_rng(cityscapes_config):
    cfg = cityscapes_config
    cfg.evaluate.stability.batch_size = 3
    state = torch.random.get_rng_state()
    loader = create_stability_dataloader(cfg)
    assert loader.batch_size == 3
    assert loader.num_workers == 0
    assert loader.drop_last is False
    assert len(loader) == 1
    iterator = iter(loader)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert next(iterator)[0].shape[0] == 2


def test_stability_loader_reports_missing_pascal_data():
    cfg = EvaluateConfig()
    cfg.data.dataset = DatasetType.VOC_SEGMENTATION
    with pytest.raises(FileNotFoundError, match="Cannot find VOC2012"):
        create_stability_dataloader(cfg)


def test_missing_part_annotations_raise_instead_of_silently_skipping(cityscapes_config):
    cfg = cityscapes_config
    part_path = cfg.data.path / "gtFinePanopticParts/val/city/city_000001_000001_gtFinePanopticParts.tif"
    part_path.unlink()
    with pytest.raises(FileNotFoundError):
        next(iter(create_stability_dataloader(cfg)))


@pytest.mark.parametrize("num_classes", [20, 21])
@pytest.mark.parametrize("official_parts_only", [False, True])
def test_pascal_stability_uses_consistency_annotations_and_saves_manifest(
    stability_voc_root, temp_dir, num_classes, official_parts_only
):
    import json

    from evaluate.registry import run

    model = DummyPPNet()
    model.prototype_class_identity = torch.zeros(2, num_classes)
    model.prototype_class_identity[0, 7 - (21 - num_classes)] = 1
    model.last_layer = torch.nn.Linear(2, num_classes, bias=False)
    checkpoint = temp_dir / "pascal_checkpoint.pth"
    torch.save(model, checkpoint)

    cfg = EvaluateConfig()
    cfg.data.dataset = DatasetType.VOC_SEGMENTATION
    cfg.data.path = stability_voc_root
    cfg.env.device = torch.device("cpu")
    cfg.model.checkpoint = checkpoint
    cfg.evaluate.metric_name = "stability"
    cfg.evaluate.stability.std_dev = 0
    cfg.evaluate.stability.official_parts_only = official_parts_only
    result = run(cfg)

    assert result.num_images == 1
    assert result.summary.num_evaluated_pairs == 1
    assert result.summary.score == 1
    assert result.comparisons[0].class_id == 7
    assert result.comparisons[0].part_ids == ((1, 2) if official_parts_only else (1, 2, 99))
    mode = "official_parts" if official_parts_only else "all_parts"
    output = temp_dir / f"stability_protoseg_pascalvoc_{mode}"
    assert (output / "image_ids.txt").read_text() == "with_parts\n"
    metadata = json.loads((output / "pascal_evaluation.json").read_text())
    assert metadata["semantic_class_offset"] == 21 - num_classes
    assert metadata["num_missing_parts"] == 1
    assert (output / "stability_score_qt_0.8_std_0_seed_0.txt").read_text() == "1.0\n"


def test_pascal_native_images_require_batch_size_one(stability_voc_root):
    cfg = EvaluateConfig()
    cfg.data.dataset = DatasetType.VOC_SEGMENTATION
    cfg.data.path = stability_voc_root
    cfg.evaluate.stability.batch_size = 2
    with pytest.raises(ValueError, match="vary in size"):
        create_stability_dataloader(cfg)


@pytest.mark.parametrize("official_parts_only,expected_parts", [(False, [1, 5]), (True, [1])])
def test_reproseg_cli_uses_actual_part_filtering_and_saves_subset(
    cityscapes_config, temp_dir, official_parts_only, expected_parts
):
    import csv
    import json

    from evaluate.registry import run
    from test.proto_segmentation.test_stability_reproseg import MultiScaleReProSeg

    cfg = cityscapes_config
    model = MultiScaleReProSeg(num_concepts=1, num_classes=20)
    model.layers.classification_layer.weight.data[12, 0] = 1
    cfg.model.checkpoint = temp_dir / "reproseg.pth"
    torch.save(model, cfg.model.checkpoint)
    cfg.evaluate.metric_name = "stability"
    cfg.evaluate.stability.official_parts_only = official_parts_only
    cfg.evaluate.stability.std_dev = 0
    cfg.evaluate.stability.max_images = 1
    result = run(cfg)
    assert result.num_images == 1
    assert result.summary.score == 1
    assert result.summary.num_evaluated_pairs == 1
    assert list(result.comparisons[0].part_ids) == expected_parts
    mode = "official_parts" if official_parts_only else "all_parts"
    output = temp_dir / f"stability_reproseg_cityscapes_{mode}" / "first_1_images"
    summary = json.loads((output / "stability_summary_qt_0.8_std_0_seed_0.json").read_text())
    assert summary["num_images"] == 1
    assert summary["score"] == 1
    with (output / "part_presence_qt_0.8_std_0_seed_0.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert json.loads(rows[0]["part_ids"]) == expected_parts
