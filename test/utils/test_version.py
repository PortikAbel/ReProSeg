import pytest

from utils.version import check_checkpoint_version, compat_version, major_minor


def test_major_minor_strips_patch():
    assert major_minor("0.2.5") == "0.2"
    assert major_minor("1.4") == "1.4"


def test_compat_version_ignores_patch(monkeypatch):
    monkeypatch.setattr("utils.version.get_version", lambda: "0.2.7")
    assert compat_version() == "0.2"


@pytest.fixture
def code_version(monkeypatch):
    monkeypatch.setattr("utils.version.get_version", lambda: "0.2.0")


def test_same_version_passes(code_version):
    check_checkpoint_version({"version": "0.2"}, "ckpt.pt")


def test_different_minor_raises(code_version):
    with pytest.raises(ValueError, match=r"0\.1.*0\.2"):
        check_checkpoint_version({"version": "0.1"}, "ckpt.pt")


def test_missing_version_raises(code_version):
    with pytest.raises(ValueError, match="no version"):
        check_checkpoint_version({}, "ckpt.pt")


def test_patch_in_saved_version_is_compatible(code_version):
    check_checkpoint_version({"version": "0.2.5"}, "ckpt.pt")
