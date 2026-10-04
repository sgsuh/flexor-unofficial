import json
import os

import pytest

import train
from flexor import XORSpec

ROOT = os.environ.get("FLEXOR_DATA_ROOT", "/data/torchvision")
CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")


def test_overrides_and_defaults():
    cfg = train.load_config(None, ["flexor.n_in=12", "schedule.milestones=[1, 2]", "flexor.n_tap=null", "name=x"])
    assert cfg["flexor"]["n_in"] == 12 and cfg["flexor"]["n_tap"] is None
    assert cfg["schedule"]["milestones"] == [1, 2]
    assert cfg["flexor"]["n_out"] == 20  # untouched default
    assert cfg["name"] == "x"
    assert train.load_config(None, ["flexor=null"])["flexor"] is None


@pytest.mark.parametrize("name", sorted(os.listdir(CONFIGS)))
def test_all_configs_build(name):
    cfg = train.load_config(os.path.join(CONFIGS, name), [])
    assert cfg["name"] == name[: -len(".yaml")]
    train.build_from_config(cfg)


def test_build_spec_stages():
    specs = train.build_spec({"n_in": 12, "n_out": 20, "q": 1, "n_tap": 2, "seed": 0,
                              "stages": [{"n_in": 19}, {"n_in": 16}, {"n_in": 7}]})
    assert [s.n_in for s in specs] == [19, 16, 7] and all(s.n_out == 20 for s in specs)
    assert train.build_spec(None) is None
    assert train.build_spec({"n_in": 8, "n_out": 10, "q": 2, "n_tap": 2, "seed": 0}) == XORSpec(8, 10, 2)


def test_lr_factor():
    f = lambda e: train.lr_factor(e, 100, [350, 400, 450], 0.5)
    assert f(0) == 0 and f(50) == pytest.approx(0.5) and f(100) == 1.0
    assert f(349.9) == 1.0 and f(350) == 0.5 and f(420) == 0.25 and f(499) == 0.125
    assert train.lr_factor(3, 0, [], 0.5) == 1.0


@pytest.mark.skipif(not os.path.isdir(ROOT), reason=f"{ROOT} not found")
def test_smoke_train_and_resume(tmp_path):
    overrides = [f"out_dir={tmp_path}", "name=smoke", "debug.limit_batches=3", "schedule.epochs=2",
                 "schedule.warmup_epochs=1", "schedule.milestones=[1]", "data.num_workers=0"]
    cfg = train.load_config(os.path.join(CONFIGS, "mnist_lenet5.yaml"), overrides)
    summary = train.train(cfg)
    run = tmp_path / "smoke"
    for f in ("config.yaml", "metrics.csv", "last.pt", "exported.pt", "summary.json"):
        assert (run / f).exists()
    assert summary["export_max_diff"] < 1e-5
    assert summary["bits_per_weight"] == pytest.approx(0.4, abs=0.01)
    assert len((run / "metrics.csv").read_text().strip().splitlines()) == 3  # header + 2 epochs

    with pytest.raises(FileExistsError):
        train.train(cfg)
    cfg["schedule"]["epochs"] = 3
    train.train(cfg, resume=True)
    assert len((run / "metrics.csv").read_text().strip().splitlines()) == 4
    assert json.loads((run / "summary.json").read_text())["name"] == "smoke"


@pytest.mark.parametrize("extra", [["flexor.xor_mode=analog"], ["flexor.xor_mode=ste", "flexor.clip=2.0"]])
def test_smoke_train_xor_modes_and_clip(tmp_path, extra):
    overrides = [f"out_dir={tmp_path}", "name=mode", "debug.limit_batches=3", "schedule.epochs=1",
                 "data.num_workers=0"] + extra
    cfg = train.load_config(os.path.join(CONFIGS, "mnist_lenet5.yaml"), overrides)
    summary = train.train(cfg)
    assert summary["export_max_diff"] < 1e-5
