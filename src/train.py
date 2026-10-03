"""One script, many configs: an experiment is a YAML file, not new code.

    python -m src.train --config configs/grading/g2_oversample.yaml
    python -m src.train --config configs/segmentation/s3_bce_dice.yaml --seed 43
    python -m src.train --config ... --set train.epochs=2 train.batch_size=4     # quick overrides
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import GradingDataset, SegmentationDataset
from src.evaluate import run_eval
from src.imbalance import get_loss, get_sampler, minority_classes
from src.models import build_model
from src.utils import MAIN_METRIC, Logger, get_device, load_config, save_config, seed_everything


def build_data(cfg, log=print):
    """Datasets + train sampler. Samplers / strong augmentation touch the training split only."""
    d, imb, seed = cfg["data"], cfg["imbalance"], cfg["train"]["seed"]
    if cfg["task"] == "grading":
        labels = pd.read_csv(d["train_csv"])["grade"].to_numpy()
        strong = minority_classes(labels) if imb.get("strong_aug_minority") else ()
        train_ds = GradingDataset(d["train_csv"], d["image_size"], train=True, strong_classes=strong)
        val_ds = GradingDataset(d["val_csv"], d["image_size"])
        sampler = get_sampler(imb.get("sampler", "none"), train_ds.labels, seed)
        if strong:
            log(f"strong augmentation for minority grades {strong}")
    else:
        train_ds = SegmentationDataset(d["train_csv"], train=True, crop_size=d["crop_size"],
                                       lesion_prob=imb.get("lesion_crop_prob", 0.0))
        val_ds = SegmentationDataset(d["val_csv"])
        sampler = None
    return train_ds, val_ds, sampler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, help="override the seed (run folder gets a _s<seed> suffix)")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="override config values")
    ap.add_argument("--force", action="store_true", help="re-train even if the run is already finished")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set, args.seed)
    t = cfg["train"]
    run_dir = Path(t["out_dir"]) / cfg["run_name"]
    if (run_dir / "val_metrics.json").exists() and not args.force:
        print(f"[skip] {run_dir} is already finished (use --force to redo)")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")  # exact config of this run, so any table row can be rerun
    log = Logger(run_dir / "train.log")

    seed_everything(t["seed"])
    device = get_device()
    use_amp = bool(t.get("amp")) and device.type == "cuda"
    main_name = MAIN_METRIC[cfg["task"]]
    log(f"run={cfg['run_name']} task={cfg['task']} device={device} seed={t['seed']} main_metric={main_name}")

    train_ds, val_ds, sampler = build_data(cfg, log)
    n_train = len(sampler) if sampler is not None else len(train_ds)
    gen = torch.Generator().manual_seed(t["seed"])
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], sampler=sampler, shuffle=sampler is None,
                              num_workers=t["num_workers"], pin_memory=device.type == "cuda",
                              drop_last=n_train > t["batch_size"], generator=gen)
    val_loader = DataLoader(val_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                            num_workers=t["num_workers"])

    model = build_model(cfg).to(device)
    loss_fn = get_loss(cfg, train_ds, log).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log(f"train images/epoch={n_train} val images={len(val_ds)} imbalance={cfg['imbalance']}")

    history, best, best_epoch, bad_epochs = [], None, 0, 0
    for epoch in range(1, t["epochs"] + 1):
        start = time.time()
        model.train()
        total, seen = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{t['epochs']}", leave=False):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
            loss = loss_fn(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total, seen = total + loss.item() * len(x), seen + len(x)
        sched.step()

        metrics = run_eval(model, val_loader, cfg, device)  # validation only; test is never touched here
        score = metrics[main_name]
        score = score if score == score else -1.0  # NaN -> never "best"
        history.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": total / max(seen, 1),
                        **{k: v for k, v in metrics.items() if not isinstance(v, list)}})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        log(f"epoch {epoch:3d}  loss {total / max(seen, 1):.4f}  val {main_name} {score:.4f}  ({time.time() - start:.0f}s)")

        if best is None or score > best[main_name]:
            best, best_epoch, bad_epochs = {**metrics, main_name: score}, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, main_name: score}, run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= t["patience"]:
                log(f"early stop: no val improvement for {t['patience']} epochs")
                break

    out = {"main_metric": main_name, "best_epoch": best_epoch, **best}
    (run_dir / "val_metrics.json").write_text(json.dumps(out, indent=2))
    log(f"done. best val {main_name} = {best[main_name]:.4f} at epoch {best_epoch} -> {run_dir}")


if __name__ == "__main__":
    main()
