#!/usr/bin/env python3
# echonet_ef_model.py
# -------------------------------------------------------------
#   End-to-end EF regression on EchoNet-Dynamic
# -------------------------------------------------------------
from __future__ import annotations
import math, random, os, argparse, json, csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.io import read_video
from torchvision.models.video import r2plus1d_18


# ──────────────────────────────────────────────────────────────
#                         DATASET
# ──────────────────────────────────────────────────────────────
class EchoVideo(Dataset):
    """
    A minimal VideoDataset for EchoNet-Dynamic.
    •  Expects .avi files in <root>/videos/
    •  FileList.csv must be in <root>/
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "TRAIN",
        frames: int = 32,
        cache: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.root = Path(root)
        self.frames = frames
        self.cache = cache
        self.rng = random.Random(seed)

        df = pd.read_csv(self.root / "FileList.csv")
        df = df[df["Split"] == split.upper()].reset_index(drop=True)
        self.meta = df

        self.transform = Compose(
            [
                Resize(128),  # make it square-ish
                CenterCrop(112),
                # torchvideo uses C x T x H x W
                # we'll rescale to [0,1] then normalize below
            ]
        )
        self.normalize = Normalize(mean=[0.45], std=[0.225])

        self._cache = {}  # optional mem cache

    def __len__(self):
        return len(self.meta)

    def _read_video_cv2(self, f: Path):
        cap = cv2.VideoCapture(str(f))
        frames = []
        ok, frame = cap.read()
        while ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # (H,W)
            frames.append(frame)
            ok, frame = cap.read()
        cap.release()
        return np.stack(frames, axis=0)  # T x H x W

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        vidname = row["FileName"] + ".avi"
        label = row["EF"] / 100.0  # scale to 0-1 for regression stability

        # lazy disk -> numpy
        if self.cache and vidname in self._cache:
            video = self._cache[vidname]
        else:
            video_np = self._read_video_cv2(self.root / "videos" / vidname)
            if self.cache:
                self._cache[vidname] = video_np
            video = video_np

        total = video.shape[0]
        if self.training:
            start = self.rng.randint(0, max(0, total - self.frames))
        else:
            start = max(0, (total - self.frames) // 2)
        end = start + self.frames
        clip = video[start:end]

        if clip.shape[0] < self.frames:
            pad = np.repeat(clip[-1:], self.frames - clip.shape[0], axis=0)
            clip = np.concatenate([clip, pad], axis=0)

        clip = clip[..., None] / 255.0
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2)
        clip = self.transform(clip)
        clip = self.normalize(clip)
        return clip.float(), torch.tensor([label], dtype=torch.float32)

    @property
    def training(self):
        return getattr(self, "_training", False)

    @training.setter
    def training(self, v: bool):
        self._training = v


# ──────────────────────────────────────────────────────────────
#                         MODEL
# ──────────────────────────────────────────────────────────────
class EFRegressor(nn.Module):
    """
    R(2+1)D backbone + 1 neuron head
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = r2plus1d_18(weights="KINETICS400_V1" if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1, 1)
        return torch.sigmoid(self.backbone(x))


# ──────────────────────────────────────────────────────────────
#                      TRAIN / VALID LOOP
# ──────────────────────────────────────────────────────────────
def step(
    net: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer | None,
    device: torch.device,
):
    running = {"loss": 0, "n": 0, "sdiff": 0}
    mse = nn.MSELoss()

    for xb, yb in tqdm(loader, leave=False):
        xb, yb = xb.to(device), yb.to(device)
        pred = net(xb)
        loss = mse(pred, yb)

        if optim:
            optim.zero_grad()
            loss.backward()
            optim.step()

        running["loss"] += loss.item() * len(xb)
        running["sdiff"] += torch.sum(torch.abs(pred - yb)).item()
        running["n"] += len(xb)

    mae = running["sdiff"] / running["n"] * 100  # back to %
    rmse = math.sqrt(running["loss"] / running["n"]) * 100
    return mae, rmse


# ──────────────────────────────────────────────────────────────
#                           MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train EF regressor on EchoNet")
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--out", type=str, default="ef_model.pt")
    parser.add_argument("--train_frac", type=float, default=1.0)
    parser.add_argument("--val_frac",   type=float, default=1.0)
    parser.add_argument("--test_frac",  type=float, default=1.0)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print("Device:", device)

    def make_loader(split, frac, shuffle):
       base_ds = EchoVideo(args.root, split, args.frames, cache=False)
       if frac >= 1.0:
           sampler = None
       else:
           rng = random.Random(args.seed)
           n = len(base_ds)
           k = max(1, int(n * frac))
           indices = rng.sample(range(n), k)
           sampler = torch.utils.data.SubsetRandomSampler(indices)
       return base_ds, DataLoader(
           base_ds,
           batch_size=args.batch * (2 if not shuffle else 1),
           shuffle=(sampler is None and shuffle),
           sampler=sampler,
           num_workers=4,
       )
    ds_train, dl_train = make_loader("TRAIN", args.train_frac, shuffle=True)
    ds_val,   dl_val   = make_loader("VAL",   args.val_frac,   shuffle=False)
    ds_test,  dl_test  = make_loader("TEST",  args.test_frac,  shuffle=False)


    net = EFRegressor(pretrained=True).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # ───────────── resume if checkpoint exists ─────────────
    best_rmse = float("inf")
    ckpt_path = Path(args.out)
    if ckpt_path.is_file():
        print(f"🔄  Found checkpoint {ckpt_path}, loading…")
        net.load_state_dict(torch.load(ckpt_path, map_location=device))
        ds_train.training = False
        val_mae, val_rmse = step(net, dl_val, None, device)
        best_rmse = val_rmse
        print(f"   ➜  Current Val RMSE {val_rmse:.2f}%  (will improve on this)")
        
    for ep in range(1, args.epochs + 1):
        print(f"\nEpoch {ep}/{args.epochs}")
        ds_train.training = True
        train_mae, train_rmse = step(net, dl_train, opt, device)
        ds_train.training = False
        val_mae, val_rmse = step(net, dl_val, None, device)

        print(
            f" Train MAE {train_mae:.2f}%  RMSE {train_rmse:.2f}% |"
            f" Val MAE {val_mae:.2f}%  RMSE {val_rmse:.2f}%"
        )
        scheduler.step()

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer":  opt.state_dict(),
                    "scheduler":  scheduler.state_dict(),
                    "best_rmse":  best_rmse,
                    "epoch":      ep,
                },
                args.out,
            )

            print("  ↳ saved:", args.out)

    # evaluate on test set
    print("\nLoading best model for test…")
    net.load_state_dict(torch.load(args.out, map_location=device))
    test_mae, test_rmse = step(net, dl_test, None, device)
    print(f"Test MAE {test_mae:.2f}%  RMSE {test_rmse:.2f}%")

    print("Done.")


# ──────────────────────────────────────────────────────────────
#                   SIMPLE INFERENCE HELPER
# ──────────────────────────────────────────────────────────────
@torch.inference_mode()
def infer(model_path: str, video_path: str, frames: int = 32):
    """
    Usage:
        >>> from echonet_ef_model import infer
        >>> ef = infer('ef_model.pt', 'some_echo.avi')
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    net = EFRegressor(pretrained=False).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    cap = cv2.VideoCapture(video_path)
    vid = []
    ok, fr = cap.read()
    while ok:
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        vid.append(fr)
        ok, fr = cap.read()
    cap.release()
    vid = np.stack(vid, 0)

    start = max(0, (vid.shape[0] - frames) // 2)
    clip = vid[start : start + frames]
    if clip.shape[0] < frames:
        clip = np.concatenate(
            [clip, np.repeat(clip[-1:], frames - clip.shape[0], 0)], 0
        )

    clip = clip[..., None] / 255.0
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2)  # 1xTxHxW
    clip = Resize(128)(clip)
    clip = CenterCrop(112)(clip)
    clip = Normalize([0.45], [0.225])(clip)
    clip = clip.unsqueeze(0).float().to(device)  # add batch

    ef = net(clip).item() * 100.0
    return ef


if __name__ == "__main__":
    main()