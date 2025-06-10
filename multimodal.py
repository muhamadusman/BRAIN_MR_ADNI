# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Multi-modal WGAN-GP that jointly generates **3-D brain volumes** and their
**full clinical metadata vectors**, with an initial dump of real samples
to verify dataset loading and a fix to produce grayscale sample grids.
"""

import os
import re
import math
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch import autograd
from torchvision.utils import save_image
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

###############################################################################
# 0)  Utility: partial weight loading
###############################################################################

def load_matching_layers(model: nn.Module, ckpt_path: str, device: str = "cpu") -> None:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[load_matching_layers] Skip — no file at {ckpt_path}")
        return
    print(f"[load_matching_layers] Loading from {ckpt_path}")
    src_state = torch.load(ckpt_path, map_location=device)
    if all(k.startswith("module.") for k in src_state):
        src_state = {k[len("module."):]: v for k, v in src_state.items()}
    dst_state = model.state_dict()
    matched = {k: v for k, v in src_state.items()
               if k in dst_state and v.shape == dst_state[k].shape}
    dst_state.update(matched)
    model.load_state_dict(dst_state)
    print(f"  ↳ loaded {len(matched)}/{len(dst_state)} tensors")

###############################################################################
# 1) Dataset – 3-D volume + metadata vector
###############################################################################

class NiftiMetaDataset(Dataset):
    """Each sample returns *(volume_tensor, metadata_tensor, ptid)*."""

    def __init__(self, root_dir: str, csv_path: str, drop_first_cols: int = 2):
        super().__init__()
        self.root_dir = root_dir

        df = pd.read_csv(csv_path, delimiter=";")
        df_numeric = (
            df.drop(df.columns[:drop_first_cols], axis=1)
              .select_dtypes(include=[np.number])
              .dropna()
        )
        self.meta_cols = list(df_numeric.columns)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        meta_scaled = self.scaler.fit_transform(df_numeric.values.astype(np.float32))
        ptids = df["PTID"].astype(str).values
        self.meta_dict: dict[str, np.ndarray] = {p: m for p, m in zip(ptids, meta_scaled)}

        pattern = re.compile(r"^(.*)\.nii(?:\.gz)?$")
        self.file_tuples: List[Tuple[str, str]] = []
        for fname in os.listdir(root_dir):
            if fname.endswith((".nii", ".nii.gz")):
                m = pattern.match(fname)
                if m and m.group(1) in self.meta_dict:
                    self.file_tuples.append((os.path.join(root_dir, fname), m.group(1)))
        if not self.file_tuples:
            raise RuntimeError("No matching PTID between NIfTI files and CSV rows.")
        print(f"[Dataset] {len(self.file_tuples)} matched samples found.")

    def __len__(self) -> int:
        return len(self.file_tuples)

    def __getitem__(self, idx: int):
        vol_path, ptid = self.file_tuples[idx]
        vol = nib.load(vol_path).get_fdata(dtype=np.float32)
        if vol.max() > 0:
            vol /= vol.max()
        vol = vol * 2.0 - 1.0
        vol = torch.from_numpy(vol).unsqueeze(0)  # (1, D, H, W)
        meta = torch.from_numpy(self.meta_dict[ptid])
        return vol, meta, ptid

    def inverse_meta(self, meta_tensor: torch.Tensor) -> np.ndarray:
        return self.scaler.inverse_transform(meta_tensor.detach().cpu().numpy())

###############################################################################
# 2) Generator – outputs (volume, metadata)
###############################################################################

class MMGenerator(nn.Module):
    def __init__(self, z_dim: int, meta_dim: int,
                 D: int = 182, H: int = 218, W: int = 182, bias: bool = False):
        super().__init__()
        self.init_d = math.ceil(D / 16)
        self.init_h = math.ceil(H / 16)
        self.init_w = math.ceil(W / 16)
        fc_out = 512 * self.init_d * self.init_h * self.init_w

        self.fc_spatial = nn.Sequential(
            nn.Linear(z_dim, fc_out, bias=bias),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, 2, 1, bias=bias), nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, 4, 2, 1, bias=bias), nn.ReLU(True),
            nn.ConvTranspose3d(128,  64, 4, 2, 1, bias=bias), nn.ReLU(True),
            nn.ConvTranspose3d( 64,   1, 4, 2, 1, bias=bias), nn.Tanh(),
        )
        self.meta_head = nn.Sequential(
            nn.Linear(z_dim, 128, bias=bias), nn.ReLU(True),
            nn.Linear(128, meta_dim, bias=bias), nn.Tanh(),
        )
        self.target_D, self.target_H, self.target_W = D, H, W

    def forward(self, z: torch.Tensor):
        v = self.fc_spatial(z).view(-1, 512, self.init_d, self.init_h, self.init_w)
        v = self.deconv(v)
        _, _, d, h, w = v.shape
        sd, sh, sw = (d - self.target_D) // 2, (h - self.target_H) // 2, (w - self.target_W) // 2
        v = v[:, :, sd:sd + self.target_D, sh:sh + self.target_H, sw:sw + self.target_W]
        m = self.meta_head(z)
        return v, m

###############################################################################
# 3) Discriminator – scores joint (volume, metadata)
###############################################################################

class MMDiscriminator(nn.Module):
    def __init__(self, meta_dim: int,
                 D: int = 182, H: int = 218, W: int = 182, bias: bool = False):
        super().__init__()
        pd, ph, pw = (16 - D % 16) % 16, (16 - H % 16) % 16, (16 - W % 16) % 16
        self.pad = (0, pw, 0, ph, 0, pd)
        nd, nh, nw = (D + pd) // 16, (H + ph) // 16, (W + pw) // 16

        self.conv = nn.Sequential(
            nn.Conv3d(1,  64, 4, 2, 1, bias=bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, 2, 1, bias=bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 4, 2, 1, bias=bias), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, 4, 2, 1, bias=bias), nn.LeakyReLU(0.2, inplace=True),
        )
        vol_feat = 512 * nd * nh * nw

        self.meta_emb = nn.Sequential(
            nn.Linear(meta_dim, 128, bias=bias), nn.LeakyReLU(0.2, inplace=True)
        )
        self.final = nn.Linear(vol_feat + 128, 1, bias=bias)

    def forward(self, v: torch.Tensor, m: torch.Tensor):
        v = F.pad(v, self.pad, "constant", 0)
        vf = self.conv(v).view(v.size(0), -1)
        mf = self.meta_emb(m)
        return self.final(torch.cat([vf, mf], dim=1))

###############################################################################
# 4) WGAN-GP helper
###############################################################################

def grad_penalty(D: nn.Module, rv: torch.Tensor, rm: torch.Tensor,
                 fv: torch.Tensor, fm: torch.Tensor, device: torch.device) -> torch.Tensor:
    alpha = torch.rand(rv.size(0), 1, 1, 1, 1, device=device)
    iv = (alpha * rv + (1 - alpha) * fv).requires_grad_(True)
    im = (alpha.squeeze()[:, None] * rm + (1 - alpha.squeeze())[:, None] * fm).requires_grad_(True)
    score = D(iv, im)
    grad_vol, grad_meta = autograd.grad(score, [iv, im], torch.ones_like(score),
                                        create_graph=True, retain_graph=True)
    grad = torch.cat([grad_vol.view(grad_vol.size(0), -1), grad_meta], dim=1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp

###############################################################################
# 5) Sample-saving helper (now guaranteed single-channel grayscale)
###############################################################################

def save_sample(volume: torch.Tensor, meta_vec: torch.Tensor, meta_cols: List[str],
                out_dir: str, idx: int) -> None:
    """Store a single‐channel grid PNG of three orthogonal slices + CSV."""
    os.makedirs(out_dir, exist_ok=True)
    vol = volume.detach().cpu()  # (1, D, H, W)
    _, D, H, W = vol.shape
    size = min(D, H, W)

    # extract & center‐crop each plane
    axial    = vol[:, D//2   , (H-size)//2:(H+size)//2, (W-size)//2:(W+size)//2]
    coronal  = vol[:, (D-size)//2:(D+size)//2, H//2   , (W-size)//2:(W+size)//2]
    sagittal = vol[:, (D-size)//2:(D+size)//2, (H-size)//2:(H+size)//2, W//2   ]

    # tile them side by side into one grayscale canvas
    canvas = torch.zeros(1, size, size*3)
    canvas[:, :,   0*size:1*size] = axial
    canvas[:, :,   1*size:2*size] = coronal
    canvas[:, :,   2*size:3*size] = sagittal

    # normalize from [-1,1] → [0,1]
    img = (canvas + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)

    save_image(img, os.path.join(out_dir, f"sample_{idx:04d}.png"))

    # save metadata row
    meta_df = pd.DataFrame(meta_vec.detach().cpu().numpy()[None, :], columns=meta_cols)
    meta_df.to_csv(os.path.join(out_dir, f"sample_{idx:04d}.csv"), index=False, sep=";")

###############################################################################
# 6) Training loop
###############################################################################

def train_mm_wgan_gp(args):
    device, G, D, loader = args.device, args.G, args.D, args.loader
    optG = optim.Adam(G.parameters(), lr=args.lrG, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lrD, betas=(0.5, 0.999))

    fixed_z = torch.randn(args.n_samples, args.z_dim, device=device)
    meta_cols = loader.dataset.meta_cols
    step = 0

    for epoch in range(args.epochs):
        g_loss_val = 0.0
        for rv, rm, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            rv, rm = rv.to(device), rm.to(device)
            bs = rv.size(0)

            # Discriminator
            optD.zero_grad()
            z = torch.randn(bs, args.z_dim, device=device)
            fv, fm = G(z)
            d_real = D(rv, rm)
            d_fake = D(fv.detach(), fm.detach())
            gp = grad_penalty(D, rv, rm, fv, fm, device) * args.lambda_gp
            d_loss = -d_real.mean() + d_fake.mean() + gp
            d_loss.backward()
            optD.step()

            # Generator (every gen_every steps)
            if step % args.gen_every == 0:
                optG.zero_grad()
                z2 = torch.randn(bs, args.z_dim, device=device)
                gv, gm = G(z2)
                g_loss = -D(gv, gm).mean()
                g_loss.backward()
                optG.step()
                g_loss_val = g_loss.item()
            step += 1

        print(f"Epoch {epoch+1}: D_loss={d_loss.item():.4f} | G_loss={g_loss_val:.4f}")

        # checkpoint + fixed‐z samples
        if (epoch+1) % args.save_every == 0:
            ckpt_dir = os.path.join(args.out_dir, "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(G.state_dict(), os.path.join(ckpt_dir, f"G_epoch{epoch+1}.pth"))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, f"D_epoch{epoch+1}.pth"))

            G.eval()
            with torch.no_grad():
                sv, sm = G(fixed_z)
            for i in range(args.n_samples):
                save_sample( sv[i], sm[i], meta_cols, args.samples_dir, i + epoch * args.n_samples )
            G.train()

    print("Training complete ✔")

###############################################################################
# 7) CLI entry‐point
###############################################################################

def parse_args():
    p = argparse.ArgumentParser("Multi-modal brain volume + metadata GAN")
    p.add_argument("--root_dir",        required=True, help="Folder with NIfTI volumes")
    p.add_argument("--csv_path",        required=True, help="Semicolon-delimited clinical CSV")
    p.add_argument("--z_dim",           type=int,   default=100)
    p.add_argument("--batch",           type=int,   default=2)
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--lrG",             type=float, default=1e-4)
    p.add_argument("--lrD",             type=float, default=1e-4)
    p.add_argument("--lambda_gp",       type=float, default=10.0)
    p.add_argument("--gen_every",       type=int,   default=5, help="G update frequency (steps)")
    p.add_argument("--save_every",      type=int,   default=5, help="Save checkpoint every N epochs")
    p.add_argument("--n_samples",       type=int,   default=4, help="# fixed samples to save")
    p.add_argument("--gpu",             type=int,   default=0, help="GPU index (CUDA)")
    p.add_argument("--out_dir",         default="mmgan_logs", help="Output directory")
    p.add_argument("--num_workers",     type=int,   default=8, help="DataLoader worker count")
    p.add_argument("--pretrained_G_vol", type=str,  default="", help="Pretrained G volume ckpt")
    p.add_argument("--pretrained_D_vol", type=str,  default="", help="Pretrained D volume ckpt")
    return p.parse_args()

def main():
    args = parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # prepare data
    dataset = NiftiMetaDataset(args.root_dir, args.csv_path)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         drop_last=True, num_workers=args.num_workers)
    args.loader = loader

    # dump a few real samples up‐front
    real_dir = os.path.join(args.out_dir, "real_samples")
    os.makedirs(real_dir, exist_ok=True)
    real_vols, real_metas, _ = next(iter(loader))
    for i, (rv, rm) in enumerate(zip(real_vols, real_metas)):
        save_sample(rv, rm, dataset.meta_cols, real_dir, i)
    print(f"[INFO] Wrote {len(real_vols)} real samples to {real_dir}")

    # build & warm-start models
    meta_dim = len(dataset.meta_cols)
    print(f"[INFO] meta_dim = {meta_dim}")
    G = MMGenerator(args.z_dim, meta_dim).to(args.device)
    D = MMDiscriminator(meta_dim).to(args.device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    if args.pretrained_G_vol:
        load_matching_layers(
            G.module if isinstance(G, nn.DataParallel) else G,
            args.pretrained_G_vol, device=args.device
        )
    if args.pretrained_D_vol:
        load_matching_layers(
            D.module if isinstance(D, nn.DataParallel) else D,
            args.pretrained_D_vol, device=args.device
        )

    os.makedirs(args.out_dir, exist_ok=True)
    args.samples_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(args.samples_dir, exist_ok=True)

    # train!
    args.G, args.D = G, D
    train_mm_wgan_gp(args)

if __name__ == "__main__":
    main()


# python main.py   \
#   --root_dir   /proj/synthetic_alzheimer/users/x_muhak/WGAN-GP_MultiModal_AGE/Data/ADNI_NIFTY_Skullstripped_MNI_template  \
#   --csv_path   /proj/synthetic_alzheimer/users/x_muhak/WGAN-GP_MultiModal_MetaData/metaData.csv \
#   --epochs     500  \
#   --gpu        0   \
#   --pretrained_G_vol  /proj/synthetic_alzheimer/users/x_muhak/WGAN-GP_MultiModal_MetaData/mmgan_logs/ckpt/G_epoch500_1.pth  \
#   --pretrained_D_vol  /proj/synthetic_alzheimer/users/x_muhak/WGAN-GP_MultiModal_MetaData/mmgan_logs/ckpt/D_epoch500_1.pth   \
#   --batch 32 --num_workers 16

