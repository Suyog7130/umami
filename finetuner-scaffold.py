
import os
import json
import math
import time
import random
import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Dict, Tuple, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_string() -> str:
    today = datetime.date.today().strftime("%Y%m%d")
    clock = datetime.datetime.now().strftime("%H%M%S")
    return f"{today}-{clock}"


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Config
# ============================================================

@dataclass
class FinetunerTrainConfig:
    train_path: str
    valid_path: str
    outdir: str = "stage3_polarization_calibrator_runs"

    input_key: str = "inputs"
    target_key: str = "targets"

    num_epochs: int = 100
    batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: Optional[float] = 1.0

    use_amp: bool = True
    amp_dtype: str = "float16"

    validate_every: int = 1
    plot_every: int = 5
    checkpoint_every: int = 5
    hard_refresh_every: int = 5

    hard_start_frac: float = 0.70
    hard_top_frac: float = 0.15
    hard_sample_weight: float = 8.0

    w_min: float = 0.05
    gamma: float = 1.0
    topk_frac: float = 0.10
    smooth_kind: str = "curvature"

    eps: float = 1e-12
    num_plot_samples: int = 4

    max_train_samples: Optional[int] = None
    max_valid_samples: Optional[int] = None

    seed: int = 1234
    device: Optional[str] = None


# ============================================================
# Dataset
# ============================================================

class FinetunerDataset(Dataset):
    """
    HDF5 dataset for stage-3 hplus/hcross residual calibration.

    Expected:
        inputs:  (N, 6, n)
        targets: (N, 2, n)

    Input channels:
        0: h_ml_plus
        1: h_ml_cross
        2: m1 repeated over time
        3: m2 repeated over time
        4: chi1 repeated over time
        5: chi2 repeated over time

    Target channels:
        0: normalized residual plus
        1: normalized residual cross

    Target normalization:
        r_norm = (h_true - h_ml) / A_ml_max
    """

    def __init__(
        self,
        hdf_path: str,
        input_key: str = "inputs",
        target_key: str = "targets",
        max_samples: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.hdf_path = hdf_path
        self.input_key = input_key
        self.target_key = target_key
        self.max_samples = max_samples
        self.dtype = dtype

        self._file = None
        self._inputs = None
        self._targets = None

        with h5py.File(self.hdf_path, "r") as f:
            n = f[self.input_key].shape[0]
            self.input_shape = tuple(f[self.input_key].shape)
            self.target_shape = tuple(f[self.target_key].shape)

        if max_samples is not None:
            n = min(n, max_samples)

        self.length = n

    def _open_if_needed(self):
        if self._file is None:
            self._file = h5py.File(self.hdf_path, "r")
            self._inputs = self._file[self.input_key]
            self._targets = self._file[self.target_key]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._open_if_needed()

        x = self._inputs[idx]
        y = self._targets[idx]

        x = torch.as_tensor(x, dtype=self.dtype)
        y = torch.as_tensor(y, dtype=self.dtype)

        return {
            "input": x,
            "target_norm_residual": y,
            "index": torch.tensor(idx, dtype=torch.long),
        }

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._inputs = None
            self._targets = None


# ============================================================
# Custom FinetuneDataLoader
# ============================================================

class FinetunerDataLoader:
    """
    Normal training:
        shuffled uniform batches

    Hard curriculum:
        hard samples get larger sampling weight
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.drop_last = drop_last

        self.hard_indices = None
        self.hard_weight = 1.0

    def set_hard_indices(
        self,
        hard_indices: Optional[Sequence[int]],
        hard_weight: float = 8.0,
    ) -> None:
        if hard_indices is None or len(hard_indices) == 0:
            self.hard_indices = None
            self.hard_weight = 1.0
        else:
            self.hard_indices = np.asarray(hard_indices, dtype=np.int64)
            self.hard_weight = float(hard_weight)

    def build(self) -> DataLoader:
        kwargs = dict(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
        )

        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor

        if self.hard_indices is None:
            return DataLoader(
                **kwargs,
                shuffle=True,
            )

        weights = torch.ones(len(self.dataset), dtype=torch.double)
        valid_hard = self.hard_indices[
            (self.hard_indices >= 0) & (self.hard_indices < len(self.dataset))
        ]
        weights[valid_hard] = self.hard_weight

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.dataset),
            replacement=True,
        )

        return DataLoader(
            **kwargs,
            sampler=sampler,
            shuffle=False,
        )


# ============================================================
# Waveform helpers
# ============================================================

def split_stage3_input(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    h_ml = x[:, 0:2, :]
    params = x[:, 2:, :]
    return h_ml, params


def compute_peak_amplitude(h_ml: torch.Tensor) -> torch.Tensor:
    amp = torch.sqrt(h_ml[:, 0, :] ** 2 + h_ml[:, 1, :] ** 2)
    amax = amp.amax(dim=-1, keepdim=True).unsqueeze(1)
    return amax


def reconstruct_true_from_normalized_residual(
    h_ml: torch.Tensor,
    target_norm_residual: torch.Tensor,
) -> torch.Tensor:
    amax = compute_peak_amplitude(h_ml)
    return h_ml + amax * target_norm_residual


def apply_predicted_normalized_residual(
    h_ml: torch.Tensor,
    pred_norm_residual: torch.Tensor,
) -> torch.Tensor:
    amax = compute_peak_amplitude(h_ml)
    return h_ml + amax * pred_norm_residual


# ============================================================
# Losses
# ============================================================

def amplitude_weights(
    h_ref: torch.Tensor,
    w_min: float = 0.05,
    gamma: float = 1.0,
) -> torch.Tensor:
    amp = torch.sqrt(h_ref[:, 0, :] ** 2 + h_ref[:, 1, :] ** 2)
    amp_norm = amp / (amp.amax(dim=-1, keepdim=True))
    w = w_min + (1.0 - w_min) * amp_norm.pow(gamma)
    return w.unsqueeze(1)


def weighted_normalized_residual_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    h_ref: torch.Tensor,
    w_min: float = 0.05,
    gamma: float = 1.0,
    eps: float = 1e-12
) -> torch.Tensor:
    w = amplitude_weights(h_ref, w_min=w_min, gamma=gamma)
    loss = w * (pred_norm - target_norm) ** 2
    denom = w.sum() * pred_norm.shape[1]
    return loss.sum() / (denom + eps)


def overlap_and_mismatch(
    h_true: torch.Tensor,
    h_pred: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h_true = h_true.float()
    h_pred = h_pred.float()

    true_c = torch.complex(h_true[:, 0, :], h_true[:, 1, :])
    pred_c = torch.complex(h_pred[:, 0, :], h_pred[:, 1, :])

    inner = torch.real(torch.sum(true_c * torch.conj(pred_c), dim=-1))
    norm_true = torch.sqrt(torch.sum(torch.abs(true_c) ** 2, dim=-1) + eps)
    norm_pred = torch.sqrt(torch.sum(torch.abs(pred_c) ** 2, dim=-1) + eps)

    overlap = inner / (norm_true * norm_pred + eps)
    mismatch = 1.0 - overlap

    return overlap, mismatch


def overlap_loss(
    h_true: torch.Tensor,
    h_pred: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, mismatch = overlap_and_mismatch(h_true, h_pred, eps=eps)
    return mismatch.mean(), mismatch


def topk_mismatch_loss(
    mismatch: torch.Tensor,
    frac: float = 0.10,
) -> torch.Tensor:
    batch = mismatch.shape[0]
    k = max(1, int(math.ceil(frac * batch)))
    vals = torch.topk(mismatch, k=k, largest=True).values
    return vals.mean()


def smoothness_loss(
    residual_norm: torch.Tensor,
    kind: str = "curvature",
) -> torch.Tensor:
    if residual_norm.shape[-1] < 3:
        return residual_norm.new_tensor(0.0)

    if kind == "slope":
        diff = residual_norm[:, :, 1:] - residual_norm[:, :, :-1]
        return (diff ** 2).mean()

    if kind == "curvature":
        curv = (
            residual_norm[:, :, 2:]
            - 2.0 * residual_norm[:, :, 1:-1]
            + residual_norm[:, :, :-2]
        )
        return (curv ** 2).mean()

    raise ValueError(f"Unknown smoothness kind: {kind}")


# ============================================================
# Schedules
# ============================================================

def get_loss_coefficients(epoch: int, cfg: Stage3TrainConfig) -> Dict[str, float]:
    frac = epoch / max(1, cfg.num_epochs)

    if frac < 0.20:
        return {
            "res": 1.0,
            "overlap": 0.10,
            "high": 0.20,
            "smooth": 1e-5,
        }

    if frac < cfg.hard_start_frac:
        return {
            "res": 0.50,
            "overlap": 1.00,
            "high": 0.50,
            "smooth": 1e-5,
        }

    return {
        "res": 0.20,
        "overlap": 1.00,
        "high": 1.00,
        "smooth": 1e-5,
    }


def set_lr_for_epoch(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Stage3TrainConfig,
) -> float:
    frac = epoch / max(1, cfg.num_epochs)

    if frac < 0.20:
        lr = cfg.lr
    elif frac < cfg.hard_start_frac:
        lr = min(cfg.lr, 1e-4)
    else:
        lr = min(cfg.lr, 3e-5)

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


# ============================================================
# Full loss computation
# ============================================================

def compute_stage3_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    coeffs: Dict[str, float],
    cfg: Stage3TrainConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    x = batch["input"].to(device, non_blocking=True)
    target_norm = batch["target_norm_residual"].to(device, non_blocking=True)

    h_ml, _ = split_stage3_input(x)

    h_true = reconstruct_true_from_normalized_residual(
        h_ml,
        target_norm,
        eps=cfg.eps,
    )

    pred_norm = model(x)

    h_pred = apply_predicted_normalized_residual(
        h_ml,
        pred_norm,
        eps=cfg.eps,
    )

    loss_res = weighted_normalized_residual_loss(
        pred_norm=pred_norm,
        target_norm=target_norm,
        h_ref=h_true,
        w_min=cfg.w_min,
        gamma=cfg.gamma,
        eps=cfg.eps,
    )

    loss_ov, mismatch = overlap_loss(
        h_true=h_true,
        h_pred=h_pred,
        eps=cfg.eps,
    )

    loss_high = topk_mismatch_loss(
        mismatch=mismatch,
        frac=cfg.topk_frac,
    )

    loss_smooth = smoothness_loss(
        residual_norm=pred_norm,
        kind=cfg.smooth_kind,
    )

    total = (
        coeffs["res"] * loss_res
        + coeffs["overlap"] * loss_ov
        + coeffs["high"] * loss_high
        + coeffs["smooth"] * loss_smooth
    )

    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_res": float(loss_res.detach().cpu()),
        "loss_overlap": float(loss_ov.detach().cpu()),
        "loss_high": float(loss_high.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
        "mismatch_mean": float(mismatch.mean().detach().cpu()),
        "mismatch_median": float(mismatch.median().detach().cpu()),
        "mismatch_max": float(mismatch.max().detach().cpu()),
    }

    tensors = {
        "x": x.detach(),
        "h_ml": h_ml.detach(),
        "h_true": h_true.detach(),
        "h_pred": h_pred.detach(),
        "target_norm": target_norm.detach(),
        "pred_norm": pred_norm.detach(),
        "mismatch": mismatch.detach(),
    }

    return total, stats, tensors


# ============================================================
# Evaluation and hard mining
# ============================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    cfg: Stage3TrainConfig,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    losses = []
    res_losses = []
    ov_losses = []
    high_losses = []
    smooth_losses = []
    mismatches = []

    coeffs = {
        "res": 1.0,
        "overlap": 1.0,
        "high": 1.0,
        "smooth": 1e-5,
    }

    for bidx, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
        if max_batches is not None and bidx >= max_batches:
            break

        loss, stats, tensors = compute_stage3_loss(
            model=model,
            batch=batch,
            coeffs=coeffs,
            cfg=cfg,
            device=device,
        )

        losses.append(stats["loss_total"])
        res_losses.append(stats["loss_res"])
        ov_losses.append(stats["loss_overlap"])
        high_losses.append(stats["loss_high"])
        smooth_losses.append(stats["loss_smooth"])
        mismatches.append(tensors["mismatch"].detach().cpu())

    mismatch_all = torch.cat(mismatches, dim=0).numpy()

    return {
        "loss_total": float(np.mean(losses)),
        "loss_res": float(np.mean(res_losses)),
        "loss_overlap": float(np.mean(ov_losses)),
        "loss_high": float(np.mean(high_losses)),
        "loss_smooth": float(np.mean(smooth_losses)),
        "mismatch_mean": float(np.mean(mismatch_all)),
        "mismatch_median": float(np.median(mismatch_all)),
        "mismatch_p90": float(np.quantile(mismatch_all, 0.90)),
        "mismatch_p95": float(np.quantile(mismatch_all, 0.95)),
        "mismatch_p99": float(np.quantile(mismatch_all, 0.99)),
        "mismatch_max": float(np.max(mismatch_all)),
    }


@torch.no_grad()
def mine_hard_samples(
    model: nn.Module,
    dataset: Dataset,
    cfg: Stage3TrainConfig,
    device: torch.device,
    top_frac: float = 0.15,
) -> np.ndarray:
    kwargs = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        drop_last=False,
    )

    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor

    loader = DataLoader(**kwargs)

    model.eval()

    all_indices = []
    all_mismatch = []

    coeffs = {
        "res": 1.0,
        "overlap": 1.0,
        "high": 1.0,
        "smooth": 1e-5,
    }

    for batch in tqdm(loader, desc="Mining hard samples", leave=False):
        _, _, tensors = compute_stage3_loss(
            model=model,
            batch=batch,
            coeffs=coeffs,
            cfg=cfg,
            device=device,
        )

        all_indices.append(batch["index"].cpu())
        all_mismatch.append(tensors["mismatch"].detach().cpu())

    indices = torch.cat(all_indices, dim=0).numpy()
    mismatch = torch.cat(all_mismatch, dim=0).numpy()

    n_hard = max(1, int(math.ceil(top_frac * len(indices))))
    hard_order = np.argsort(mismatch)[-n_hard:]
    hard_indices = indices[hard_order]

    return hard_indices


# ============================================================
# Plotting
# ============================================================

@torch.no_grad()
def plot_batch_predictions(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Stage3TrainConfig,
    device: torch.device,
    outpath: str,
    title: str = "",
    num_samples: int = 4,
) -> None:
    model.eval()

    coeffs = {
        "res": 1.0,
        "overlap": 1.0,
        "high": 1.0,
        "smooth": 1e-5,
    }

    _, _, tensors = compute_stage3_loss(
        model=model,
        batch=batch,
        coeffs=coeffs,
        cfg=cfg,
        device=device,
    )

    h_ml = tensors["h_ml"].cpu()
    h_true = tensors["h_true"].cpu()
    h_pred = tensors["h_pred"].cpu()
    pred_norm = tensors["pred_norm"].cpu()
    target_norm = tensors["target_norm"].cpu()
    mismatch = tensors["mismatch"].cpu()

    b = min(num_samples, h_ml.shape[0])
    t = np.arange(h_ml.shape[-1])

    fig, axes = plt.subplots(
        b,
        4,
        figsize=(22, 4 * b),
        squeeze=False,
    )

    for i in range(b):
        axes[i, 0].plot(t, h_true[i, 0].numpy(), label="true hp", linewidth=1)
        axes[i, 0].plot(t, h_ml[i, 0].numpy(), label="ml hp", linewidth=1)
        axes[i, 0].plot(t, h_pred[i, 0].numpy(), label="corr hp", linewidth=1)
        axes[i, 0].set_title(f"hp, mismatch={mismatch[i].item():.3e}")
        axes[i, 0].legend()

        axes[i, 1].plot(t, h_true[i, 1].numpy(), label="true hc", linewidth=1)
        axes[i, 1].plot(t, h_ml[i, 1].numpy(), label="ml hc", linewidth=1)
        axes[i, 1].plot(t, h_pred[i, 1].numpy(), label="corr hc", linewidth=1)
        axes[i, 1].set_title("hc")
        axes[i, 1].legend()

        axes[i, 2].plot(t, target_norm[i, 0].numpy(), label="target r_norm hp", linewidth=1)
        axes[i, 2].plot(t, pred_norm[i, 0].numpy(), label="pred r_norm hp", linewidth=1)
        axes[i, 2].set_title("normalized hp residual")
        axes[i, 2].legend()

        axes[i, 3].plot(t, target_norm[i, 1].numpy(), label="target r_norm hc", linewidth=1)
        axes[i, 3].plot(t, pred_norm[i, 1].numpy(), label="pred r_norm hc", linewidth=1)
        axes[i, 3].set_title("normalized hc residual")
        axes[i, 3].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_loss_curves(history: List[Dict[str, float]], outpath: str) -> None:
    if len(history) == 0:
        return

    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, [h["train_loss_total"] for h in history], label="train total")
    if "valid_loss_total" in history[-1]:
        axes[0, 0].plot(epochs, [h.get("valid_loss_total", np.nan) for h in history], label="valid total")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Total loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [h["train_loss_res"] for h in history], label="res")
    axes[0, 1].plot(epochs, [h["train_loss_overlap"] for h in history], label="overlap")
    axes[0, 1].plot(epochs, [h["train_loss_high"] for h in history], label="high")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Train loss components")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [h["train_mismatch_mean"] for h in history], label="train mean")
    axes[1, 0].plot(epochs, [h["train_mismatch_median"] for h in history], label="train median")
    if "valid_mismatch_mean" in history[-1]:
        axes[1, 0].plot(epochs, [h.get("valid_mismatch_mean", np.nan) for h in history], label="valid mean")
        axes[1, 0].plot(epochs, [h.get("valid_mismatch_median", np.nan) for h in history], label="valid median")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Mismatch mean and median")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, [h["train_mismatch_max"] for h in history], label="train max")
    if "valid_mismatch_p99" in history[-1]:
        axes[1, 1].plot(epochs, [h.get("valid_mismatch_p99", np.nan) for h in history], label="valid p99")
        axes[1, 1].plot(epochs, [h.get("valid_mismatch_max", np.nan) for h in history], label="valid max")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Worst-case mismatch")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ============================================================
# Saving
# ============================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Stage3TrainConfig,
    history: List[Dict[str, float]],
    outpath: str,
    extra: Optional[Dict] = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
        "history": history,
        "extra": extra or {},
    }
    torch.save(payload, outpath)


def save_history_csv(history: List[Dict[str, float]], outpath: str) -> None:
    if len(history) == 0:
        return
    keys = sorted(set().union(*[h.keys() for h in history]))
    with open(outpath, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in history:
            vals = [str(row.get(k, "")) for k in keys]
            f.write(",".join(vals) + "\n")


# ============================================================
# Main training loop
# ============================================================

def train_finetuner(
    model: nn.Module,
    cfg: FinetunerTrainConfig,
) -> nn.Module:
    set_seed(cfg.seed)

    device = get_device(cfg.device)
    ensure_dir(cfg.outdir)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    run_id = now_string()
    run_dir = os.path.join(cfg.outdir, f"finetuner_run_{run_id}")
    plot_dir = os.path.join(run_dir, "plots")
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    ensure_dir(run_dir)
    ensure_dir(plot_dir)
    ensure_dir(ckpt_dir)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    train_dataset = FinetunerDataset(
        hdf_path=cfg.train_path,
        input_key=cfg.input_key,
        target_key=cfg.target_key,
        max_samples=cfg.max_train_samples,
    )

    valid_dataset = FinetunerDataset(
        hdf_path=cfg.valid_path,
        input_key=cfg.input_key,
        target_key=cfg.target_key,
        max_samples=cfg.max_valid_samples,
    )

    train_loader_builder = FinetunerDataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
        drop_last=True,
    )

    valid_kwargs = dict(
        dataset=valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        drop_last=False,
    )

    if cfg.num_workers > 0:
        valid_kwargs["prefetch_factor"] = cfg.prefetch_factor

    valid_loader = DataLoader(**valid_kwargs)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    use_cuda_amp = cfg.use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda" if use_cuda_amp else "cpu")

    history = []
    best_valid_score = float("inf")
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    fixed_plot_batch = None
    hard_indices = None

    for epoch in range(1, cfg.num_epochs + 1):
        epoch_start = time.perf_counter()

        lr = set_lr_for_epoch(optimizer, epoch, cfg)
        coeffs = get_loss_coefficients(epoch, cfg)

        hard_phase = (epoch / cfg.num_epochs) >= cfg.hard_start_frac

        if hard_phase:
            should_refresh = (
                hard_indices is None
                or ((epoch - 1) % cfg.hard_refresh_every == 0)
            )

            if should_refresh:
                hard_indices = mine_hard_samples(
                    model=model,
                    dataset=train_dataset,
                    cfg=cfg,
                    device=device,
                    top_frac=cfg.hard_top_frac,
                )
                np.save(
                    os.path.join(run_dir, f"hard_indices_epoch_{epoch:04d}.npy"),
                    hard_indices,
                )

            train_loader_builder.set_hard_indices(
                hard_indices=hard_indices,
                hard_weight=cfg.hard_sample_weight,
            )
        else:
            train_loader_builder.set_hard_indices(None)

        train_loader = train_loader_builder.build()
        model.train()

        accum = {
            "loss_total": [],
            "loss_res": [],
            "loss_overlap": [],
            "loss_high": [],
            "loss_smooth": [],
            "mismatch_mean": [],
            "mismatch_median": [],
            "mismatch_max": [],
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}")

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            if use_cuda_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    loss, stats, tensors = compute_stage3_loss(
                        model=model,
                        batch=batch,
                        coeffs=coeffs,
                        cfg=cfg,
                        device=device,
                    )

                scaler.scale(loss).backward()

                if cfg.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                scaler.step(optimizer)
                scaler.update()

            else:
                loss, stats, tensors = compute_stage3_loss(
                    model=model,
                    batch=batch,
                    coeffs=coeffs,
                    cfg=cfg,
                    device=device,
                )

                loss.backward()

                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                optimizer.step()

            for k in accum:
                accum[k].append(stats[k])

            pbar.set_postfix({
                "loss": f"{stats['loss_total']:.3e}",
                "mm": f"{stats['mismatch_mean']:.3e}",
                "lr": f"{lr:.1e}",
            })

            if fixed_plot_batch is None:
                fixed_plot_batch = {
                    key: val.detach().cpu() if torch.is_tensor(val) else val
                    for key, val in batch.items()
                }

        train_stats = {
            f"train_{k}": float(np.mean(v))
            for k, v in accum.items()
        }

        row = {
            "epoch": epoch,
            "lr": lr,
            "coef_res": coeffs["res"],
            "coef_overlap": coeffs["overlap"],
            "coef_high": coeffs["high"],
            "coef_smooth": coeffs["smooth"],
            "hard_phase": int(hard_phase),
            **train_stats,
        }

        if epoch % cfg.validate_every == 0:
            valid_stats = evaluate_model(
                model=model,
                loader=valid_loader,
                cfg=cfg,
                device=device,
            )

            row.update({
                f"valid_{k}": v
                for k, v in valid_stats.items()
            })

            valid_score = valid_stats["mismatch_p99"]

            if valid_score < best_valid_score:
                best_valid_score = valid_score
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    cfg=cfg,
                    history=history + [row],
                    outpath=best_path,
                    extra={
                        "best_valid_score": best_valid_score,
                        "score_name": "valid_mismatch_p99",
                    },
                )

        epoch_time = time.perf_counter() - epoch_start
        row["epoch_time_sec"] = epoch_time

        history.append(row)

        save_history_csv(
            history,
            os.path.join(run_dir, "history.csv"),
        )

        plot_loss_curves(
            history,
            os.path.join(plot_dir, "loss_curves.png"),
        )

        if epoch % cfg.plot_every == 0 and fixed_plot_batch is not None:
            plot_batch_predictions(
                model=model,
                batch=fixed_plot_batch,
                cfg=cfg,
                device=device,
                outpath=os.path.join(plot_dir, f"predictions_epoch_{epoch:04d}.png"),
                title=f"Epoch {epoch}",
                num_samples=cfg.num_plot_samples,
            )

        if epoch % cfg.checkpoint_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                history=history,
                outpath=os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:04d}.pt"),
                extra={
                    "best_valid_score": best_valid_score,
                },
            )

        latest_path = os.path.join(ckpt_dir, "latest.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            history=history,
            outpath=latest_path,
            extra={
                "best_valid_score": best_valid_score,
            },
        )

        msg = (
            f"Epoch {epoch:04d} done in {epoch_time:.1f}s | "
            f"train mismatch mean={row['train_mismatch_mean']:.3e}, "
            f"train mismatch max={row['train_mismatch_max']:.3e}"
        )

        if "valid_mismatch_p99" in row:
            msg += (
                f" | valid median={row['valid_mismatch_median']:.3e}, "
                f"valid p99={row['valid_mismatch_p99']:.3e}, "
                f"valid max={row['valid_mismatch_max']:.3e}"
            )

        print(msg)

    final_path = os.path.join(ckpt_dir, "final_model.pt")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=cfg.num_epochs,
        cfg=cfg,
        history=history,
        outpath=final_path,
        extra={
            "best_valid_score": best_valid_score,
        },
    )

    print(f"Training complete. Best model: {best_path}")
    return model


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    cfg = FinetunerTrainConfig(
        train_path="../data/stage3_train_hphc_residuals.hdf",
        valid_path="../data/stage3_valid_hphc_residuals.hdf",
        outdir="../v0p1/results/stage3_polarization",

        input_key="inputs",
        target_key="targets",

        num_epochs=100,
        batch_size=512,
        num_workers=4,

        lr=3e-4,
        weight_decay=1e-5,

        validate_every=1,
        plot_every=5,
        checkpoint_every=5,

        hard_start_frac=0.70,
        hard_top_frac=0.15,
        hard_sample_weight=8.0,

        w_min=0.05,
        gamma=1.0,
        topk_frac=0.10,

        use_amp=True,
        amp_dtype="float16",

        max_train_samples=None,
        max_valid_samples=None,
    )

    model = ResidualCalibrationCNN(
        input_channels=6,
        output_channels=2,
        hidden_channels=64,
        num_blocks=6,
        kernel_size=7,
        dropout=0.0,
    )
    train_finetuner(model, cfg)
    # -- save config to JSON
    cfg_path = os.path.join(cfg.outdir, "finetuner_training_config.json")
    ensure_dir(os.path.dirname(cfg_path))
    with open(cfg_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
