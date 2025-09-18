# Multi Trainer - configuration search + pruning + retrain (PyTorch)
# ---------------------------------------------------------------
# Minimal-friction module that works with the Multi CVAE refactor.
# It trains multiple architecture configurations for 10 epochs, prunes
# the bottom half every 2 epochs by validation loss, saves the top K
# models each epoch, then retrains the single best configuration for
# another 10 epochs and saves that final model too.
#
# Your data: pass any iterable/generator / DataLoader yielding dicts
# with keys {'x', 'y'} (and optional others like 'z' - ignored by default).
#
# Multi touches:
# - Auto GPU + AMP mixed precision
# - Optional torch.compile for speed (PyTorch >= 2)
# - Narrow search around a prior (e.g., 2C2E1D) to save time
# - Clean run folders, JSON config snapshots, CSV logs
# - Human-friendly prompts or auto-continue flags
# - Encoder/decoder per-index overrides (e.g., encoder 1 has no CNN)

from init import *  # expects torch, nn, optim, etc.
from utils import *
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable
import os, json, math, time, csv
from pathlib import Path

from multicvae import build_model_from_code  # provided in Multi refactor


# -----------------------------
# Small utilities
# -----------------------------
def now_uid() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def default_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)


# -----------------------------
# Config generation around a prior
# -----------------------------
def suggest_configs_around_prior(
    *,
    input_shape: Tuple[int, ...],
    latent_dim: int,
    num_classes: int,
    prior_code: str = "2C2E1D",
    cnn_layers_choices: Sequence[int] = (2, 3),
    fc_layers_choices: Sequence[int] = (1, 2),
    cnn_out_base_choices: Sequence[int] = (32, 64),
    one_encoder_may_disable_cnn: bool = True,
    cond_dim_choices: Sequence[int] = (0, 16),  # 0 → no conditioners
) -> List[Dict[str, Any]]:
    """Produce a small, high-quality set of candidate configs near a prior.
    Keeps search tight so runtime stays sane on large datasets.
    """
    C, L = input_shape
    base_pre_in = (C * L) + num_classes  # when concatenating y

    configs = []
    for n_cnn in cnn_layers_choices:
        for n_fc in fc_layers_choices:
            for base_out in cnn_out_base_choices:
                for cond_dim in cond_dim_choices:
                    kwargs = dict(
                        input_shape=input_shape,
                        latent_dim=latent_dim,
                        num_classes=num_classes,
                        activation='silu',
                        use_batchnorm=True,
                        dropout_p=0.1,
                        # layer counts
                        n_layers_cnn_encoder=n_cnn,
                        n_layers_cnn_decoder=n_cnn,
                        n_layers_pre_fc=n_fc,
                        n_layers_post_fc=max(1, n_fc - 1),
                        encoder_has_cnn=True,
                        decoder_has_cnn=True,
                        encoder_has_fc=True,
                        decoder_has_fc=True,
                        # fc sizes
                        pre_fc_sizes=[base_pre_in, 512, 256][: n_fc + 1],
                        post_fc_sizes=[256 + num_classes, latent_dim] if n_fc >= 1 else [base_pre_in, latent_dim],
                        # cnn specs (simple doubling)
                        cnn_in_channels=[1] + [base_out, base_out * 2][: max(0, n_cnn - 1)],
                        cnn_out_channels=[base_out] + [base_out * 2][: max(0, n_cnn - 1)],
                        cnn_kernel_size=[7] + [5] * (n_cnn - 1),
                        cnn_dilation=[1] * n_cnn,
                        cnn_pool_kernel_size=[2] * n_cnn,
                    )
                    code = prior_code
                    encoder_overrides = {}
                    if one_encoder_may_disable_cnn:
                        # disable CNN on encoder index 1 (second) in this candidate
                        encoder_overrides = {1: {"has_cnn": False}}
                    conders = 0 if cond_dim == 0 else 2
                    configs.append({
                        "code": code,
                        "kwargs": {**kwargs, "n_conditioners": conders, "cond_dim": (cond_dim if cond_dim else None)},
                        "encoder_overrides": encoder_overrides,
                        "tag": f"code-{code}_cnn{n_cnn}_fc{n_fc}_c{conders}_base{base_out}"
                    })
    return configs


# -----------------------------
# Training core
# -----------------------------
class MultiTrainer:
    def __init__(
        self,
        *,
        train_data: Iterable[Dict[str, Any]],
        val_data: Optional[Iterable[Dict[str, Any]]] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        optimizer_ctor: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs_stage1: int = 10,
        epochs_stage2: int = 10,
        prune_every: int = 2,
        topk_save_per_epoch: int = 3,
        save_dir: str = "runs",
        run_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        amp: bool = True,
        autocast_dtype: torch.dtype = torch.float16,
        compile_model: bool = False,
        grad_accum_steps: int = 1,
        seed: int = 1337,
        auto_continue: Optional[bool] = None,  # None → prompt via input()
        build_fn: Callable[..., Any] = build_model_from_code,
        extra_forward_kwargs_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.train_data = train_data
        self.val_data = val_data or train_data
        self.loss_fn = loss_fn or default_loss
        self.optimizer_ctor = optimizer_ctor
        self.lr = lr
        self.weight_decay = weight_decay
        self.E1 = epochs_stage1
        self.E2 = epochs_stage2
        self.prune_every = prune_every
        self.topk = topk_save_per_epoch
        self.save_root = Path(save_dir)
        self.run_id = run_name or now_uid()
        self.out_dir = self.save_root / f"search_{self.run_id}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.amp = amp and (self.device.type == 'cuda')
        self.autocast_dtype = autocast_dtype
        self.compile_model = compile_model and hasattr(torch, 'compile')
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.seed = seed
        self.auto_continue = auto_continue
        self.build_fn = build_fn
        self.extra_forward_kwargs_fn = extra_forward_kwargs_fn

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True

        # CSV log
        self.csv_path = self.out_dir / "metrics.csv"
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["epoch", "config_tag", "val_loss", "phase"])  # phase ∈ {stage1, stage2}

    # -------- helpers --------
    def _build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        model = self.build_fn(cfg["code"], **cfg["kwargs"]).to(self.device)
        # apply encoder/decoder overrides post-hoc (per-index tweaks)
        if cfg.get("encoder_overrides"):
            for idx, od in cfg["encoder_overrides"].items():
                for k, v in od.items():
                    setattr(model.encoders[idx], k, v)
        if cfg.get("decoder_overrides"):
            for idx, od in cfg["decoder_overrides"].items():
                for k, v in od.items():
                    setattr(model.decoders[idx], k, v)
        if self.compile_model:
            model = torch.compile(model)
        return model

    def _optimizer(self, params) -> torch.optim.Optimizer:
        return self.optimizer_ctor(params, lr=self.lr, weight_decay=self.weight_decay)

    def _train_or_eval_epoch(self, model: nn.Module, data: Iterable[Dict[str, Any]], train: bool = True) -> float:
        model.train(train)
        total_loss, n = 0.0, 0
        opt = None
        scaler = None
        if train:
            opt = self._optimizer(model.parameters())
            scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        for step, batch in enumerate(data):
            batch = to_device(batch, self.device)
            x = batch['x']
            y = batch['y']
            fwd_kwargs = {}
            if self.extra_forward_kwargs_fn is not None:
                fwd_kwargs = self.extra_forward_kwargs_fn(batch)

            with torch.set_grad_enabled(train):
                if train and self.amp:
                    with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                        pred = model(x, y, **fwd_kwargs)
                        loss = self.loss_fn(pred, x)  # reconstruction by default
                else:
                    pred = model(x, y, **fwd_kwargs)
                    loss = self.loss_fn(pred, x)

            if train:
                (scaler.scale(loss) if self.amp else loss).backward()
                if (step + 1) % self.grad_accum_steps == 0:
                    if self.amp:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

            total_loss += loss.detach().item()
            n += 1
        return total_loss / max(1, n)

    def _save_checkpoint(self, epoch: int, rank: int, model: nn.Module, cfg: Dict[str, Any], val_loss: float, phase: str):
        ep_dir = self.out_dir / f"{phase}_epoch{epoch:02d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        # concise tag
        tag = cfg.get("tag", cfg["code"]) 
        # filename captures essentials
        fname = ep_dir / f"rank{rank}_val{val_loss:.6f}_{tag}.pt"
        torch.save({
            'model_state': model.state_dict(),
            'config': cfg,
            'val_loss': val_loss,
            'epoch': epoch,
            'phase': phase,
        }, fname)
        # also snapshot human-readable config
        with open(ep_dir / f"rank{rank}_{tag}.json", 'w') as jf:
            json.dump(cfg, jf, indent=2)

    def _log_csv(self, epoch: int, cfg_tag: str, val_loss: float, phase: str):
        with open(self.csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([epoch, cfg_tag, val_loss, phase])

    # -------- main routine --------
    def search_and_optionally_retrain(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Stage 1: search with pruning
        active = [{**cfg} for cfg in configs]
        models = [self._build_model(cfg) for cfg in active]

        best_overall = {"loss": float('inf'), "cfg": None, "model": None}

        for epoch in range(1, self.E1 + 1):
            ep_scores = []
            # train + evaluate each active config
            for i, (cfg, model) in enumerate(zip(active, models)):
                _ = self._train_or_eval_epoch(model, self.train_data, train=True)
                val_loss = self._train_or_eval_epoch(model, self.val_data, train=False)
                ep_scores.append((val_loss, i))
                self._log_csv(epoch, cfg.get("tag", cfg["code"]), val_loss, phase="stage1")
                if val_loss < best_overall["loss"]:
                    best_overall = {"loss": val_loss, "cfg": cfg, "model": model}
            # choose top-K for saving
            ep_scores.sort(key=lambda t: t[0])
            for rank, (_, i) in enumerate(ep_scores[: self.topk], start=1):
                self._save_checkpoint(epoch, rank, models[i], active[i], ep_scores[rank-1][0], phase="stage1")

            # prune every N epochs (keep upper half best → round down)
            if epoch % self.prune_every == 0 and len(active) > 1:
                keep = math.ceil(len(active) / 2.0)
                keep_idx = [i for (_, i) in ep_scores[: keep]]
                active = [active[i] for i in keep_idx]
                models = [models[i] for i in keep_idx]

        # Save best config after stage1
        best_cfg = best_overall["cfg"]
        best_cfg_path = self.out_dir / "best_after_stage1.json"
        with open(best_cfg_path, 'w') as jf:
            json.dump(best_cfg, jf, indent=2)
        # Save its checkpoint too
        self._save_checkpoint(self.E1, 1, best_overall["model"], best_cfg, best_overall["loss"], phase="stage1_best")

        # Prompt whether to continue
        proceed = self.auto_continue
        if proceed is None:
            try:
                ans = input("Multi: Continue to retrain best configuration for 10 more epochs? [y/N] ").strip().lower()
                proceed = ans in ("y", "yes")
            except Exception:
                proceed = False
        if not proceed:
            print(f"Best configuration saved to: {best_cfg_path}")
            return {"best_config": best_cfg, "continued": False}

        # Stage 2: retrain the best configuration from scratch (fresh model)
        best_model = self._build_model(best_cfg)
        best_val = float('inf')
        best_file = None
        for epoch in range(1, self.E2 + 1):
            _ = self._train_or_eval_epoch(best_model, self.train_data, train=True)
            val_loss = self._train_or_eval_epoch(best_model, self.val_data, train=False)
            self._log_csv(epoch, best_cfg.get("tag", best_cfg["code"]), val_loss, phase="stage2")
            self._save_checkpoint(epoch, 1, best_model, best_cfg, val_loss, phase="stage2")
            if val_loss < best_val:
                best_val = val_loss
                best_file = self.out_dir / f"stage2_best_epoch{epoch:02d}.pt"
                torch.save({'model_state': best_model.state_dict(), 'config': best_cfg, 'val_loss': best_val, 'epoch': epoch, 'phase': 'stage2'}, best_file)

        final_path = self.out_dir / "best_retrained_final.pt"
        torch.save({'model_state': best_model.state_dict(), 'config': best_cfg, 'phase': 'stage2_final'}, final_path)
        return {"best_config": best_cfg, "continued": True, "stage2_best": str(best_file), "final": str(final_path)}


# -----------------------------
# Example wiring
# -----------------------------
"""
from torch.utils.data import DataLoader

# Suppose your generator yields {'x': x, 'y': y}
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Create candidate configs near a known-good prior (2C2E1D)
configs = suggest_configs_around_prior(
    input_shape=(1, 2048), latent_dim=128, num_classes=10,
    prior_code="2C2E1D", cnn_layers_choices=(2,3), fc_layers_choices=(1,2), cnn_out_base_choices=(32,64),
    one_encoder_may_disable_cnn=True, cond_dim_choices=(0,16)
)

trainer = MultiTrainer(
    train_data=train_loader,
    val_data=val_loader,
    epochs_stage1=10,
    epochs_stage2=10,
    prune_every=2,
    topk_save_per_epoch=3,
    amp=True,
    compile_model=True,   # if PyTorch >= 2.0
    auto_continue=None,   # None - prompt; or True/False
)

result = trainer.search_and_optionally_retrain(configs)
print(result)
"""
