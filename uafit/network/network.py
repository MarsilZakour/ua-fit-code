"""``UAFitNetwork``: images of all views -> the evidence the UA-Fit solver fits.

Forward graph of the released HRNet-W40 checkpoint (inference only):

    image -> HRNet-W40 -> MultiResNeck -> {HeatmapHead, CoarseToFineDOVFHead (vote field),
                                           UncertaintyFieldHead (vote precision)}
      -> precision-weighted 2D vote consensus per view            (consensus.py)
      -> robust, uncertainty-weighted multi-view triangulation     (triangulation.py) -> joints_world
      -> shape head (pooled descriptor + bone-length residual MLP)                     -> mano_betas

Checkpoint modules that are not built (unused at inference; listed with reasons in
``tests/test_network_parity.py``): the descriptor head and pose refiner (training
only), the epipolar refiner (zero refinement steps in the released configuration),
HRNet's ImageNet classification head, and duplicated MANO / normalisation buffers.
The solver's pose prior is rebuilt from MANO's PCA (``uafit.hand_models``) and is
identical to the ``prior_*`` buffers stored in the checkpoint.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from . import heads
from .consensus import precision_consensus_2d
from .hrnet import HRNetW40, W40_OUT_CHANNELS
from .triangulation import pad_by_scene, triangulate_omega2


# ──────────────────────────────────────────────────────────────────────────
# Network <-> fitter hand-off
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class NetworkEvidence:
    """Everything the solver needs, produced by one ``UAFitNetwork`` forward.

    V_total = sum(views_per_scene) is the flat (batch*views) layout every
    per-view tensor uses; B = len(views_per_scene) is the per-scene layout.
    """
    joints_world: torch.Tensor          # (B, 21, 3) m, triangulated joints (legacy "mu3d"), OpenPose order
    mano_betas: torch.Tensor            # (B, 10) shape-head MANO betas (legacy "beta_hat")
    vote_field: torch.Tensor            # (V_total, 21, 2, h, w) per-view dense joint vote field (legacy "dovf_field")
    vote_precision_chol: torch.Tensor   # (V_total, 21, 3, h, w) raw Cholesky params of the per-vote 2x2 precision (legacy "chol_field")
    intrinsics_heatmap: torch.Tensor    # (V_total, 3, 3) intrinsics scaled to heatmap pixels
    world_to_camera: torch.Tensor       # (V_total, 4, 4) inverse of batch["target_cam_extr"]
    views_per_scene: List[int]          # batch["cam_view_num"], len B
    heatmap_size: Tuple[int, int]       # (h, w)
    joints_2d_heatmap: Optional[torch.Tensor] = None     # (V_total, 21, 2) per-view 2D joint consensus, heatmap px
    joints_2d_precision: Optional[torch.Tensor] = None   # (V_total, 21, 2, 2) its precision (inverse covariance), 1/px^2


@dataclass(frozen=True)
class NetworkConfig:
    """The handful of config keys the network actually needs (vs. the legacy
    330-line training YAML). Defaults match ``configs/uafit_eval.yaml`` /
    the released W40 checkpoint."""
    feat_dim: int = 128
    num_joints: int = 21
    heatmap_size: Tuple[int, int] = (64, 64)
    image_size: Tuple[int, int] = (256, 256)
    center_idx: int = 0
    mano_assets_root: str = "assets/mano_v1_2"
    # MODEL.CHANNELS_LAST in configs/uafit_eval.yaml. Purely a conv memory-format
    # (NHWC) choice for throughput; mathematically a no-op, but it changes which
    # cuDNN convolution algorithm gets picked, which shifts floating-point
    # rounding by ~1e-2 on the raw per-pixel vote_field after 4 decoder levels
    # (still <0.03 heatmap px; <3e-5 m after triangulation). Matched here so the
    # parity gate compares bit-for-bit against the released checkpoint's own
    # config instead of against a silently-different memory format.
    channels_last: bool = True


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class UAFitNetwork(nn.Module):
    """Image(s) -> (joints_world, mano_betas, vote_field, vote_precision_chol).

    Submodule / parameter / buffer names (``img_backbone``, ``neck``,
    ``heatmap_head``, ``dovf_head``, ``unc_head``, ``init_head``,
    ``beta_bone_mlp``, ``log_tau``, ``fit_log_radius``, ``prior_prec``,
    ``prior_ref``, ``prior_anchor``) are identical to the released
    checkpoint's top-level keys, so ``load_state_dict(ckpt, strict=False)``
    populates every weight here without any remapping.
    """

    def __init__(self, cfg: NetworkConfig = NetworkConfig()):
        super().__init__()
        self.cfg = cfg
        j = cfg.num_joints

        self.img_backbone = HRNetW40()
        self.neck = heads.MultiResNeck(W40_OUT_CHANNELS, feat_dim=cfg.feat_dim)
        self.heatmap_head = heads.HeatmapHead(cfg.feat_dim, j, cfg.heatmap_size)
        self.dovf_head = heads.CoarseToFineDOVFHead(cfg.feat_dim, j, cfg.heatmap_size,
                                                    n_levels=len(W40_OUT_CHANNELS))
        self.unc_head = heads.UncertaintyFieldHead(cfg.feat_dim, j)
        self.init_head = heads.ManoInitHead(cfg.feat_dim, n_pose=48, n_betas=10)
        self.beta_bone_mlp = heads.build_beta_bone_mlp()

        self.log_tau = nn.Parameter(torch.zeros(()))              # consensus temperature
        self.fit_log_radius = nn.Parameter(torch.zeros(1))        # learned Huber radius of the solver (log px)
        self._channels_last_done = False

    # ──────────────────────────────────────────────────────────────────
    def _hm_scale(self, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        hm_h, hm_w = self.cfg.heatmap_size
        img_h, img_w = self.cfg.image_size
        return (torch.tensor(hm_w / img_w, device=device, dtype=dtype),
                torch.tensor(hm_h / img_h, device=device, dtype=dtype))

    def _renorm(self, img: torch.Tensor) -> torch.Tensor:
        mean = img.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = img.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        return (img + 0.5 - mean) / std

    # ──────────────────────────────────────────────────────────────────
    def forward(self, batch: dict) -> NetworkEvidence:
        images = batch["image"]
        if images.dim() == 5:
            images = images.reshape(-1, *images.shape[-3:])
        k_all = batch["target_cam_intr"].reshape(-1, 3, 3)
        extr_all = batch["target_cam_extr"].reshape(-1, 4, 4)           # T_cam->world
        w2c_all = torch.linalg.inv(extr_all)                            # world->cam
        views_per_scene = [int(n) for n in batch["cam_view_num"]]
        device = images.device

        images = self._renorm(images)
        if self.cfg.channels_last:
            if not self._channels_last_done:            # convert weights once (mirrors legacy _backbone_forward)
                self.img_backbone.to(memory_format=torch.channels_last)
                self.neck.to(memory_format=torch.channels_last)
                self.heatmap_head.to(memory_format=torch.channels_last)
                self.dovf_head.to(memory_format=torch.channels_last)
                self._channels_last_done = True
            images = images.contiguous(memory_format=torch.channels_last)
        feats = self.img_backbone(images)                              # high-res -> low-res
        pyramid = self.neck(feats)
        _, hm_probs, _ = self.heatmap_head(pyramid[0])
        vote_field = self.dovf_head(pyramid)                            # (BN, J, 2, h, w)
        vote_chol = self.unc_head(pyramid[0])                           # (BN, J, 3, h, w)

        sx, sy = self._hm_scale(device, k_all.dtype)
        k_hm = k_all.clone()
        k_hm[:, 0, :] = k_hm[:, 0, :] * sx
        k_hm[:, 1, :] = k_hm[:, 1, :] * sy

        # Phase A: precision-weighted dense-vote consensus -> per-view 2D + fused Omega2.
        cons, omega2 = precision_consensus_2d(hm_probs.float(), vote_field.float(),
                                              vote_chol.float(), self.log_tau)

        # Phase B: robust, uncertainty-weighted multi-view triangulation -> joints_world.
        n_max = max(views_per_scene)
        cons_pad, vmask = pad_by_scene(cons, views_per_scene, n_max)
        k_pad, _ = pad_by_scene(k_hm.float(), views_per_scene, n_max)
        w2c_pad, _ = pad_by_scene(w2c_all.float(), views_per_scene, n_max)
        om_pad, _ = pad_by_scene(omega2, views_per_scene, n_max)
        joints_world, _sigma3 = triangulate_omega2(cons_pad, k_pad, w2c_pad, om_pad, vmask,
                                                    robust_iter=4, robust_c2=25.0)
        # Phase C (recurrent epipolar refiner) is a no-op for this checkpoint's
        # eval config (EPI.T=0) -- joints_world IS the model's mu_final; see
        # the module docstring.

        # Feed-forward MANO shape: pooled per-scene descriptor + bone-length residual.
        dv = pyramid[-1].mean(dim=(-2, -1))                              # (BN, feat_dim)
        betas_list = []
        off = 0
        for n in views_per_scene:
            desc = dv[off:off + n].mean(0, keepdim=True)
            betas_list.append(self.init_head(desc)[1])
            off += n
        mano_betas = torch.cat(betas_list, 0)                            # (B, 10)
        mu_d = joints_world.detach()
        bone_len = (mu_d[:, heads.BONE_CHILD] - mu_d[:, heads.BONE_PARENT]).norm(dim=-1)
        mano_betas = mano_betas + self.beta_bone_mlp(bone_len * 10.0)

        return NetworkEvidence(
            joints_world=joints_world,
            mano_betas=mano_betas,
            vote_field=vote_field,
            vote_precision_chol=vote_chol,
            intrinsics_heatmap=k_hm,
            world_to_camera=w2c_all,
            views_per_scene=views_per_scene,
            heatmap_size=self.cfg.heatmap_size,
            joints_2d_heatmap=cons,
            joints_2d_precision=omega2,
        )

    @property
    def huber_radius_px(self) -> float:
        """The solver's learned Huber radius in heatmap pixels (3.0 for the released checkpoint)."""
        return float(self.fit_log_radius.detach().exp().item())


# ──────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────

RELEASE_FILE = "uafit_w40.safetensors"        # released weights (network only, no MANO data)
LEGACY_FILE = "DOVFManoMVEpi.pth.tar"         # original research checkpoint


def _find_checkpoint_file(checkpoint: Union[str, Path]) -> Path:
    """A weights file, a directory holding one, or a Hugging Face model id ("<owner>/<repo>") to download from."""
    p = Path(checkpoint)
    if p.is_file():
        return p
    if p.is_dir():
        for name in (RELEASE_FILE, LEGACY_FILE):
            if (p / name).is_file():
                return p / name
        raise FileNotFoundError(f"neither {RELEASE_FILE} nor {LEGACY_FILE} in {p}")
    if len(str(checkpoint).split("/")) == 2:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=str(checkpoint), filename=RELEASE_FILE))
    raise FileNotFoundError(f"no checkpoint at {checkpoint}")


def _load_state_dict(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    return torch.load(path, map_location="cpu")


def load_uafit_network(checkpoint_path: Union[str, Path], config_path: Optional[str] = None,
                       device: Union[str, torch.device] = "cpu") -> UAFitNetwork:
    """Build a :class:`UAFitNetwork` and load the released checkpoint into it.

    Args:
        checkpoint_path: a weights file (``uafit_w40.safetensors`` or the original
            ``DOVFManoMVEpi.pth.tar``), a directory holding one, or the Hugging Face
            model id to download the released weights from.
        config_path: reserved for a future YAML override of
            :class:`NetworkConfig`; ``None`` uses the released W40 defaults.
        device: device to move the network to.

    Returns:
        The loaded, ``eval()``-mode network. ``network.state_dict().keys()``
        is a strict subset of the checkpoint's keys (asserted here); the
        checkpoint keys NOT in that subset are training-only / disabled-at-eval
        modules and are enumerated in ``tests/test_network_parity.py``.
    """
    cfg = NetworkConfig()
    if config_path is not None:
        import yaml
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        cfg = NetworkConfig(**{**cfg.__dict__, **overrides})

    net = UAFitNetwork(cfg)
    ckpt_file = _find_checkpoint_file(checkpoint_path)
    state_dict = _load_state_dict(ckpt_file)

    own_keys = set(net.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = own_keys - ckpt_keys
    assert not missing, f"UAFitNetwork has keys absent from the checkpoint: {sorted(missing)}"

    net.load_state_dict(state_dict, strict=False)
    net.ignored_checkpoint_keys = sorted(ckpt_keys - own_keys)   # for diagnostics/tests
    net.to(device)
    net.eval()
    return net
