"""Hand models UA-Fit can fit to its network evidence: MANO (the paper), MHR and UmeTrack.

The network predicts MANO-convention evidence (21 mesh-regressed MANO joints in
OpenPose order, plus MANO shape betas). Every hand model therefore exposes the
same interface to the solver: a parafit ``Model`` whose 21 landmarks are those
MANO joints, an initialisation from the triangulated joints, its pose prior, and
MANO-topology vertices for evaluation.

* ``mano``: parafit's ``ManoModel``, numerically identical to the paper's solver.
* ``mhr`` / ``umetrack``: the target model wrapped by
  ``parafit.mapping.ManoConventionHand``; each MANO joint rides on a calibrated
  target bone (``scripts/build_mano_correspondence.py``), the hand shape is a
  uniform scale transferred from the predicted MANO betas, and MANO-topology
  vertices are read off the target mesh through the dense correspondence.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from parafit.core.lie import rotation_to_axis_angle
from parafit.energies.base import Energy
from parafit.energies.pose_prior import PosePriorEnergy
from parafit.mapping import ManoConventionHand, ManoCorrespondence, hand_scale_between
from parafit.mapping.calibration import rigid_align
from parafit.models.base import Model
from parafit.models.mano import ManoModel

HAND_MODELS = ("mano", "mhr", "umetrack")


@dataclass(frozen=True)
class HandModelPaths:
    """Asset locations (see ``assets/README.md``)."""
    mano_assets: str = "assets/mano_v1_2"
    mhr_cache: str = "assets/mhr/mhr_lod1_cache.pt"
    umetrack_json: str = "assets/umetrack/generic_hand_model.json"
    correspondence_dir: str = "assets/correspondences"


@dataclass(frozen=True)
class PriorWeights:
    """Pose prior of the fit. The MANO values are the paper's (PCA-Mahalanobis finger prior + root anchor)."""
    finger_pose: float = 0.01      # MANO: PCA-Mahalanobis precision scale; MHR/UmeTrack: L2 to the rest pose
    root_anchor: float = 0.05      # L2 pull of root rotation / translation to the initialisation


class FittableHand(ABC):
    """A hand model the UA-Fit solver can fit (MANO-convention landmarks)."""

    name: str
    model: Model

    @abstractmethod
    def set_shape(self, mano_betas: Tensor) -> None:
        """Adopt the network's MANO shape (B,10) for the next fit."""

    @abstractmethod
    def initial_params(self, joints_world: Tensor) -> Tensor:
        """(B,21,3) triangulated joints -> initial parameters (B,P): rigid alignment of the rest hand."""

    @abstractmethod
    def prior_energies(self, init_params: Tensor) -> list[Energy]:
        """Pose prior terms of the objective."""

    @abstractmethod
    def mano_vertices(self, params: Tensor) -> Tensor:
        """MANO-topology vertices (B,778,3) in metres, for MPVPE against MANO ground truth."""

    def landmarks(self, params: Tensor) -> Tensor:
        """MANO-convention joints (B,21,3) in metres."""
        return self.model.forward(params).landmarks


class ManoHand(FittableHand):
    """MANO exactly as in the paper: axis-angle pose (48) + translation (3), flat-hand mean, wrist-centred."""

    name = "mano"

    def __init__(self, mano_assets: str, prior: PriorWeights = PriorWeights(), device="cpu"):
        self.model = ManoModel(mano_assets, side="right", center_idx=0, flat_hand_mean=True, joints="regressed")
        self.model.mano_layer.to(device)
        self.prior = prior
        self.device = torch.device(device)
        self._pca_precision, self._pca_mean = self._pca_pose_prior(mano_assets)

    def _pca_pose_prior(self, mano_assets: str) -> tuple[Tensor, Tensor]:
        """Mahalanobis precision of MANO's PCA finger-pose distribution, normalised to unit mean diagonal, and its mean."""
        from manotorch.manolayer import ManoLayer
        pca = ManoLayer(joint_rot_mode="pca", use_pca=True, ncomps=45, mano_assets_root=mano_assets,
                        center_idx=0, flat_hand_mean=True)
        basis = pca.th_selected_comps.detach().float()
        precision = torch.inverse(basis.t() @ basis)
        precision = precision / precision.diag().mean()
        mean = pca.th_hands_mean.detach().float().reshape(45)
        return precision.to(self.device), mean.to(self.device)

    def set_shape(self, mano_betas: Tensor) -> None:
        self.model.betas = mano_betas

    def initial_params(self, joints_world: Tensor) -> Tensor:
        # the mean-shape (betas = 0) flat hand is aligned, as in the paper
        B, dev = joints_world.shape[0], joints_world.device
        betas, self.model.betas = self.model.betas, None
        with torch.no_grad():
            zero = torch.zeros(B, 51, device=dev)
            R, _ = rigid_align(self.model.forward(zero).landmarks, joints_world)
            root = rotation_to_axis_angle(R)
            pose = torch.cat([root, torch.zeros(B, 45, device=dev)], 1)
            rotated = self.model.forward(torch.cat([pose, torch.zeros(B, 3, device=dev)], 1)).landmarks
            translation = (joints_world - rotated).mean(1)
        self.model.betas = betas
        return torch.cat([pose, translation], 1)

    def prior_energies(self, init_params: Tensor) -> list[Energy]:
        P, dev = 51, init_params.device
        finger_precision = torch.zeros(P, P, device=dev)
        finger_precision[3:48, 3:48] = self.prior.finger_pose * self._pca_precision
        finger_ref = torch.zeros(P, device=dev)
        finger_ref[3:48] = self._pca_mean
        anchor = torch.zeros(P, device=dev)
        anchor[0:3] = anchor[48:51] = self.prior.root_anchor
        B = init_params.shape[0]
        return [PosePriorEnergy(finger_ref.expand(B, P), precision=finger_precision.expand(B, P, P)),
                PosePriorEnergy(init_params, precision=anchor)]

    def mano_vertices(self, params: Tensor) -> Tensor:
        return self.model.forward(params).verts


class MappedHand(FittableHand):
    """MHR / UmeTrack fitted through the MANO joint correspondence."""

    def __init__(self, name: str, hand, correspondence: ManoCorrespondence, mano_assets: str,
                 prior: PriorWeights = PriorWeights(), device="cpu"):
        self.name = name
        self.model = ManoConventionHand(hand, correspondence.to(device))
        self.prior = prior
        self.device = torch.device(device)
        self._mano = ManoModel(mano_assets, side="right", center_idx=0, flat_hand_mean=True, joints="regressed")
        self._mano.mano_layer.to(device)
        k = hand.articulation_slice.stop - hand.articulation_slice.start
        with torch.no_grad():
            self._rest_landmarks = self.model.local_landmarks(torch.zeros(1, k, device=device))   # (1,21,3), unit scale

    def set_shape(self, mano_betas: Tensor) -> None:
        """Uniform hand scale that matches the target's segment lengths to MANO's at the predicted betas."""
        B = mano_betas.shape[0]
        self._mano.betas = mano_betas
        with torch.no_grad():
            mano_rest = self._mano.forward(torch.zeros(B, 51, device=mano_betas.device)).landmarks
        self.model.hand_scale = hand_scale_between(self._rest_landmarks.expand(B, -1, -1), mano_rest)

    def initial_params(self, joints_world: Tensor) -> Tensor:
        B = joints_world.shape[0]
        k = self._rest_landmarks.new_zeros(B, self.model.n_params - 6)
        with torch.no_grad():
            rest = self.model.broadcast_scale(joints_world) * self._rest_landmarks.expand(B, -1, -1)
            R, t = rigid_align(rest, joints_world)
        return self.model.join(rotation_to_axis_angle(R), k, t)

    def prior_energies(self, init_params: Tensor) -> list[Energy]:
        P = self.model.n_params
        rest_pull = init_params.new_zeros(P)
        rest_pull[self.model.hand.articulation_slice] = self.prior.finger_pose
        anchor = init_params.new_zeros(P)
        anchor[:3] = anchor[-3:] = self.prior.root_anchor
        return [PosePriorEnergy(torch.zeros_like(init_params), precision=rest_pull),
                PosePriorEnergy(init_params, precision=anchor)]

    def mano_vertices(self, params: Tensor) -> Tensor:
        return self.model.mano_vertices(params)

    def mesh(self, params: Tensor) -> tuple[Tensor, Tensor]:
        """The target model's own mesh (B,V,3) and faces (F,3)."""
        return self.model.mesh(params), self.model.hand.faces


def load_articulated_hand(name: str, side: str = "right", device="cpu", with_mesh: bool = True,
                          mhr_cache: str = HandModelPaths.mhr_cache,
                          umetrack_json: str = HandModelPaths.umetrack_json):
    """The raw MHR / UmeTrack hand (parafit ``ArticulatedHand``), wrist-local, metres."""
    if name == "mhr":
        from parafit.models.mhr import MhrModel
        from parafit.models.mhr_hand import MhrHandModel
        return MhrHandModel(MhrModel.from_cache(mhr_cache, device=str(device)), side=side, full_mesh=with_mesh)
    if name == "umetrack":
        from parafit.models.umetrack import UmeTrackModel, load_umetrack_hand
        return UmeTrackModel(load_umetrack_hand(umetrack_json), side=side, device=str(device), with_mesh=with_mesh)
    raise ValueError(f"unknown articulated hand {name!r}; expected 'mhr' or 'umetrack'")


def build_hand(name: str, paths: HandModelPaths = HandModelPaths(), prior: PriorWeights = PriorWeights(),
               device="cpu") -> FittableHand:
    """The fittable hand ``name`` in {"mano", "mhr", "umetrack"} (right hand; UA-Fit evaluates mirrored right hands)."""
    if name == "mano":
        return ManoHand(paths.mano_assets, prior, device)
    if name not in HAND_MODELS:
        raise ValueError(f"unknown hand model {name!r}; expected one of {HAND_MODELS}")
    correspondence = ManoCorrespondence.load(f"{paths.correspondence_dir}/{name}_right.npz", device=str(device))
    hand = load_articulated_hand(name, "right", device, True, paths.mhr_cache, paths.umetrack_json)
    return MappedHand(name, hand, correspondence, paths.mano_assets, prior, device)
