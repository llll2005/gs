"""
C3: Boundary Graph Construction for DT-ADMM-GAT

Builds the graph G=(V, E) where:
  V = boundary Gaussians (those within d_boundary of any adjacent block face)
  E = cross-block KNN edges (scipy cKDTree, bidirectional)

Used by C4 (GAT z-update) to propagate geometric consensus across block boundaries.

Design decisions (from Gemini evaluation, see 紀錄/C3_boundary_graph_spec.md):
  - cKDTree over Morton code: Morton's Z-order jumps make CPU KNN slower than cKDTree
  - Fixed d_boundary (not relative): relative sizing causes asymmetric graph → ADMM collapse
  - Spectral normalization on GAT W: enforced in C4, not here
  - L_z design confirmed correct: z*_v = x_v + y_v/rho is dynamic supervision target
  - block_aabbs MUST come from partitions.pt (Gaussian positions drift far outside their
    original block during training, making position-derived AABBs useless for boundary detection)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree


# ── Partition AABB loader ─────────────────────────────────────────────────────

def load_partition_aabbs(partitions_pt_path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Load per-block XY AABBs from the partitions.pt file produced by partition_from_colmap.py.

    Returns {block_id: (min_xy, max_xy)} where min/max_xy are shape-(2,) float32 arrays.
    The block_id is the flat partition index (0..N-1) used by train_citygs_partitions.py.

    AABB is computed as: center ± size/2  (XY only; Z ignored for aerial scenes).
    """
    pt   = torch.load(partitions_pt_path, map_location="cpu", weights_only=False)
    xy   = pt["partition_coordinates"]["xy"].numpy().astype(np.float32)   # (N, 2)
    size = pt["scene_config"]["partition_size"].numpy().astype(np.float32) # (N, 2)

    aabbs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for bid in range(len(xy)):
        half  = size[bid] / 2.0
        aabbs[bid] = (xy[bid] - half, xy[bid] + half)
    return aabbs


# ── Checkpoint feature extractor ─────────────────────────────────────────────

def load_block_features(ckpt_path: str, device: str = "cpu") -> Optional[torch.Tensor]:
    """
    Load a (N, 12) feature tensor from a block checkpoint.

    Feature layout (matches DT-ADMM-GAT node feature spec):
      [0:3]  mu       — Gaussian centre (world coords, raw)
      [3:6]  normal   — surfel normal (z-axis of rotation matrix, unit vector)
      [6:8]  scale_2d — surfel scale (exp of stored log-scale)
      [8:9]  opacity  — opacity (sigmoid of stored logit)
      [9:12] SH_dc    — DC spherical harmonic (raw, shape (1,3) squeezed)

    Quaternion convention (from internal/utils/general_utils.py::build_rotation):
      q = [w, x, y, z]  →  normal = R[:, :, 2]  (third column of rotation matrix)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd   = ckpt.get("state_dict", ckpt)

    prefix = "gaussian_model.gaussians."
    try:
        means    = sd[prefix + "means"]       # (N, 3)
        quats    = sd[prefix + "rotations"]   # (N, 4) [w, x, y, z]
        log_s    = sd[prefix + "scales"]      # (N, 2) log-scale
        logit_o  = sd[prefix + "opacities"]   # (N, 1) logit
        shs_dc   = sd[prefix + "shs_dc"]      # (N, 1, 3)
    except KeyError as e:
        raise KeyError(f"Unexpected checkpoint format in {ckpt_path}: {e}")

    # Normal = third column of rotation matrix R
    # build_rotation convention: q[:,0]=w, q[:,1]=x, q[:,2]=y, q[:,3]=z
    q = quats / (quats.norm(dim=1, keepdim=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # R[:, :, 2] = [2(xz+wy), 2(yz-wx), 1-2(x²+y²)]
    nx = 2 * (x * z + w * y)
    ny = 2 * (y * z - w * x)
    nz = 1 - 2 * (x * x + y * y)
    normals = torch.stack([nx, ny, nz], dim=1)  # (N, 3), already normalised

    scales   = torch.exp(log_s)                 # (N, 2)
    opacities = torch.sigmoid(logit_o)          # (N, 1)
    sh_dc    = shs_dc[:, 0, :]                  # (N, 3)

    return torch.cat([means, normals, scales, opacities, sh_dc], dim=1).to(device)  # (N, 12)


# ── BoundaryGraph ─────────────────────────────────────────────────────────────

@dataclass
class BoundaryGraphConfig:
    grid_dim: Tuple[int, int] = (5, 5)
    d_boundary: float = 0.1
    """Width of the boundary band in COLMAP units (≈5 m for MatrixCity aerial).
    Fixed global value ensures the graph is symmetric across block pairs (see spec §A2)."""
    aabb_dilation: float = 0.0
    """Extra margin added to partition AABB before in-AABB filtering.
    Set to 3*d_boundary for ADMM iteration > 0: Gaussians held near z_v by the AL
    penalty are allowed to be slightly outside their partition AABB."""
    k_neighbors: int = 8
    """KNN fan-out per boundary Gaussian toward each adjacent block."""
    connectivity: str = "4"
    """'4' = cardinal only (recommended for aerial grid); '8' = include diagonals."""
    d_max_edge_factor: float = 6.0
    """Edges with dist > d_boundary * factor are discarded.
    Must cover face_gap + 2*d_boundary (face_gap ≈ 0-0.3 for MatrixCity aerial)."""
    normal_cos_threshold: float = 0.5
    """cos(60°). Edges whose normals diverge more than 60° are discarded."""


class BoundaryGraph:
    """
    Builds and caches the cross-block boundary graph for one ADMM outer iteration.

    Typical usage (Plan A – offline graph, see spec §3.1):
        bg = BoundaryGraph(BoundaryGraphConfig())
        feats = {bid: load_block_features(ckpt_path) for bid, ckpt_path in ckpts.items()}
        graph = bg.build(feats)
        # → pass graph["edge_index"] and graph["node_features"] to C4 GAT
    """

    def __init__(
        self,
        config: BoundaryGraphConfig,
        block_aabbs: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        self.config     = config
        self.block_aabbs = block_aabbs  # required; use load_partition_aabbs() to obtain
        self.adjacency  = self._compute_adjacency()

    # ── Adjacency ─────────────────────────────────────────────────────────────

    def _compute_adjacency(self) -> Dict[int, List[int]]:
        rows, cols = self.config.grid_dim
        adj: Dict[int, List[int]] = {}
        for r in range(rows):
            for c in range(cols):
                bid  = r * cols + c
                nbrs = []
                if r > 0:        nbrs.append((r - 1) * cols + c)
                if r < rows - 1: nbrs.append((r + 1) * cols + c)
                if c > 0:        nbrs.append(r * cols + (c - 1))
                if c < cols - 1: nbrs.append(r * cols + (c + 1))
                if self.config.connectivity == "8":
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            nbrs.append(nr * cols + nc)
                adj[bid] = nbrs
        return adj

    # ── AABB helpers ──────────────────────────────────────────────────────────

    def _get_aabb(self, bid: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.block_aabbs is None or bid not in self.block_aabbs:
            raise ValueError(
                f"block_aabbs missing for bid={bid}. "
                "Pass block_aabbs=load_partition_aabbs(partitions_pt_path) to BoundaryGraph. "
                "Do NOT derive AABBs from Gaussian positions — they drift far outside the "
                "original block during training and will produce ~0% boundary nodes."
            )
        return self.block_aabbs[bid]

    # ── Boundary identification ────────────────────────────────────────────────

    def identify_boundary_gaussians(
        self, bid: int, positions: np.ndarray
    ) -> np.ndarray:
        """
        Boolean mask: True for Gaussians that are (a) within the partition AABB and
        (b) within d_boundary of an adjacent-block face.

        Condition (a) is required because Gaussians drift far outside their original
        partition AABB during independent block training. Without it, ~90%+ of
        Gaussians are flagged and the graph becomes meaningless.
        """
        min_xyz, max_xyz = self._get_aabb(bid)
        d    = self.config.d_boundary
        rows, cols = self.config.grid_dim
        r, c = bid // cols, bid % cols

        # (a) Within partition AABB (XY only), optionally dilated for ADMM iter > 0
        d_dil = self.config.aabb_dilation
        in_aabb = (
            (positions[:, 0] >= min_xyz[0] - d_dil) & (positions[:, 0] <= max_xyz[0] + d_dil) &
            (positions[:, 1] >= min_xyz[1] - d_dil) & (positions[:, 1] <= max_xyz[1] + d_dil)
        )

        # (b) Near a shared face (Gemini §A2: symmetric definition)
        near_face = np.zeros(len(positions), dtype=bool)
        if r > 0:        near_face |= positions[:, 1] >= max_xyz[1] - d  # top Y face
        if r < rows - 1: near_face |= positions[:, 1] <= min_xyz[1] + d  # bottom Y face
        if c > 0:        near_face |= positions[:, 0] <= min_xyz[0] + d  # left X face
        if c < cols - 1: near_face |= positions[:, 0] >= max_xyz[0] - d  # right X face

        return in_aabb & near_face

    # ── Main build ────────────────────────────────────────────────────────────

    def build(self, block_gaussians: Dict[int, torch.Tensor]) -> Dict:
        """
        Build the boundary graph from per-block Gaussian feature tensors.

        Args:
            block_gaussians: {block_id: Tensor(N_i, 12)}
                Produced by load_block_features(); columns defined in that docstring.

        Returns dict with keys:
            edge_index      (2, E) long   — cross-block edges (bidirectional)
            node_features   (N_bnd, 12)   — boundary Gaussian features
            node_block_id   (N_bnd,) long — which block each node belongs to
            node_local_idx  (N_bnd,) long — index within that block's Gaussian array
            block_offsets   dict[int,int] — block_id → start index in node_features
            boundary_masks  dict[int, ndarray[bool]] — for debugging / C5 topology
        """
        block_np = {bid: g.cpu().numpy() for bid, g in block_gaussians.items()}

        # 1) Boundary masks
        boundary_masks: Dict[int, np.ndarray] = {
            bid: self.identify_boundary_gaussians(bid, g[:, :3])
            for bid, g in block_np.items()
        }

        # 2) Flatten boundary nodes into contiguous arrays
        all_feats:  List[np.ndarray] = []
        all_bids:   List[np.ndarray] = []
        all_lidxs:  List[np.ndarray] = []
        block_offsets: Dict[int, int] = {}
        offset = 0

        for bid in sorted(block_np.keys()):
            block_offsets[bid] = offset
            mask = boundary_masks[bid]
            if not mask.any():
                continue
            local_idxs = np.where(mask)[0].astype(np.int64)
            all_feats.append(block_np[bid][mask])
            all_bids.append(np.full(mask.sum(), bid, dtype=np.int64))
            all_lidxs.append(local_idxs)
            offset += int(mask.sum())

        if not all_feats:
            return self._empty_graph()

        node_feats = np.concatenate(all_feats,  axis=0)  # (N_bnd, 12)
        node_bids  = np.concatenate(all_bids,   axis=0)  # (N_bnd,)
        node_lidxs = np.concatenate(all_lidxs,  axis=0)  # (N_bnd,)

        # Pre-build per-block global index lookup (avoids repeated np.where)
        bid_to_global: Dict[int, np.ndarray] = {
            bid: np.where(node_bids == bid)[0]
            for bid in block_np
        }

        # 3) KNN edges across adjacent block pairs
        d_max      = self.config.d_boundary * self.config.d_max_edge_factor
        cos_thresh = self.config.normal_cos_threshold
        edges_src: List[int] = []
        edges_dst: List[int] = []
        processed: set = set()

        for bid_i in sorted(block_np.keys()):
            for bid_j in self.adjacency.get(bid_i, []):
                if bid_j not in block_np:
                    continue
                pair = (min(bid_i, bid_j), max(bid_i, bid_j))
                if pair in processed:
                    continue
                processed.add(pair)

                g_i = bid_to_global[bid_i]  # global indices for block i boundary nodes
                g_j = bid_to_global[bid_j]
                if len(g_i) == 0 or len(g_j) == 0:
                    continue

                pos_i = node_feats[g_i, :3]    # (N_bi, 3)
                pos_j = node_feats[g_j, :3]    # (N_bj, 3)
                nrm_i = node_feats[g_i, 3:6]   # (N_bi, 3)
                nrm_j = node_feats[g_j, 3:6]   # (N_bj, 3)

                # KNN from BOTH sides so every boundary node sees its k nearest
                # neighbors — one-sided KNN is asymmetric and misses many edges.
                edge_set: set = set()  # dedup within this pair

                for (g_src, pos_src, nrm_src, g_dst, pos_dst, nrm_dst) in [
                    (g_i, pos_i, nrm_i, g_j, pos_j, nrm_j),
                    (g_j, pos_j, nrm_j, g_i, pos_i, nrm_i),
                ]:
                    k = min(self.config.k_neighbors, len(g_dst))
                    tree = cKDTree(pos_dst)
                    dists, nn_local = tree.query(pos_src, k=k)

                    if k == 1:
                        dists    = dists[:, None]
                        nn_local = nn_local[:, None]

                    for li_src, (dist_row, nn_row) in enumerate(zip(dists, nn_local)):
                        for dist, li_dst in zip(dist_row, nn_row):
                            if dist > d_max:
                                continue
                            cos_ang = float(np.dot(nrm_src[li_src], nrm_dst[li_dst]))
                            if cos_ang < cos_thresh:
                                continue
                            s = int(g_src[li_src])
                            d_ = int(g_dst[int(li_dst)])
                            key = (min(s, d_), max(s, d_))
                            if key not in edge_set:
                                edge_set.add(key)
                                edges_src += [s, d_]
                                edges_dst += [d_, s]

        if not edges_src:
            print("[BoundaryGraph] WARNING: no cross-block edges found. "
                  "Check d_boundary or block adjacency.")
            return self._empty_graph()

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

        # Diagnostics
        n_bnd  = len(node_feats)
        n_edge = edge_index.shape[1]
        pcts   = {
            bid: boundary_masks[bid].sum() / max(len(block_np[bid]), 1) * 100
            for bid in block_np
        }
        print(
            f"[BoundaryGraph] nodes={n_bnd:,}  edges={n_edge:,}  "
            f"boundary%: min={min(pcts.values()):.1f}  "
            f"max={max(pcts.values()):.1f}  "
            f"mean={np.mean(list(pcts.values())):.1f}"
        )

        return {
            "edge_index":     edge_index,
            "node_features":  torch.tensor(node_feats, dtype=torch.float32),
            "node_block_id":  torch.tensor(node_bids,  dtype=torch.long),
            "node_local_idx": torch.tensor(node_lidxs, dtype=torch.long),
            "block_offsets":  block_offsets,
            "boundary_masks": boundary_masks,
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_graph() -> Dict:
        return {
            "edge_index":     torch.zeros(2, 0, dtype=torch.long),
            "node_features":  torch.zeros(0, 12, dtype=torch.float32),
            "node_block_id":  torch.zeros(0, dtype=torch.long),
            "node_local_idx": torch.zeros(0, dtype=torch.long),
            "block_offsets":  {},
            "boundary_masks": {},
        }

    def boundary_stats(self, block_gaussians: Dict[int, torch.Tensor]) -> Dict:
        """Dry-run: return per-block boundary statistics without building edges."""
        stats = {}
        for bid, g in block_gaussians.items():
            pos  = g[:, :3].cpu().numpy()
            mask = self.identify_boundary_gaussians(bid, pos)
            n_total    = len(pos)
            n_boundary = int(mask.sum())
            stats[bid] = {
                "n_total":    n_total,
                "n_boundary": n_boundary,
                "pct":        n_boundary / max(n_total, 1) * 100,
            }
        return stats
