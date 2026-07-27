import argparse
import os

import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial import cKDTree


def connectivity_structure(connectivity):
    if connectivity == 6:
        return np.array([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ], dtype=np.uint8)
    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=np.uint8)
    raise ValueError(f"Unsupported connectivity: {connectivity}")


def ensure_3d(volume):
    return volume[0] if volume.ndim == 4 else volume


def weighted_vote(neighbor_labels, neighbor_distances):
    eps = 1e-8
    scores = {}
    dists = {}
    for lab, dist in zip(neighbor_labels, neighbor_distances):
        lab = int(lab)
        dist = float(dist)
        scores[lab] = scores.get(lab, 0.0) + 1.0 / (dist + eps)
        dists.setdefault(lab, []).append(dist)

    max_score = max(scores.values())
    tied = [lab for lab, score in scores.items() if np.isclose(score, max_score)]
    if len(tied) == 1:
        return tied[0], float(np.mean(dists[tied[0]]))

    # Tie-breaker requested in plan: smallest mean distance.
    best = sorted((float(np.mean(dists[lab])), lab) for lab in tied)[0]
    return int(best[1]), float(best[0])


def sample_pore_voxels(coords, pore_id, max_voxels_per_pore):
    if len(coords) <= max_voxels_per_pore:
        return coords
    rng = np.random.RandomState(int(pore_id))
    idx = rng.choice(len(coords), max_voxels_per_pore, replace=False)
    return coords[idx]


def save_corrected_volume(volume_path, labeled_pores, labels, out_path):
    original = np.load(volume_path)
    vol = ensure_3d(original)
    output = vol.copy()
    output[output == 2] = 0

    for cid in sorted(set(int(x) for x in labels.tolist())):
        pore_indices = np.where(labels == cid)[0] + 1
        output[np.isin(labeled_pores, pore_indices)] = cid

    np.save(out_path, output)
    return out_path


def run_add_back(
    predicted_volume,
    cluster_state,
    outdir,
    min_voxels,
    knn_k,
    connectivity,
    max_voxels_per_pore,
    write_corrected_volume,
):
    state = np.load(cluster_state, allow_pickle=True)

    required_keys = [
        "centroids_3d",
        "labels",
        "labeled_pores",
        "shell_mask",
        "volume_path",
        "pore_voxels",
        "pore_voxels_owner",
        "chamber_voxels",
    ]
    missing = [k for k in required_keys if k not in state]
    if missing:
        raise KeyError(f"Cluster state is missing required keys: {missing}")

    pred = np.load(predicted_volume)
    pred_vol = ensure_3d(pred)

    labeled_pores = state["labeled_pores"]
    if pred_vol.shape != labeled_pores.shape:
        raise ValueError(
            f"Shape mismatch: predicted volume {pred_vol.shape} vs labeled_pores {labeled_pores.shape}"
        )

    structure = connectivity_structure(connectivity)
    pred_labeled, n_pred = ndimage.label(pred_vol == 2, structure=structure)
    pred_sizes = np.bincount(pred_labeled.ravel())

    existing_mask = labeled_pores > 0
    overlap_labels = set(np.unique(pred_labeled[existing_mask]).tolist())
    overlap_labels.discard(0)

    existing_centroids = np.asarray(state["centroids_3d"], dtype=np.float64)
    existing_labels = np.asarray(state["labels"])

    has_reference = len(existing_centroids) > 0 and len(existing_labels) > 0
    tree = cKDTree(existing_centroids) if has_reference else None

    candidates = []
    for comp_label in range(1, n_pred + 1):
        size = int(pred_sizes[comp_label])
        if size < min_voxels:
            continue
        if comp_label in overlap_labels:
            continue

        coords = np.argwhere(pred_labeled == comp_label)
        centroid = coords.mean(axis=0)

        if has_reference:
            k_query = min(int(knn_k), len(existing_centroids))
            dists, idxs = tree.query(centroid, k=k_query)
            if k_query == 1:
                dists = np.array([float(dists)])
                idxs = np.array([int(idxs)])
            else:
                dists = np.asarray(dists, dtype=np.float64)
                idxs = np.asarray(idxs, dtype=np.int64)

            neigh_labels = existing_labels[idxs]
            assigned_label, tie_mean_dist = weighted_vote(neigh_labels, dists)
            nearest_dist = float(np.min(dists))
        else:
            assigned_label = 2
            tie_mean_dist = 0.0
            nearest_dist = 0.0

        candidates.append({
            "pred_label": int(comp_label),
            "size": size,
            "coords": coords,
            "centroid": centroid,
            "assigned_label": int(assigned_label),
            "nearest_dist": nearest_dist,
            "tie_mean_dist": tie_mean_dist,
        })

    new_labeled = labeled_pores.copy()
    next_pid = int(new_labeled.max()) + 1

    added_centroids = []
    added_labels = []
    added_rows = []
    added_voxel_samples = []
    added_voxel_owners = []

    for item in candidates:
        pore_id = next_pid
        next_pid += 1

        coords = item["coords"]
        new_labeled[coords[:, 0], coords[:, 1], coords[:, 2]] = pore_id

        added_centroids.append(item["centroid"])
        added_labels.append(item["assigned_label"])

        sampled = sample_pore_voxels(coords, pore_id, max_voxels_per_pore)
        added_voxel_samples.append(sampled)
        added_voxel_owners.append(np.full(len(sampled), pore_id - 1, dtype=np.int32))

        added_rows.append({
            "pore_id": pore_id,
            "source_component": item["pred_label"],
            "voxels": item["size"],
            "assigned_label": item["assigned_label"],
            "centroid_z": float(item["centroid"][0]),
            "centroid_y": float(item["centroid"][1]),
            "centroid_x": float(item["centroid"][2]),
            "nearest_dist": item["nearest_dist"],
            "tie_mean_dist": item["tie_mean_dist"],
        })

    if added_centroids:
        centroids_out = np.vstack([existing_centroids, np.vstack(added_centroids)])
        labels_out = np.concatenate([existing_labels, np.array(added_labels, dtype=existing_labels.dtype)])
    else:
        centroids_out = existing_centroids.copy()
        labels_out = existing_labels.copy()

    pore_voxels = np.asarray(state["pore_voxels"])
    pore_voxels_owner = np.asarray(state["pore_voxels_owner"])

    if added_voxel_samples:
        vox_new = np.vstack(added_voxel_samples)
        own_new = np.concatenate(added_voxel_owners)
        pore_voxels_out = np.vstack([pore_voxels, vox_new]) if pore_voxels.size else vox_new
        pore_voxels_owner_out = (
            np.concatenate([pore_voxels_owner, own_new]) if pore_voxels_owner.size else own_new
        )
    else:
        pore_voxels_out = pore_voxels.copy()
        pore_voxels_owner_out = pore_voxels_owner.copy()

    os.makedirs(outdir, exist_ok=True)
    state_base = os.path.basename(cluster_state).replace("_cluster_state.npz", "")

    # Keep output tree explicit so editor inputs stay clean and analysis artifacts stay separate.
    editor_state_dir = os.path.join(outdir, "editor_state")
    summary_npz_dir = os.path.join(outdir, "summary_npz")
    summary_tsv_dir = os.path.join(outdir, "summary_tsv")
    corrected_volume_dir = os.path.join(outdir, "corrected_volume")
    for p in (editor_state_dir, summary_npz_dir, summary_tsv_dir):
        os.makedirs(p, exist_ok=True)
    if write_corrected_volume:
        os.makedirs(corrected_volume_dir, exist_ok=True)

    out_state = os.path.join(editor_state_dir, f"{state_base}.npz")
    np.savez_compressed(
        out_state,
        centroids_3d=centroids_out,
        labels=labels_out,
        labeled_pores=new_labeled,
        shell_mask=state["shell_mask"],
        volume_path=state["volume_path"],
        pore_voxels=pore_voxels_out,
        pore_voxels_owner=pore_voxels_owner_out,
        chamber_voxels=state["chamber_voxels"],
    )

    report_npz = os.path.join(summary_npz_dir, f"{state_base}.npz")

    np.savez_compressed(
        report_npz,
        base_name=state_base,
        min_voxels=min_voxels,
        knn_k=knn_k,
        connectivity=connectivity,
        existing_pores=len(existing_labels),
        added_pores=len(added_rows),
        final_pores=len(labels_out),
        assigned_labels=np.array([row["assigned_label"] for row in added_rows], dtype=np.int32),
        assigned_pore_ids=np.array([row["pore_id"] for row in added_rows], dtype=np.int32),
        assigned_sizes=np.array([row["voxels"] for row in added_rows], dtype=np.int32),
    )

    report_tsv = os.path.join(summary_tsv_dir, f"{state_base}.tsv")
    with open(report_tsv, "w", encoding="utf-8") as f:
        f.write(
            "pore_id\tsource_component\tvoxels\tassigned_label\t"
            "centroid_z\tcentroid_y\tcentroid_x\tnearest_dist\ttie_mean_dist\n"
        )
        for row in added_rows:
            f.write(
                f"{row['pore_id']}\t{row['source_component']}\t{row['voxels']}\t"
                f"{row['assigned_label']}\t{row['centroid_z']:.6f}\t{row['centroid_y']:.6f}\t"
                f"{row['centroid_x']:.6f}\t{row['nearest_dist']:.6f}\t{row['tie_mean_dist']:.6f}\n"
            )

    corrected_volume_path = ""
    if write_corrected_volume:
        volume_path = str(state["volume_path"])
        corrected_volume_path = os.path.join(corrected_volume_dir, f"{state_base}.npy")
        save_corrected_volume(volume_path, new_labeled, labels_out, corrected_volume_path)

    print(f"Input state: {cluster_state}")
    print(f"Predicted volume: {predicted_volume}")
    print(f"Existing pores: {len(existing_labels)}")
    print(f"Added pores: {len(added_rows)}")
    print(f"Final pores: {len(labels_out)}")
    print(f"Output state: {out_state}")
    print(f"Summary TSV: {report_tsv}")
    print(f"Summary NPZ: {report_npz}")
    if corrected_volume_path:
        print(f"Corrected volume: {corrected_volume_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add back predicted pore components to cluster_state with configurable size and KNN assignment."
    )
    parser.add_argument("predicted_volume", help="Path to predicted .npy volume")
    parser.add_argument("cluster_state", help="Path to *_cluster_state.npz")
    parser.add_argument(
        "--outdir",
        default="data/corrected_cluster_state",
        help="Directory to save updated cluster_state and reports",
    )
    parser.add_argument("--min-voxels", type=int, default=20, help="Minimum component size to add back")
    parser.add_argument("--knn-k", type=int, default=3, help="K for nearest-neighbor chamber vote")
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[6, 26],
        default=6,
        help="Connected-components neighborhood (default: 6, consistent with current pipeline)",
    )
    parser.add_argument(
        "--max-voxels-per-pore",
        type=int,
        default=30,
        help="Max sampled voxels per added pore for editor rendering",
    )
    parser.add_argument(
        "--write-corrected-volume",
        action="store_true",
        help="Also write corrected labeled volume .npy",
    )
    args = parser.parse_args()

    run_add_back(
        predicted_volume=args.predicted_volume,
        cluster_state=args.cluster_state,
        outdir=args.outdir,
        min_voxels=args.min_voxels,
        knn_k=args.knn_k,
        connectivity=args.connectivity,
        max_voxels_per_pore=args.max_voxels_per_pore,
        write_corrected_volume=args.write_corrected_volume,
    )
