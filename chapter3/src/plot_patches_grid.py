"""
plot_patches_grid.py
=======================

Quick visual sanity-check for object/patch crops: shows a square grid of
cropped detection images via plain imshow, no labels or annotations drawn
on the crops themselves (just the title above each subplot).

Usage
-----
    from plot_patches_grid import plot_patches

    # Random sample of 9 patches from the dataset
    plot_patches(dataset, num_objects=9, patches_field="ground_truth")

    # Specific patches, e.g. retrieved from a to_patches() view
    pairs = [(s.sample_id, s.id) for s in
             dataset.to_patches("ground_truth").take(6)]
    plot_patches(dataset, patches_field="ground_truth", patch_list=pairs)
"""

import random
import math

import matplotlib.pyplot as plt
from PIL import Image


def plot_patches(
    dataset,
    patches_field: str = "ground_truth",
    num_objects: int = 9,
    patch_list: list | None = None,
    figsize_per_cell: float = 2.8,
    seed: int | None = None,
):
    """
    Plots a grid of cropped detection images (plain imshow, no boxes/labels
    drawn -- just the raw crop content, with a small title per subplot).

    Parameters
    ----------
    dataset          : fo.Dataset or fo.DatasetView
    patches_field    : str -- name of the Detections field to crop from
                        (default "ground_truth")
    num_objects      : int -- how many RANDOM patches to plot if patch_list
                        is not given. Grid is the smallest square that fits
                        num_objects (e.g. 9 -> 3x3, 10 -> 4x4 with 6 empty cells).
    patch_list       : list[tuple(sample_id, detection_id)] or None -- if
                        given, plots exactly these patches in this order
                        instead of a random sample. Easily obtained via:
                            [(s.sample_id, s.id) for s in
                             dataset.to_patches(patches_field).take(N)]
    figsize_per_cell : float -- inches per grid cell (controls overall figure size)
    seed             : int or None -- random seed for reproducible random sampling
                        (only used when patch_list is None)

    Returns
    -------
    fig, axes
    """
    if seed is not None:
        random.seed(seed)

    # ── Resolve which (sample_id, detection_id) pairs to plot ───────────────
    if patch_list is not None:
        pairs = list(patch_list)
    else:
        # Build the full list of (sample_id, detection_id) pairs that exist,
        # then randomly sample num_objects of them.
        all_pairs = []
        for sample in dataset.select_fields(patches_field).iter_samples():
            det_obj = sample[patches_field]
            if det_obj and det_obj.detections:
                for det in det_obj.detections:
                    all_pairs.append((sample.id, det.id))

        if not all_pairs:
            raise ValueError(
                f"No detections found in field '{patches_field}' across the "
                f"given dataset/view -- nothing to plot."
            )

        n = min(num_objects, len(all_pairs))
        if n < num_objects:
            print(f"  Only {len(all_pairs)} patches available, plotting {n} "
                  f"instead of the requested {num_objects}.")
        pairs = random.sample(all_pairs, n)

    n_plots = len(pairs)
    grid_size = math.ceil(math.sqrt(n_plots))

    fig, axes = plt.subplots(
        grid_size, grid_size,
        figsize=(figsize_per_cell * grid_size, figsize_per_cell * grid_size),
    )
    axes = axes.flatten() if n_plots > 1 else [axes]

    for ax, (sample_id, detection_id) in zip(axes, pairs):
        sample = dataset[sample_id]
        det_obj = sample[patches_field]

        detection = next(
            (d for d in det_obj.detections if d.id == detection_id), None
        ) if det_obj else None

        if detection is None:
            ax.set_title(f"NOT FOUND\n{detection_id[:8]}", fontsize=8, color="red")
            ax.axis("off")
            continue

        with Image.open(sample.filepath) as img:
            img_w, img_h = img.size
            bx, by, bw, bh = detection.bounding_box
            x1 = max(0, int(bx * img_w))
            y1 = max(0, int(by * img_h))
            x2 = min(img_w, int((bx + bw) * img_w))
            y2 = min(img_h, int((by + bh) * img_h))
            crop = img.convert("RGB").crop((x1, y1, x2, y2))

        ax.imshow(crop)
        ax.set_title(f"{detection.label}\n{sample_id[:8]}...", fontsize=8)
        ax.axis("off")

    # Turn off any unused trailing axes (e.g. 9 patches in a 4x4 grid)
    for ax in axes[n_plots:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return fig, axes