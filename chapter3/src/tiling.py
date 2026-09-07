"""
roi_grid.py
===========

Standalone utility to visualize a shift-inward tile grid on a FiftyOne
dataset. Stores the grid as a Detections field (one box per tile) so you
can inspect tile placement directly in the App.

Usage
-----
    import fiftyone as fo
    from roi_grid import add_roi_grid

    dataset = fo.load_dataset("your_dataset")
    add_roi_grid(dataset, tile_size=1024, overlap=100)

    session = fo.launch_app(dataset)
"""


def compute_tiles(img_w: int, img_h: int, tile_size: int, overlap: int) -> list:
    """
    Returns a list of tiles as absolute pixel rectangles:
        [x_start, y_start, x_end, y_end]  (all integers)

    Shift-inward edge rule: when the stride does not divide the image evenly,
    the last tile in a row/column is shifted LEFT/UP so its right/bottom edge
    aligns with the image boundary, keeping its size exactly
    tile_size x tile_size (no resize, no padding).

        x_end   = min(x + tile_size, img_w)
        x_start = max(0, x_end - tile_size)
    """
    stride = tile_size - overlap
    tiles = []

    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x_end   = min(x + tile_size, img_w)
            y_end   = min(y + tile_size, img_h)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            tiles.append((x_start, y_start, x_end, y_end))

    return tiles


def add_roi_grid(dataset, tile_size: int = 1024, overlap: int = 100,
                  field_name: str = "roi_grid"):
    """
    Computes the shift-inward tile grid for every sample and stores it as a
    fo.Detections field for visualization in the App.

    Each tile is stored as a Detection whose label encodes its TRUE pixel
    origin (tile_<y_start>_<x_start>), and whose bounding_box is normalised
    to [0,1] as required by FiftyOne.

    Parameters
    ----------
    dataset    : fo.Dataset or fo.DatasetView
    tile_size  : int  - square tile side length in pixels (e.g. 1024)
    overlap    : int  - overlap between adjacent tiles in pixels
    field_name : str  - name of the field to store the grid in
    """
    import fiftyone as fo

    # Make sure width/height metadata is populated
    dataset.compute_metadata(skip_failures=True)

    print(f"Adding '{field_name}' (tile={tile_size}px, overlap={overlap}px) ...")
    updated = 0

    for sample in dataset.iter_samples(autosave=True, progress=True):
        if sample.metadata is None or not sample.metadata.width:
            continue

        W = sample.metadata.width
        H = sample.metadata.height
        tiles = compute_tiles(W, H, tile_size, overlap)

        rois = []
        for (x_start, y_start, x_end, y_end) in tiles:
            rois.append(fo.Detection(
                label=f"tile_{y_start}_{x_start}",   # true pixel origin
                bounding_box=[
                    x_start / W,
                    y_start / H,
                    (x_end - x_start) / W,
                    (y_end - y_start) / H,
                ],
            ))

        sample[field_name] = fo.Detections(detections=rois)
        updated += 1

    print(f"  '{field_name}' added to {updated} samples "
          f"({len(tiles) if updated else 0} tiles/image at this resolution).")