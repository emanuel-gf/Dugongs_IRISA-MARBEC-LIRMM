"""
relabel_calves.py
==================

Reads a metadata CSV (one row per ground-truth box, grouped by an image-name
column) and creates a NEW Detections field that is a copy of an existing
ground-truth field, except that boxes flagged as calves are relabeled to
"calf" (and all other matched boxes set to the species label), based on
the CSV's "calf" column.

The original field is left untouched.

Matching strategy
------------------
The CSV's name column (e.g. "label_name" or "picture_name") holds
"<image_stem>.txt". Multiple rows can share the same name (one row per
box). Rows are matched to existing Detection objects in
field_ground.detections BY COORDINATES:

  - If the CSV has width/height columns: match on full (cx, cy, w, h),
    matching the FiftyOne bounding_box converted to centre format.
  - If the CSV only has x_center/y_center (no width/height): match on
    centroid distance alone. This is weaker -- two very close detections
    in a crowded frame could be ambiguous -- but it's the only option
    when box size isn't recorded in the sheet.

In both cases, the closest unmatched row within `tolerance` is accepted;
each CSV row can only be claimed by one detection.

Usage
-----
    import fiftyone as fo
    from relabel_calves import relabel_calves, relabel_calves_wp

    # New Caledonia (CSV, has width/height)
    dataset = fo.load_dataset("nc_dataset")
    relabel_calves(
        dataset=dataset,
        csv_path="/path/to/nc_metadata.csv",
        name_column="label_name",
        field_ground="ground_truth",
        new_field="ground_truthv2",
        original_label="dugong",
    )

    # West Papua (XLSX, centroid-only, no width/height)
    dataset = fo.load_dataset("wp145")
    relabel_calves_wp(
        dataset=dataset,
        xlsx_path="/path/to/dugong_environmental_variables_WP.xlsx",
        name_column="picture_name",
        field_ground="ground_truth",
        new_field="ground_truthv2",
        original_label="Dugong",
    )
"""

import pandas as pd
import numpy as np
from pathlib import Path


def relabel_calves(
    dataset,
    csv_path: str,
    name_column: str = "label_name",
    field_ground: str = "ground_truth",
    new_field: str = "ground_truthv2",
    new_label: str = "calf",
    original_label: str = "Dugong",
    tolerance: float = 1e-3,
):
    """
    Creates `new_field` on the dataset as a copy of `field_ground`, with
    calf detections relabeled. The source field is never modified.
    Reads metadata from a CSV file.

    Parameters
    ----------
    dataset        : fo.Dataset or fo.DatasetView
    csv_path       : str   - path to the metadata CSV
    name_column    : str   - name of the column holding "<stem>.txt"
                              (e.g. "label_name" for NC, "picture_name" for WP)
    field_ground   : str   - name of the existing Detections field to read from (default "ground_truth")
    new_field      : str   - name of the new Detections field to create (default "ground_truthv2")
    new_label      : str   - label to assign when calf == "yes" (default "calf")
    original_label : str   - label to assign to non-calf matched detections (default "Dugong")
    tolerance      : float - max allowed Euclidean distance (in normalised
                              coordinate space) for a Detection<->CSV-row
                              match to be accepted (default 1e-3)
    """
    df = pd.read_csv(csv_path)
    _relabel_calves_from_df(
        dataset=dataset,
        df=df,
        name_column=name_column,
        field_ground=field_ground,
        new_field=new_field,
        new_label=new_label,
        original_label=original_label,
        tolerance=tolerance,
    )


def relabel_calves_wp(
    dataset,
    xlsx_path: str,
    sheet_name=0,
    name_column: str = "picture_name",
    field_ground: str = "ground_truth",
    new_field: str = "ground_truthv2",
    new_label: str = "calf",
    original_label: str = "Dugong",
    tolerance: float = 1e-3,
):
    """
    Same as relabel_calves, but reads metadata from an Excel (.xlsx) file
    instead of a CSV. West Papua's metadata sheet is distributed as .xlsx,
    which pd.read_csv cannot parse (it will raise a UnicodeDecodeError
    since the file is a binary Excel container, not plain text).

    Parameters
    ----------
    dataset        : fo.Dataset or fo.DatasetView
    xlsx_path      : str   - path to the metadata .xlsx file
    sheet_name     : str or int - sheet to read (default 0, the first sheet)
    name_column    : str   - name of the column holding "<stem>.txt"
                              (default "picture_name", matching WP's sheet)
    field_ground   : str   - name of the existing Detections field to read from (default "ground_truth")
    new_field      : str   - name of the new Detections field to create (default "ground_truthv2")
    new_label      : str   - label to assign when calf == "yes" (default "calf")
    original_label : str   - label to assign to non-calf matched detections (default "Dugong")
    tolerance      : float - max allowed Euclidean distance (in normalised
                              coordinate space) for a Detection<->CSV-row
                              match to be accepted (default 1e-3)
    """
    # openpyxl is the engine pandas needs for .xlsx; raise a clear error
    # up front rather than a confusing one from inside read_excel.
    try:
        import openpyxl  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Reading .xlsx files requires 'openpyxl'. Install it with: "
            "pip install openpyxl --break-system-packages"
        ) from e

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
    _relabel_calves_from_df(
        dataset=dataset,
        df=df,
        name_column=name_column,
        field_ground=field_ground,
        new_field=new_field,
        new_label=new_label,
        original_label=original_label,
        tolerance=tolerance,
    )


def _relabel_calves_from_df(
    dataset,
    df: pd.DataFrame,
    name_column: str,
    field_ground: str,
    new_field: str,
    new_label: str,
    original_label: str,
    tolerance: float,
):
    """
    Shared matching + relabeling logic, given an already-loaded DataFrame
    (regardless of whether it came from a .csv or .xlsx source).
    """
    import fiftyone as fo

    required_cols = {name_column, "calf", "x_center", "y_center"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata table is missing required column(s): {missing}")

    has_size_cols = {"width", "height"}.issubset(df.columns)
    match_cols = ["x_center", "y_center", "width", "height"] if has_size_cols \
                 else ["x_center", "y_center"]

    print(f"Matching mode: {'(cx, cy, w, h)' if has_size_cols else '(cx, cy) centroid-only'} "
          f"-- {'width/height found' if has_size_cols else 'width/height NOT in metadata'}")

    # Stem to match against sample filepaths, e.g.
    # "GH034207-611a6e787d0af_1771.txt" -> "GH034207-611a6e787d0af_1771"
    df["stem_filepath"] = df[name_column].astype(str).apply(lambda x: Path(x).stem)

    # "is this row a calf box" -- driven purely by the calf column
    df["is_calf_row"] = df["calf"].astype(str).str.strip().str.lower() == "yes"

    print(f"Loaded {len(df)} rows across {df['stem_filepath'].nunique()} unique label files.")

    updated_samples    = 0
    updated_detections = 0
    skipped_mismatch    = 0
    copied_unchanged    = 0
    empty_ground        = 0
    mismatch_examples   = []   # collect a few examples for debugging

    for sample in dataset.iter_samples(autosave=True, progress=True):
        source_detections_obj = sample[field_ground]

        if not source_detections_obj or not source_detections_obj.detections:
            empty_ground += 1
            continue

        stem = Path(sample.filepath).stem

        # Always build a fresh copy of the detections so the new field is
        # fully independent from the source field (no shared references).
        new_detections = [
            fo.Detection(
                label=det.label,
                bounding_box=list(det.bounding_box),
                confidence=det.confidence,
                **{
                    k: v for k, v in det.iter_fields()
                    if k not in ("label", "bounding_box", "confidence", "id", "tags", "attributes")
                },
            )
            for det in source_detections_obj.detections
        ]
        # Preserve tags per-detection (iter_fields above excludes them)
        for new_det, old_det in zip(new_detections, source_detections_obj.detections):
            new_det.tags = list(old_det.tags)

        # All metadata rows belonging to this specific sample
        rows = df.loc[df["stem_filepath"] == stem]

        if rows.empty:
            # No match -- copy the field as-is under the new name
            sample[new_field] = fo.Detections(detections=new_detections)
            copied_unchanged += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append((stem, "NO_METADATA_MATCH", len(new_detections), None))
            continue

        if len(rows) != len(new_detections):
            if len(mismatch_examples) < 5:
                mismatch_examples.append((stem, "COUNT_MISMATCH", len(new_detections), len(rows)))
            sample[new_field] = fo.Detections(detections=new_detections)
            skipped_mismatch += 1
            continue

        # Build matching coordinates for each existing Detection. FiftyOne
        # stores bounding_box as [top_left_x, top_left_y, w, h] -- convert
        # to centre format to match the metadata's convention.
        if has_size_cols:
            det_coords = np.array([
                [
                    det.bounding_box[0] + det.bounding_box[2] / 2,  # cx
                    det.bounding_box[1] + det.bounding_box[3] / 2,  # cy
                    det.bounding_box[2],                            # w
                    det.bounding_box[3],                            # h
                ]
                for det in new_detections
            ])
        else:
            det_coords = np.array([
                [
                    det.bounding_box[0] + det.bounding_box[2] / 2,  # cx
                    det.bounding_box[1] + det.bounding_box[3] / 2,  # cy
                ]
                for det in new_detections
            ])

        row_coords  = rows[match_cols].to_numpy()
        row_is_calf = rows["is_calf_row"].to_numpy()

        sample_changed   = False
        matched_row_idxs = set()

        for i, det in enumerate(new_detections):
            # Find the closest unmatched metadata row to this detection
            diffs = np.linalg.norm(row_coords - det_coords[i], axis=1)
            for used_idx in matched_row_idxs:
                diffs[used_idx] = np.inf   # don't let two detections claim the same row

            best_idx  = int(np.argmin(diffs))
            best_diff = diffs[best_idx]

            if best_diff > tolerance:
                # No sufficiently close match for this detection
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(
                        (stem, f"NO_COORD_MATCH (closest diff={best_diff:.5f})",
                         len(new_detections), len(rows))
                    )
                continue

            matched_row_idxs.add(best_idx)

            new_assigned_label = new_label if row_is_calf[best_idx] else original_label
            if det.label != new_assigned_label:
                det.label = new_assigned_label
                updated_detections += 1
                sample_changed = True

        sample[new_field] = fo.Detections(detections=new_detections)
        if sample_changed:
            updated_samples += 1

    print(f"\nDone.")
    print(f"  New field created       : '{new_field}'")
    print(f"  Samples with relabeling : {updated_samples}")
    print(f"  Detections relabeled     : {updated_detections}")
    print(f"  Samples copied unchanged (no metadata match): {copied_unchanged}")
    print(f"  Samples with empty/None '{field_ground}' (skipped entirely): {empty_ground}")
    if skipped_mismatch:
        print(f"  Copied unchanged due to count mismatch: {skipped_mismatch} "
              f"label file(s) -- check these manually.")

    if mismatch_examples:
        print(f"\n  --- First {len(mismatch_examples)} examples for debugging ---")
        for stem, reason, n_det, n_csv in mismatch_examples:
            print(f"    [{reason}] stem='{stem}'  n_detections={n_det}  n_csv_rows={n_csv}")