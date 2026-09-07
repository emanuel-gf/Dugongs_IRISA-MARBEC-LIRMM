"""
Map the dataset IDs stored in the json for a json containing the filepath of the tiles. It creates a 
following structure containg th full filepath of the image tiles:
{
  "0": {
    "p5": {
      "aclr": {
        "images": [...],
        "labels": [...],
        "metadata": [...]
      },
      "random": {
        "images": [...],
        "labels": [...],
        "metadata": [...]
      }
    }
  }
}

"""

import os
import json
import glob
import argparse
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser(
        description="Map selected FiftyOne sample IDs to patch filepaths."
    )

    p.add_argument("--dataset", "-d", required=True)
    p.add_argument("--json_path", required=True)
    p.add_argument("--patches_dir", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--port", default="44123")

    return p.parse_args()





## -mine
## MAPDICT
import glob
def get_files_by_stem(filepath_stem, patch_folder):
    dict_out = {}
    foolder_meta = os.path.join(patch_folder, 'metadata')
    list_meta = list(glob.glob(os.path.join(foolder_meta, f'{filepath_stem}__*.json')))
    foolder_meta = os.path.join(patch_folder, 'images')
    list_images = list(glob.glob(os.path.join(foolder_meta, f'{filepath_stem}__*.jpg')))
    foolder_meta = os.path.join(patch_folder, 'labels')
    list_labels = list(glob.glob(os.path.join(foolder_meta, f'{filepath_stem}__*.txt')))
    dict_out['metadata'] = list_meta
    dict_out['label'] = list_labels
    dict_out['images'] = list_images
    return dict_out

def mapdict_patches_filepath(list_paths, patch_folder):
    """
    Map each filepath into three tiles-filepaths respectively to images, labels and metadata. 
    """
    dict_map_filepath = {}
    for path in list_paths:
        stem = Path(path).stem
        dict_map_filepath[stem] = get_files_by_stem(stem, patch_folder)

    ## flat dict
    filepath_all_images   = [f for d in dict_map_filepath.values() for f in d.get('images', [])]
    filepath_all_labels   = [f for d in dict_map_filepath.values() for f in d.get('label', [])]
    filepath_all_metadata = [f for d in dict_map_filepath.values() for f in d.get('metadata', [])]

    ## -------------
    print(f'images:{len(filepath_all_images)}')
    print(f"labels:{len(filepath_all_labels)}")
    print(f"metadata:{len(filepath_all_metadata)}")

    return filepath_all_images, filepath_all_labels, filepath_all_metadata


def main():

    args = get_args()

    # MUST happen before importing fiftyone
    os.environ["FIFTYONE_DATABASE_URI"] = (
        f"mongodb://localhost:{args.port}"
    )

    import fiftyone as fo

    try:
        print(
            f"Connected to MongoDB on port {args.port}"
        )
        print(
            f"Available datasets: {fo.list_datasets()}"
        )

    except Exception as e:
        print(f"Failed to connect:\n{e}")
        return

    assert (
        args.dataset in fo.list_datasets()
    ), f"Dataset '{args.dataset}' not found."

    dataset = fo.load_dataset(args.dataset)

    print(
        f"Loaded dataset '{args.dataset}' "
        f"({len(dataset)} samples)"
    )

    json_path = Path(args.json_path)

    assert json_path.is_file(), (
        f"JSON not found: {json_path}"
    )

    assert Path(args.patches_dir).is_dir(), (f"Patches dir folder do not exist")

    with open(json_path, "r") as f:
        ids_json = json.load(f)

    output = {}

    for seed, seed_dict in ids_json.items():

        print(f"\nSeed: {seed}")

        output[seed] = {
        "train": {},
        "test": {},
        "val": {},
        }

        ## TRAIN
        for partition, partition_dict in seed_dict.items():

            print(f"  Partition: {partition}")

            output[seed]["train"][partition] = {}

            for method, sample_ids in partition_dict.items():

                print(
                    f"    Method: {method} "
                    f"({len(sample_ids)} samples)"
                )

                filepaths = (
                    dataset
                    .select(sample_ids)
                    .values("filepath")
                )
                
                list_images, list_labels, list_metadata = mapdict_patches_filepath(
                    list_paths = filepaths,
                    patch_folder = Path(args.patches_dir)
                )

                output[seed]['train'][partition][method] = {}
                output[seed]['train'][partition][method]['images'] = list_images
                output[seed]['train'][partition][method]['labels'] = list_labels
                output[seed]['train'][partition][method]['metadata'] = list_metadata
           
        #
        # TEST
        #
        test_tag = f"test_{seed}"

        test_view = dataset.match_tags(test_tag)

        print(
            f"TEST | seed={seed} "
            f"| n={len(test_view)}"
        )

        test_filepaths = test_view.values("filepath")

        list_images, list_labels, list_metadata = mapdict_patches_filepath(
            test_filepaths,
            args.patches_dir,
        )
        output[seed]["test"] = {}
        output[seed]['test']['images'] = list_images
        output[seed]['test']['labels'] = list_labels
        output[seed]['test']['metadata'] = list_metadata

        #
        # VAL
        #
        val_tag = f"val_{seed}"

        val_view = dataset.match_tags(val_tag)

        print(
            f"VAL | seed={seed} "
            f"| n={len(val_view)}"
        )

        val_filepaths = val_view.values("filepath")

        list_images, list_labels, list_metadata = mapdict_patches_filepath(
            val_filepaths,
            args.patches_dir,
        )

        output[seed]["val"] = {}
        output[seed]['val']['images'] = list_images
        output[seed]['val']['labels'] = list_labels
        output[seed]['val']['metadata'] = list_metadata
                

    output_path = Path(args.output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"\nSaved output to:\n{output_path}"
    )


if __name__ == "__main__":
    main()