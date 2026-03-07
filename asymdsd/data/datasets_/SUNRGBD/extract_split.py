import os
import zipfile
from pathlib import Path

import scipy.io as sio

# TODO: USE TEMPDIR


def extract_split(data_path: str | Path):
    with zipfile.ZipFile(Path(data_path) / "SUNRGBDtoolbox.zip", "r") as zip_file:
        with zip_file.open(
            "SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat"
        ) as allsplit_file:
            split = sio.loadmat(allsplit_file)

    # Construct Hash Maps
    # TODO USE SETS INSTEAD
    dict_train = {}
    dict_val = {}

    N_train = len(split["alltrain"][0])
    N_val = len(split["alltest"][0])

    # These are the paths mapped to 0
    for i in range(N_train):
        folder_path = split["alltrain"][0][i][0][16:]
        dict_train[folder_path] = 0

    for i in range(N_val):
        folder_path = split["alltest"][0][i][0][16:]
        dict_val[folder_path] = 0

    # Load metadata
    meta_data = sio.loadmat(Path(data_path) / "SUNRGBDMeta3DBB_v2.mat")
    SUNRGBDMeta = meta_data["SUNRGBDMeta"][0]

    # Ensure output directory exists
    output_dir = Path(data_path) / "sunrgbd_trainval"
    os.makedirs(output_dir, exist_ok=True)

    # Open output files
    with (
        open(os.path.join(output_dir, "train_data_idx.txt"), "w") as fid_train,
        open(os.path.join(output_dir, "val_data_idx.txt"), "w") as fid_val,
    ):
        for imageId in range(len(SUNRGBDMeta)):
            data = SUNRGBDMeta[imageId]
            depthpath = data["depthpath"][0]
            depthpath = depthpath[16:]
            filepath = os.path.dirname(os.path.dirname(depthpath))

            if filepath in dict_train:
                fid_train.write(f"{imageId+1}\n")  # MATLAB indices start from 1
            elif filepath in dict_val:
                fid_val.write(f"{imageId+1}\n")


# THis simply writes a file with imageIDs to save which instances are used for training and which for validation.
