import numpy as np
import xarray as xr
import os, time, argparse
from multiprocessing import Pool


def validate_files(files):
    assert len(files) == 2, "There must be exactly two .nc files in the directory."
    assert any(
        file.endswith("_pl.nc") for file in files
    ), "One file must end with '_pl.nc'."
    assert any(
        file.endswith("_sfc.nc") for file in files
    ), "One file must end with '_sfc.nc'."


# Function to process surface variables
def process_surface(filename):
    name = "surface"
    start_time = time.time()
    ds = xr.open_dataset(filename)
    variables = ["msl", "u10", "v10", "t2m"]
    data = np.stack([ds[var].values.astype(np.float32) for var in variables])
    data = np.squeeze(data)
    elapsed_time = time.time() - start_time
    print(f"Success: Processed [{filename}] ... Time: [{elapsed_time:.5f} seconds]")
    return (data, name)


# Function to process upper-air variables
def process_upper(filename):
    name = "upper"
    start_time = time.time()
    ds = xr.open_dataset(filename)
    variables = ["z", "q", "t", "u", "v"]
    levels = [
        1000,
        925,
        850,
        700,
        600,
        500,
        400,
        300,
        250,
        200,
        150,
        100,
        50,
    ]
    data = np.stack(
        [ds[var].sel(level=levels).values.astype(np.float32) for var in variables]
    )
    data = np.squeeze(data)
    elapsed_time = time.time() - start_time
    print(f"Success: Processed [{filename}] ... Time: [{elapsed_time:.5f} seconds]")
    return (data, name)


def process_files(files):
    res = []
    for file in files:
        if file.endswith("_pl.nc"):
            res.append(process_upper(file))
        elif file.endswith("_sfc.nc"):
            res.append(process_surface(file))
    return res


def run_reformat(filenames):
    """
    Retrieve data in the following format
    {
        "surface": ndrray
        "upper": ndarray

    }
    """
    validate_files(filenames)
    return {name: data for data, name in process_files(filenames)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)

    args = parser.parse_args()
    # List of filenames to process
    filenames = [
        os.path.join(args.src, f) for f in os.listdir(args.src) if f.endswith(".nc")
    ]

    for name, data in run_reformat(filenames).items():
        dest_file = os.path.join(args.dest, f"input_{name}.npy")
        np.save(dest_file, data)
