import numpy as np
import argparse, os, time
from multiprocessing import Pool


def check(array, name):
    passed = True
    # Check if the data type of the array is np.float32
    if array.dtype != np.float32:
        passed = False
        print(
            f"Data integrity error [{name}] array! \n Expected array dtype [float32], actual dtype: [{array.dtype}]"
        )

    # Check if the array has a specific dimension, e.g., (10, 10)
    expectations = {"surface": (4, 721, 1440), "upper": (5, 13, 721, 1440)}
    s = expectations[name]
    if array.shape != s:
        print(
            f"Data integrity error [{name}] array! \n Expected array dimension [{s}], actual dimension: [{array.shape}]"
        )
        passed = False

    if not phys_check(array, file_type=name):
        passed = False
    return passed


def phys_check(array, file_type, describe=False):
    # variables should follow the following sequence. This check also implicitly check for this.
    # sfc_variables = ["msl", "u10", "v10", "t2m"]
    # pl_variables = ["z", "q", "t", "u", "v"]

    pl_phys_range = {
        "z": [
            f"geopotential below 0 m2/s2 or above 50000 * 9.85 m2/s2",
            1,
            0,
            50000 * 9.85,
        ],
        "q": [
            "specific humidity below 0 kg/kg or above 0.04 kg/kg",
            2,
            0,
            0.04,
        ],
        "t": [
            "temperature below -93.15 C or above 56.85 C",
            3,
            -93.15 + 273.15,
            56.85 + 273.15,
        ],
        "u": [
            "u_component_of_wind below -150 m/s or above 150 m/s",
            4,
            -150,
            150,
        ],
        "v": [
            "v_component_of_wind below -150 m/s or above 150 m/s",
            5,
            -150,
            150,
        ],
    }

    sfc_phys_range = {
        "msl": [
            "mean_sea_level_pressure below 87000 Pa or above 108500 Pa",
            1,
            87000,
            108500,
        ],
        "u10": [
            "10m_u_component_of_wind below -75 m/s or above 75 m/s",
            2,
            -75,
            75,
        ],
        "v10": [
            "10m_v_component_of_wind below -75 m/s or above 75 m/s",
            3,
            -75,
            75,
        ],
        "t2m": [
            "2m_temperature below -60 C or above +60 C",
            4,
            -60 + 273.15,
            60 + 273.15,
        ],
    }

    if file_type == "surface":
        phys_range = sfc_phys_range
    else:
        phys_range = pl_phys_range

    for k, v in phys_range.items():

        msg, id, low, high = v
        idx = id - 1
        below_range = np.sum(array[idx] < low)
        above_range = np.sum(array[idx] > high)

        if describe:
            std = np.std(array[idx])
            print(
                f"{file_type}[{k}] min: {np.min(array[idx]):5f}, max: {np.max(array[idx]):5f}, mean: {np.mean(array[idx]):5f}, std: {std:5f}"
            )

        total_entries = np.prod(array.shape[1:])

        if below_range + above_range > 0:
            print(
                f"Data Integrity Warning: [{file_type}] array may have numbers outside of physical reality! \n variable {file_type}[{k}] out of reality definition: {msg} \n [{below_range}] entries below range and [{above_range}] entries above range out of [{total_entries}] total entries"
            )
            if not describe:
                std = np.std(array[idx])
            print(
                f" Description: min: {np.min(array[idx]):5f}, max: {np.max(array[idx]):5f}, mean: {np.mean(array[idx]):5f}, std: {std:5f}"
            )


def run_check(arrays, names):
    start_time = time.time()
    para_args = list(zip(arrays, names))
    with Pool(processes=2) as pool:
        res = pool.starmap(check, para_args)
    passed = all(res)
    if passed:
        status = f"Success: Passed integrity check"
    else:
        status = f"Completed integrity check"

    elapsed_time = time.time() - start_time
    print(f"{status} ... Time: [{elapsed_time:.5f} seconds]")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)

    args = parser.parse_args()
    # List of filenames to process
    filenames = [
        os.path.join(args.src, f) for f in os.listdir(args.src) if f.endswith(".npy")
    ]

    # Load the array from the .npy file
    arrays = [np.load(f) for f in filenames]
    names = ["surface" if "surface" in f else "upper" for f in filenames]

    res = run_check(arrays, names)
