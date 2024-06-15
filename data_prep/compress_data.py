import numpy as np
import os, time, argparse


def swap_extension_to_npz(filename):
    base, ext = os.path.splitext(filename)
    if ext == ".npy":
        return base + ".npz"
    else:
        raise ValueError("The file does not have a '.npy' extension")


def run_compress(src):
    start_time = time.time()
    dst = swap_extension_to_npz(src)

    subtask_start_time = time.time()
    arr = np.load(src).astype(np.float32)
    elapsed_time = time.time() - subtask_start_time
    print(f"Subtask completed: Loaded [{src}] ... Time: [{elapsed_time:.5f} seconds]")

    subtask_start_time = time.time()
    np.savez_compressed(dst, array=arr)
    elapsed_time = time.time() - subtask_start_time
    print(
        f"Subtask completed: Compressed and saved to [{dst}] ... Time: [{elapsed_time:.5f} seconds]"
    )

    elapsed_time = time.time() - start_time
    print(
        f"Success: Processed [{src}] Stored at [{dst}] ... Time: [{elapsed_time:.5f} seconds]"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)

    args = parser.parse_args()
    run_compress(args.src)
