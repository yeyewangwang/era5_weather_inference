import os, time, io
import multiprocessing as mp
from datetime import datetime, timedelta

import boto3
import numpy as np
from data_prep.get_era5 import run_retrieve
from data_prep.reformat_era5_to_npy import run_reformat
from data_prep.integrity_check import run_check
from inf_step import run_inf, get_ort_sessions


def delete_era5(filenames):
    for f in filenames:
        # Delete the ERA5 file
        try:
            os.remove(f)
            print(f"Success: Deleted [{f}]")
        except FileNotFoundError:
            print(f"{f} does not exist.")
        except PermissionError:
            print(f"Permission denied: {f}.")
        except Exception as e:
            print(f"Error occurred while deleting {f}: {e}")


def flush_to_disk(upper, surface, timestamp, sub_dir, is_output):
    start_time = time.time()
    dt_suffix = timestamp.strftime("%m_%Y_%d_%HZ")
    in_or_out = "output" if is_output else "input"

    # Initialize a session using a specific profile
    session = boto3.Session(profile_name="yoyo_ssh")
    # Use the session to create an S3 resource object
    s3 = session.resource("s3")
    # Bucket and object name
    bucket_name = "yyooera5"

    saved_locs = {}
    for name, data in {"surface": surface, "upper": upper}.items():
        filename = f"{dt_suffix}_{in_or_out}_{name}"
        # filedir = os.path.join(f"{in_or_out}_data", sub_dir)
        # filepath = os.path.join(filedir, filename)
        # if not os.path.isdir(filedir):
        #     os.makedirs(filedir, exist_ok=True)

        # saved_locs[name] = filepath
        # np.save(filepath, data)

        # Create an in-memory bytes buffer
        buffer = io.BytesIO()
        np.save(buffer, data)
        # Reset buffer position to the beginning after writing
        buffer.seek(0)

        object_name = f"{sub_dir}/{in_or_out}_data/{filename}.npy"
        # Upload the file-like object to S3
        s3.Object(bucket_name, object_name).put(Body=buffer.getvalue())
        saved_locs[name] = f"s3://{bucket_name}/{object_name}"

    elapsed_time = time.time() - start_time
    print(
        f"Success: Stored [{saved_locs['upper']}] and [{saved_locs['surface']}] ... Time: [{elapsed_time:.5f} seconds]"
    )


class DataBatch:
    def __init__(self, surface, upper, timestamp):
        self.timestamp = timestamp
        self.surface = surface
        self.upper = upper


def prep_process(queue):
    year = "2023"
    month = "12"
    date = "01"
    hour = "00"
    forecast_step_delta = 12
    forecast_steps = 20

    # Starting datetime object for this series of forecasts
    init_base_dt = datetime.strptime(
        f"{year}-{month}-{date}-{hour}:00", "%Y-%m-%d-%H:%M"
    )

    for f_step in range(forecast_steps):

        # Base time for an individual series of forecast
        base_dt = init_base_dt + timedelta(hours=(f_step) * forecast_step_delta)
        base_str = base_dt.strftime("%d_%HZ")

        print(f"Preparing input data for [{base_dt.strftime('%m_%Y_%d_%HZ')}]")

        # Download from internet to EBS volume
        filenames = run_retrieve(
            "../ERA5",
            base_dt.year,
            base_dt.strftime("%m"),
            base_dt.strftime("%d"),
            base_dt.strftime("%H"),
        )

        # Load from EBS volume, reformat
        names_to_data = run_reformat(filenames)
        delete_era5(filenames)

        names = ["upper", "surface"]
        data = [names_to_data[k] for k in names]
        run_check(data, names)

        input, input_surface = data
        # Flush results to an S3 bucket
        flush_to_disk(
            upper=input,
            surface=input_surface,
            timestamp=base_dt,
            sub_dir=base_str,
            is_output=False,
        )

        queue.put(DataBatch(surface=input_surface, upper=input, timestamp=base_dt))
        print(
            f"Data queued up for inference: ... {[{base_dt.strftime('%m_%Y_%d_%HZ')}]}"
        )


def inf_process(queue):
    while True:
        try:
            data_batch = queue.get(timeout=0.1)
            if data_batch is None:
                break

            # sessions = get_ort_sessions()
            # ort_session_24 = sessions[24]
            # ort_session_6 = sessions[6]

            input, input_surface = data_batch.upper, data_batch.surface
            input_24, input_surface_24 = input, input_surface
            base_time = data_batch.timestamp
            base_str = base_time.strftime("%d_%HZ")

            # inf_steps = 20
            inf_steps = 20
            inf_step_delta = 6  # in hours

            for i in range(inf_steps):
                sessions = get_ort_sessions()
                ort_session_24 = sessions[24]
                ort_session_6 = sessions[6]
                target_time = base_time + timedelta(hours=(i + 1) * inf_step_delta)
                print(f"Running inference for [{target_time.strftime('%m_%Y_%d_%HZ')}]")
                if (i + 1) % 4 == 0:
                    output, output_surface = run_inf(
                        [input_24, input_surface_24], ort_session_24
                    )
                    input_24, input_surface_24 = output, output_surface
                else:
                    output, output_surface = run_inf(
                        [input, input_surface], ort_session_6
                    )
                input, input_surface = output, output_surface

                # Run check
                run_check([input, input_surface], ["upper", "surface"])

                # Flush results to an S3 bucket
                flush_to_disk(
                    output,
                    output_surface,
                    target_time,
                    sub_dir=base_str,
                    is_output=True,
                )
        except mp.queues.Empty:
            continue  # Queue is empty, continue checking


if __name__ == "__main__":
    start_time = time.time()
    print("Starting pipelined download and inference")

    data_queue = mp.Queue(
        maxsize=1
    )  # Adjust maxsize based on memory and performance requirements

    downloader_process = mp.Process(target=prep_process, args=(data_queue,))
    inference_process = mp.Process(target=inf_process, args=(data_queue,))

    downloader_process.start()
    inference_process.start()

    downloader_process.join()
    data_queue.put(None)  # Signal the inference process to exit
    inference_process.join()

    elapsed_time = time.time() - start_time
    print(f"Done!  pipelined download and inference ... time [{elapsed_time:.5f}]")
