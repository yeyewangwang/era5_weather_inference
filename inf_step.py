import os, time
import numpy as np
import onnx
import onnxruntime as ort


def clear_gpu_memory():
    ort.get_device().reset()


def get_ort_sessions():
    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 1

    # Set the behavier of cuda provider
    cuda_provider_options = {
        "arena_extend_strategy": "kSameAsRequested",
    }
    cuda_providers = [("CUDAExecutionProvider", cuda_provider_options)]
    cpu_providers = ["CPUExecutionProvider"]

    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session_24 = ort.InferenceSession(
        "pangu_weather_24.onnx",
        sess_options=options,
        providers=cuda_providers,
    )
    ort_session_6 = ort.InferenceSession(
        "pangu_weather_6.onnx",
        sess_options=options,
        providers=cuda_providers,
    )
    return {6: ort_session_6, 24: ort_session_24}


def run_inf(data, ort_session):
    start_time = time.time()

    input, input_surface = data

    # Run the inference session
    output, output_surface = ort_session.run(
        None, {"input": input, "input_surface": input_surface}
    )

    elapsed_time = time.time() - start_time
    print(f"Success: Inference completed ... Time: [{elapsed_time:.5f} seconds]")
    return output, output_surface


if __name__ == "__main__":
    # The directory of your input and output data
    input_data_dir = "input_data"

    # Load the upper-air numpy arrays
    start_time = time.time()
    input = np.load(os.path.join(input_data_dir, "input_upper.npy")).astype(np.float32)
    # Load the surface numpy arrays
    input_surface = np.load(os.path.join(input_data_dir, "input_surface.npy")).astype(
        np.float32
    )
    elapsed_time = time.time() - start_time
    print(
        f"Loaded ['input_upper.npy'] and ['input_surface.npy'] ... Time: [{elapsed_time:.5f} seconds]"
    )

    print("Starting inference ...")
    output, output_surface = run_inf([input, input_surface], get_ort_sessions()[24])

    # Save the results
    start_time = time.time()
    output_data_dir = "output_data"
    np.save(os.path.join(output_data_dir, "output_upper"), output)
    np.save(os.path.join(output_data_dir, "output_surface"), output_surface)
    elapsed_time = time.time() - start_time
    print(
        f"Saved ['output_upper.npy'] and ['output_surface.npy'] ... Time: [{elapsed_time:.5f} seconds]"
    )
