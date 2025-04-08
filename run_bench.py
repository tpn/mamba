import datetime
import pytest
import socket
import subprocess

HOST = socket.gethostname()

def get_gpu_name():
    cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    raw = res.stdout.decode().splitlines()[0]
    return raw.replace(" ", "-")

def get_gpu_ram():
    cmd = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    mb = int(res.stdout.decode().split()[0])
    return f"{mb // 1024}GB"

GPU_NAME = get_gpu_name()
GPU_RAM = get_gpu_ram()

def run_realtime(cmd, text_file):
    print(f'Running command: {" ".join(cmd)}')
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    with open(text_file, "w+") as f:
        for line in proc.stdout:
            print(line, end="")
            f.write(line)

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

@pytest.mark.parametrize("genlen", [100, 1000, 10000])
@pytest.mark.parametrize("promptlen", [10, 100, 1000])
@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32, 64])
def test_nsys_profile(promptlen, genlen, batch):
    # Skip if promptlen > genlen.
    if promptlen > genlen:
        pytest.skip("Skipping test where promptlen > genlen.")

    # Obtain a yyyy-mm-dd-hh-mm-ss timestamp using Python.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    out = (f"mamba-bench-{HOST}-{GPU_NAME}-{GPU_RAM}-promptlen{promptlen}-"
           f"genlen{genlen}-b{batch}-{timestamp}")
    text_file = f"{out}.txt"
    cmd = [
        "nsys", "profile",
        f"--output={out}",
        "--python-sampling=true",
        "--stats=true",
        "--trace=cuda,cudnn,cublas",
        "--cudabacktrace=all",
        "--python-backtrace=cuda",
        "--sample=cpu",
        "--cuda-memory-usage=true",
        "python", "benchmarks/benchmark_generation_mamba_simple.py",
        "--model-name", "state-spaces/mamba-2.8b",
        "--topp", "0.9",
        "--temperature", "0.7",
        "--repetition-penalty", "1.2",
        "--promptlen", str(promptlen),
        "--genlen", str(genlen),
        "--batch", str(batch),
    ]
    cmd = [
        "nsys", "profile",
        f"--output={out}",
        "--stats=true",
        "--trace=cuda,cudnn,cublas",
        "--cuda-memory-usage=true",
        "--gpu-metrics-devices=cuda-visible",
        "python", "benchmarks/benchmark_generation_mamba_simple.py",
        "--model-name", "state-spaces/mamba-2.8b",
        "--topp", "0.9",
        "--temperature", "0.7",
        "--repetition-penalty", "1.2",
        "--promptlen", str(promptlen),
        "--genlen", str(genlen),
        "--batch", str(batch),
    ]
    run_realtime(cmd, text_file)
