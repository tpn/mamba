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

    prefix = text_file[:-4]
    last_line_len = 0

    with open(text_file, "w+") as f:
        for line in proc.stdout:
            percent_line = (
                line.startswith('[') and
                '%' in line and
                prefix in line
            )
            # Use carriage return to move cursor to start of line for percent lines
            if percent_line:
                # Move to beginning of line and clear any remaining chars with spaces
                print(f"\r{line.rstrip()}", end="", flush=True)
                # If the new line is shorter than the previous one, add spaces to clear
                if len(line.rstrip()) < last_line_len:
                    print(" " * (last_line_len - len(line.rstrip())), end="", flush=True)
                    print("\r" + line.rstrip(), end="", flush=True)
                last_line_len = len(line.rstrip())
            else:
                print(line, end="", flush=True)
                f.write(line)
                last_line_len = 0  # Reset last_line_len for non-percent lines

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

@pytest.mark.parametrize("scan", ['cuda', 'ref'])
@pytest.mark.parametrize("genlen", [100, 1000, 10000])
@pytest.mark.parametrize("promptlen", [10, 100, 1000])
@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32, 64, 128, 256])
def test_nsys_profile(scan, promptlen, genlen, batch):
    # Obtain a yyyy-mm-dd-hh-mm-ss timestamp using Python.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    out = (f"mamba-bench-{HOST}-{GPU_NAME}-{GPU_RAM}-promptlen{promptlen}-"
           f"genlen{genlen}-b{batch}-scan-{scan}-{timestamp}")
    text_file = f"{out}.txt"
    # Currently unused; copy back in as needed.
    unused = [
        "--python-sampling=true",
        "--stats=true",
        "--trace=cuda,cudnn,cublas",
        "--cudabacktrace=all",
        "--python-backtrace=cuda",
        "--sample=cpu",
        "--cuda-memory-usage=true",
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
        "--scan", scan,
    ]
    run_realtime(cmd, text_file)
