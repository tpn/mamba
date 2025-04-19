#!/usr/bin/env python3

try:
    import cuda.parallel.experimental.algorithms
    print('cuda.parallel.experimental.algorithms imported successfully')
except ImportError:
    print('Error! cuda.parallel.experimental.algorithms not found!')

import argparse
import datetime
import itertools
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
        print(f"Command failed with return code {proc.returncode}")
        #raise subprocess.CalledProcessError(proc.returncode, cmd)

def run_benchmark(scan, promptlen, genlen, batch, with_nsys=False):
    # Obtain a yyyy-mm-dd-hh-mm-ss timestamp using Python.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    prefix = "nsys-mamba-bench" if with_nsys else "mamba-bench"
    out = (f"{prefix}-{HOST}-{GPU_NAME}-{GPU_RAM}-"
           f"promptlen{promptlen}-genlen{genlen}-b{batch}-"
           f"scan-{scan}-{timestamp}")
    text_file = f"{out}.txt"

    cmd = []

    if with_nsys:
        cmd.extend([
            "nsys", "profile",
            f"--output={out}",
            "--stats=true",
            "--trace=cuda,cudnn,cublas",
            "--cuda-memory-usage=true",
            "--gpu-metrics-devices=cuda-visible",
        ])

    cmd.extend([
        "python", "benchmarks/benchmark_generation_mamba_simple.py",
        "--model-name", "state-spaces/mamba-2.8b",
        "--topp", "0.9",
        "--temperature", "0.7",
        "--repetition-penalty", "1.2",
        "--promptlen", str(promptlen),
        "--genlen", str(genlen),
        "--batch", str(batch),
        "--scan", scan,
    ])

    run_realtime(cmd, text_file)

def parse_comma_separated_values(values_str):
    if ',' not in values_str:
        return [values_str.strip()]
    return [val.strip() for val in values_str.split(',')]

def parse_int_list(values_str):
    if ',' not in values_str:
        return [int(values_str.strip())]
    return [int(val.strip()) for val in values_str.split(',')]


SCAN_OPTIONS = [
    "cuda",
    "cuda2",
    "ref",
    "torch",
    "torch-cudaparallel",
    "cudaparallel",
]


def main():
    parser = argparse.ArgumentParser(
        description='Run Mamba benchmarks',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--promptlen', type=str, default='10',
                        help='Single or comma-separated list of prompt lengths to benchmark')
    parser.add_argument('--genlen', type=str, default='10',
                        help='Single or comma-separated list of generation lengths to benchmark')
    parser.add_argument('--batch', type=str, default='1',
                        help='Single or comma-separated list of batch sizes to benchmark')
    parser.add_argument('--with-nsys', action='store_true',
                        help='Run with Nsys profiling')
    parser.add_argument("--scan", type=str, choices=SCAN_OPTIONS, default="cuda",
                        help=(
                            "Selective scan implementation to use:\n"
                            "   cuda (selective_scan_cuda),\n"
                            "   cuda2 (selective_scan2_cuda),\n"
                            "   ref (reference implementation),\n"
                            "   torch (pytorch associative scan wrapper),\n"
                            "   torch-cudaparallel (actual PyTorch associative scan),\n"
                            "   cudaparallel (cuda.parallel associative scan)\n"
                        ))

    args = parser.parse_args()

    # Parse comma-separated values
    scans = [args.scan]  # Use the single scan argument
    promptlens = parse_int_list(args.promptlen)
    genlens = parse_int_list(args.genlen)
    batches = parse_int_list(args.batch)

    # Create cartesian product of all parameters
    parameter_combinations = itertools.product(scans, promptlens, genlens, batches)

    # Run benchmarks for all parameter combinations
    for scan, promptlen, genlen, batch in parameter_combinations:
        print(f"\nRunning benchmark with: scan={scan}, promptlen={promptlen}, genlen={genlen}, batch={batch}")
        run_benchmark(scan, promptlen, genlen, batch, with_nsys=args.with_nsys)

if __name__ == "__main__":
    main()
