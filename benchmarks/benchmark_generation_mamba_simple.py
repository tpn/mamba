# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time
import json
import os

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

# Define a global variable for SCAN_OPTION to be accessed by imports
os.environ["MAMBA_SCAN_OPTION"] = "cuda2"  # Default to cuda2

parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--scan", type=str, choices=["cuda", "cuda2", "ref"], default="cuda2",
                    help="Selective scan implementation to use: cuda (selective_scan_cuda), cuda2 (selective_scan2_cuda), or ref (reference implementation)")
args = parser.parse_args()

# Set the scan option to be used by the imports
os.environ["MAMBA_SCAN_OPTION"] = args.scan

# Import mamba models after setting the environment variable to ensure it uses the correct implementation
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams, update_graph_cache
from mamba_ssm.utils.generation import sample, modify_logit_for_repetition_penalty

repeats = 3
device = "cuda"
dtype = torch.float16

print(f"Loading model {args.model_name}")
print(f"Using scan implementation: {args.scan}")
model_name = args.model_name
is_mamba = (
    model_name.startswith("state-spaces/mamba")
    or model_name.startswith("state-spaces/transformerpp")
    #or model_name.startswith("tiiuae/falcon-mamba-7b")
)
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

torch.random.manual_seed(0)
if args.prompt is None:
    input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
else:
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)
max_length = input_ids.shape[1] + args.genlen

if is_mamba:
    # Create a function that performs the standard generate operation to verify our separate timing approach
    def standard_generate():
        return model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            min_p=args.minp,
            repetition_penalty=args.repetition_penalty,
        )

    # Run once to initialize CUDA graphs
    std_out = standard_generate()

    if args.prompt is not None:
        print(tokenizer.batch_decode(std_out.sequences.tolist()))

    @torch.inference_mode()
    def benchmark_separate_timing():
        batch_size, seqlen_og = input_ids.shape

        # 1. Setup and process prompt
        torch.cuda.synchronize()
        prompt_start = time.time()

        # Initialize the inference parameters
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None

        # Setup CUDA graph cache similar to the original decode function
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)

        # Process the prompt
        position_ids = None
        model_output = model(
            input_ids,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=1,
        )
        logits = model_output.logits.squeeze(dim=1)

        # Apply repetition penalty if needed
        if args.repetition_penalty != 1.0:
            logits = modify_logit_for_repetition_penalty(
                logits, input_ids, args.repetition_penalty
            )

        inference_params.seqlen_offset += input_ids.shape[1]

        torch.cuda.synchronize()
        prompt_end = time.time()
        prompt_time = prompt_end - prompt_start

        # 2. Generate tokens
        torch.cuda.synchronize()
        decode_start = time.time()

        # Generate the remaining tokens one by one
        curr_ids = sample(
            logits,
            top_k=args.topk,
            top_p=args.topp,
            min_p=args.minp,
            temperature=args.temperature
        ).unsqueeze(-1)

        generated_ids = [curr_ids]
        sequences_cat = torch.cat([input_ids, curr_ids], dim=1)

        for _ in range(args.genlen - 1):
            position_ids = torch.full(
                (batch_size, 1),
                inference_params.seqlen_offset,
                dtype=torch.long,
                device=input_ids.device,
            )

            # Use CUDA graph when available for decoding
            if inference_params.seqlen_offset > 0 and model._decoding_cache is not None:
                logits = model._decoding_cache.run(
                    curr_ids, position_ids, inference_params.seqlen_offset
                ).squeeze(dim=1)
            else:
                model_output = model(
                    curr_ids,
                    position_ids=position_ids,
                    inference_params=inference_params,
                    num_last_tokens=1,
                )
                logits = model_output.logits.squeeze(dim=1)

            # Apply repetition penalty if needed
            if args.repetition_penalty != 1.0:
                logits = modify_logit_for_repetition_penalty(
                    logits, sequences_cat, args.repetition_penalty
                )

            # Sample next token
            curr_ids = sample(
                logits,
                top_k=args.topk,
                top_p=args.topp,
                min_p=args.minp,
                temperature=args.temperature
            ).unsqueeze(-1)

            generated_ids.append(curr_ids)
            sequences_cat = torch.cat([sequences_cat, curr_ids], dim=1)

            inference_params.seqlen_offset += 1

        torch.cuda.synchronize()
        decode_end = time.time()
        decode_time = decode_end - decode_start

        # Return complete sequence for verification
        sequences = torch.cat([input_ids] + generated_ids, dim=1)
        return sequences, prompt_time, decode_time

    # Run the benchmark
    test_out, _, _ = benchmark_separate_timing()

    # Run multiple times and average
    prompt_times = []
    decode_times = []
    for _ in range(repeats):
        _, prompt_time, decode_time = benchmark_separate_timing()
        prompt_times.append(prompt_time)
        decode_times.append(decode_time)

    avg_prompt_time = sum(prompt_times) / repeats * 1000  # Convert to ms
    avg_decode_time = sum(decode_times) / repeats * 1000  # Convert to ms
    total_time = avg_prompt_time + avg_decode_time
    tokens_per_sec = args.genlen / (sum(decode_times) / repeats)

    print(f"Prompt length: {len(input_ids[0])}, generation length: {args.genlen}")
    print(f"{args.model_name} prompt processing time: {avg_prompt_time:.0f}ms")
    print(f"{args.model_name} decoding time: {avg_decode_time:.0f}ms")
    print(f"{args.model_name} tokens per second during decoding: {tokens_per_sec:.1f}")
    print(f"{args.model_name} total time (prompt + decoding): {total_time:.0f}ms")
else:
    # Original benchmark code for non-Mamba models
    fn = lambda: model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_length=max_length,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty,
    )
    out = fn()
    if args.prompt is not None:
        print(tokenizer.batch_decode(out.sequences.tolist()))

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
    print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")
