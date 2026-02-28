# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
import torch
import torch.distributed as dist
import functools
import types
import requests
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

def download_from_modelstore(model_id, modelstore_url, local_dir):
    """Downloads model files from modelstore. Returns True if successful."""
    print(f"Checking for model {model_id} in modelstore at {modelstore_url}...")
    
    # Get model metadata
    try:
        resp = requests.get(f"{modelstore_url}/models/{model_id}", timeout=10)
        if resp.status_code == 404:
            print(f"Model {model_id} not found in modelstore.")
            return False
        resp.raise_for_status()
    except Exception as e:
        print(f"Error connecting to modelstore: {e}")
        return False

    model_data = resp.json()
    print(f"Downloading model {model_id} to {local_dir}...")
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    for file_info in model_data["spec"]["files"]:
        rel_path = file_info["path"]
        sha256 = file_info["sha256"]
        
        file_local_path = local_path / rel_path
        file_local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download blob
        print(f"  Downloading {rel_path}...")
        blob_url = f"{modelstore_url}/blobs/{sha256}"
        with requests.get(blob_url, stream=True) as r:
            r.raise_for_status()
            with open(file_local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    print("Download complete.")
    return True

def print_diagnostics():
    rank = int(os.environ.get("RANK", 0))
    prefix = f"[Rank {rank}] "
    print(f"{prefix}Diagnostics:")
    print(f"{prefix}PyTorch version: {torch.__version__}")
    print(f"{prefix}CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"{prefix}CUDA version: {torch.version.cuda}")
        print(f"{prefix}cuDNN version: {torch.backends.cudnn.version()}")
        print(f"{prefix}Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"{prefix}Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"{prefix}  Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"{prefix}  Multi-processor count: {props.multi_processor_count}")
            print(f"{prefix}  Compute capability: {props.major}.{props.minor}")
    print(f"{prefix}" + "-" * 20)

def get_transformer_layer_cls(model):
    """Identify the transformer layer class for auto wrapping."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return type(model.model.layers[0])
    elif hasattr(model, "layers"):
        return type(model.layers[0])
    return None

def main():
    parser = argparse.ArgumentParser(description="Run simple inference benchmark")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model ID to use")
    parser.add_argument("--enable-fsdp", action="store_true", help="Enable FSDP sharding")
    args = parser.parse_args()

    # Initialize distributed environment if applicable
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"[Rank {rank}] Initialized process group with {backend} backend. World size: {world_size}")
        
        if args.enable_fsdp and torch.cuda.is_available():
             torch.cuda.set_device(local_rank)
    else:
        rank = 0
        local_rank = 0
        print("Not running in distributed mode.")

    print_diagnostics()

    model_id = args.model
    modelstore_url = os.getenv("MODELSTORE_URL", "http://modelstore.modelstore")
    
    if modelstore_url:
        # Use a local directory to download the model
        local_model_dir = f"/tmp/models/{model_id}"
        success = False
        if rank == 0:
            success = download_from_modelstore(model_id, modelstore_url, local_model_dir)
        
        if dist.is_initialized():
            # Broadcast success to other ranks
            device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
            success_tensor = torch.tensor(1 if success else 0, device=device)
            dist.broadcast(success_tensor, src=0)
            success = success_tensor.item() == 1
            dist.barrier()
        
        if success:
            model_id = local_model_dir
        else:
            print(f"[Rank {rank}] Failed to download model {model_id} from modelstore at {modelstore_url}")
            exit(1)

    print(f"[Rank {rank}] Loading model: {model_id}")

    # Set seed for reproducibility and consistency across ranks
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if args.enable_fsdp and dist.is_initialized():
        if not torch.cuda.is_available():
            if rank == 0:
                print("[Rank 0] Warning: FSDP requested but CUDA not available. Falling back to non-sharded model.")
            args.enable_fsdp = False
            
    if args.enable_fsdp and dist.is_initialized():
        print(f"[Rank {rank}] FSDP Enabled. Loading model on CPU first...")
        # For FSDP, we load on CPU (low_cpu_mem_usage=True is default in recent transformers)
        # We avoid device_map="auto" because we want FSDP to handle placement
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto", # or torch.bfloat16
            low_cpu_mem_usage=True,
            device_map=None, 
        )
        
        layer_cls = get_transformer_layer_cls(model)
        auto_wrap_policy = None
        if layer_cls:
            print(f"[Rank {rank}] Found transformer layer class: {layer_cls.__name__}")
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={layer_cls},
            )
        else:
            print(f"[Rank {rank}] Warning: Could not identify transformer layer class. FSDP efficiency might be reduced.")

        print(f"[Rank {rank}] Wrapping model with FSDP...")
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else None
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=device_id,
            use_orig_params=True, # Required for HF models
            sync_module_states=True if torch.cuda.is_available() else False, # Only sync if on GPU
        )
        
        # Monkey-patch generate method
        # We need to bind the original generate method (from the base class) to the FSDP instance.
        # model.module is the original model instance (but stripped of FSDP wrapper).
        # We use type(model.module) to get the class, which has the generate method mixed in.
        model.generate = types.MethodType(type(model.module).generate, model)
        
        # Add attributes to FSDP wrapper for compatibility with HF generate
        for attr in ["config", "generation_config", "can_generate", "main_input_name", "base_model_prefix"]:
            if hasattr(model.module, attr) and not hasattr(model, attr):
                setattr(model, attr, getattr(model.module, attr))
        
        model.device = torch.device(f"cuda:{device_id}") if device_id is not None else torch.device("cpu")
        
    else:
        # Legacy/Single-node behavior
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prompt = "What is Raleigh Scattering?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    device = model.device
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    if rank == 0:
        print(f"Prompt: {prompt}")
        print("Starting generation...")
    
    start_time = time.time()
    
    # Use synced_gpus=True for distributed generation to avoid hangs
    # and only if we are using distributed mode
    is_distributed = dist.is_initialized() and dist.get_world_size() > 1
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            synced_gpus=is_distributed and torch.cuda.is_available()
        )
    
    end_time = time.time()
    
    if rank == 0:
        generated_ids_trimmed = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        print("\nResponse:")
        print(response)
        
        # Calculate metrics
        num_tokens = sum(len(ids) for ids in generated_ids_trimmed)
        duration = end_time - start_time
        tokens_per_second = num_tokens / duration if duration > 0 else 0
        
        print("\nMetrics:")
        print(f"Tokens generated: {num_tokens}")
        print(f"Duration: {duration:.4f} seconds")
        print(f"Tokens per second: {tokens_per_second:.2f} tokens/s")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
