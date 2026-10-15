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

import grpc
from concurrent import futures
import threading
import queue
import torch
import os
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from datasets import load_dataset

import finetuning_pb2
import finetuning_pb2_grpc
import json
import hashlib
import requests
from pathlib import Path

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

EMPTY_CONFIG_JSON = b"{}"  # the OCI empty descriptor: artifacts with no config point at sha256("{}")


def download_from_modelstore(model_id, modelstore_url, local_dir, log_queue):
    """Downloads a model's files from modelstore. Returns True if successful.

    modelstore is an OCI registry with a Hugging Face facade: the tree
    listing names the files and `resolve` serves them, so the same URLs
    transformers would use are used here (and HF_ENDPOINT=modelstore_url
    would do the same job inside from_pretrained)."""
    log_queue.put(f"Checking for model {model_id} in modelstore at {modelstore_url}...")
    try:
        resp = requests.get(f"{modelstore_url}/api/models/{model_id}/tree/main", timeout=10)
        if resp.status_code == 404:
            log_queue.put(f"Model {model_id} not found in modelstore.")
            return False
        resp.raise_for_status()
    except Exception as e:
        log_queue.put(f"Error connecting to modelstore: {e}")
        return False

    entries = [e for e in resp.json() if e.get("type") == "file"]
    log_queue.put(f"Downloading model {model_id} ({len(entries)} files) to {local_dir}...")
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        rel_path = entry["path"]
        file_local_path = local_path / rel_path
        file_local_path.parent.mkdir(parents=True, exist_ok=True)
        log_queue.put(f"  Downloading {rel_path}...")
        with requests.get(f"{modelstore_url}/{model_id}/resolve/main/{rel_path}", stream=True) as r:
            r.raise_for_status()
            with open(file_local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

    log_queue.put("Download complete.")
    return True


def push_to_modelstore(model_dir, model_id, log_queue, modelstore_url=os.getenv("MODELSTORE_URL", "http://modelstore.modelstore")):
    """Pushes a directory to modelstore as an OCI artifact tagged
    <model_id>:latest: every file becomes a blob, the manifest lists them
    by name (the same layout and tag `modelstore upload` writes; the Hub
    facade reads it back as revision main)."""
    log_queue.put(f"Pushing fine-tuned model to modelstore at {modelstore_url}...")
    base = modelstore_url.rstrip("/")

    def put_blob(sha256, body, label):
        # content-addressed: a blob the registry already has is a no-op
        head = requests.head(f"{base}/v2/{model_id}/blobs/sha256:{sha256}", timeout=10)
        if head.status_code == 200:
            return True
        resp = requests.put(f"{base}/blobs/{sha256}", data=body, timeout=600)
        if resp.status_code not in (200, 201):
            log_queue.put(f"Failed to push blob {label}: {resp.status_code} {resp.reason}")
            return False
        return True

    layers = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, model_dir).replace(os.sep, "/")
            sha256 = calculate_sha256(file_path)
            log_queue.put(f"Pushing blob {rel_path} (sha256: {sha256})...")
            try:
                with open(file_path, 'rb') as f:
                    if not put_blob(sha256, f, rel_path):
                        return
            except Exception as e:
                log_queue.put(f"Error pushing blob {rel_path}: {str(e)}")
                return
            layers.append({
                "mediaType": "application/vnd.gke-labs.model.file.v1",
                "digest": f"sha256:{sha256}",
                "size": os.path.getsize(file_path),
                "annotations": {"org.opencontainers.image.title": rel_path},
            })

    log_queue.put(f"Registering model {model_id}...")
    try:
        empty_sha = hashlib.sha256(EMPTY_CONFIG_JSON).hexdigest()
        if not put_blob(empty_sha, EMPTY_CONFIG_JSON, "empty config"):
            return
        manifest = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "artifactType": "application/vnd.gke-labs.model.weights.v1+json",
            "config": {"mediaType": "application/vnd.oci.empty.v1+json",
                       "digest": f"sha256:{empty_sha}", "size": len(EMPTY_CONFIG_JSON)},
            "layers": layers,
        }
        resp = requests.put(f"{base}/v2/{model_id}/manifests/latest", data=json.dumps(manifest),
                            headers={"Content-Type": "application/vnd.oci.image.manifest.v1+json"}, timeout=60)
        if resp.status_code not in (200, 201):
            log_queue.put(f"Failed to register model {model_id}: {resp.status_code} {resp.reason} {resp.text[:200]}")
        else:
            log_queue.put(f"Model {model_id} registered successfully as {model_id}:latest.")
    except Exception as e:
        log_queue.put(f"Error registering model {model_id}: {str(e)}")

def get_diagnostics():
    import transformers
    import trl
    diag = []
    diag.append(f"PyTorch version: {torch.__version__}")
    diag.append(f"Transformers version: {transformers.__version__}")
    diag.append(f"TRL version: {trl.__version__}")
    diag.append(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        diag.append(f"Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            diag.append(f"Device {i}: {torch.cuda.get_device_name(i)}")
    return "\n".join(diag)

class QueueCallback(TrainerCallback):
    def __init__(self, log_queue):
        self.log_queue = log_queue

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.log_queue.put(f"Step {state.global_step}: {logs}")

class FinetuningService(finetuning_pb2_grpc.FinetuningServiceServicer):
    def StartSFT(self, request, context):
        yield finetuning_pb2.SFTResponse(log_entry="--- Diagnostics ---")
        yield finetuning_pb2.SFTResponse(log_entry=get_diagnostics())
        yield finetuning_pb2.SFTResponse(log_entry="-------------------")
        yield finetuning_pb2.SFTResponse(log_entry=f"Starting SFT for model {request.model_id} with dataset {request.dataset_id}")
        
        log_queue = queue.Queue()
        
        def run_training():
            try:
                model_id = request.model_id
                modelstore_url = os.environ.get("MODELSTORE_URL", "http://modelstore.modelstore")
                if modelstore_url:
                    local_model_dir = f"/tmp/models/{model_id}"
                    if download_from_modelstore(model_id, modelstore_url, local_model_dir, log_queue):
                        model_id = local_model_dir
                    else:
                        log_queue.put(f"Failed to download model {model_id} from modelstore at {modelstore_url}")
                        return

                # Load model and tokenizer
                log_queue.put(f"Loading tokenizer for {model_id}...")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                log_queue.put(f"Loading model {model_id}...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float32 if not torch.cuda.is_available() else "auto"
                )
                
                # Load dataset
                log_queue.put(f"Loading dataset {request.dataset_id}...")
                dataset = load_dataset(request.dataset_id, split="train")
                
                # Training arguments
                # SFTConfig was introduced in trl 0.8.0.
                # In 0.8.0+, dataset_text_field and max_seq_length should be in SFTConfig.
                sft_config = SFTConfig(
                    output_dir="/tmp/output",
                    max_steps=request.max_steps if request.max_steps > 0 else 5,
                    per_device_train_batch_size=1,
                    logging_steps=1,
                    save_strategy="no",
                    report_to="none",
                    use_cpu=not torch.cuda.is_available(),
                    dataset_text_field=request.dataset_text_field,
                    max_seq_length=512,
                )
                
                log_queue.put("Initializing SFTTrainer...")
                trainer = SFTTrainer(
                    model=model,
                    train_dataset=dataset,
                    args=sft_config,
                    callbacks=[QueueCallback(log_queue)]
                )
                
                log_queue.put("Starting training...")
                trainer.train()
                
                log_queue.put("Saving model...")
                output_dir = "/tmp/output"
                trainer.save_model(output_dir)
                
                output_model_id = f"{request.model_id}-finetuned"
                push_to_modelstore(output_dir, output_model_id, log_queue)
                
                log_queue.put("Fine-tuning completed successfully")
            except Exception as e:
                log_queue.put(f"Error during fine-tuning: {str(e)}")
            finally:
                log_queue.put(None) # Signal end

        thread = threading.Thread(target=run_training)
        thread.start()
        
        while True:
            try:
                log_entry = log_queue.get(timeout=1.0)
                if log_entry is None:
                    break
                yield finetuning_pb2.SFTResponse(log_entry=log_entry)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    finetuning_pb2_grpc.add_FinetuningServiceServicer_to_server(FinetuningService(), server)
    server.add_insecure_port('[::]:50051')
    print("Server starting on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
