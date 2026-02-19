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
import http.client
import json
import hashlib
from urllib.parse import urlparse

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def push_to_modelstore(model_dir, model_id, log_queue, modelstore_url=os.getenv("MODELSTORE_URL", "http://modelstore")):
    log_queue.put(f"Pushing fine-tuned model to modelstore at {modelstore_url}...")
    parsed_url = urlparse(modelstore_url)
    
    files_to_register = []
    
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, model_dir)
            sha256 = calculate_sha256(file_path)
            
            log_queue.put(f"Pushing blob {rel_path} (sha256: {sha256})...")
            try:
                conn = http.client.HTTPConnection(parsed_url.netloc)
                with open(file_path, 'rb') as f:
                    blob_path = f"/blobs/{sha256}"
                    conn.request("PUT", blob_path, body=f)
                    resp = conn.getresponse()
                    if resp.status not in (200, 201):
                        log_queue.put(f"Failed to push blob {rel_path}: {resp.status} {resp.reason}")
                    resp.read()
                conn.close()
                files_to_register.append({
                    "path": rel_path,
                    "sha256": sha256
                })
            except Exception as e:
                log_queue.put(f"Error pushing blob {rel_path}: {str(e)}")

    # Register the model
    log_queue.put(f"Registering model {model_id}...")
    try:
        model_spec = {
            "apiVersion": "generationai.labs.gke.io/v1alpha1",
            "kind": "Model",
            "metadata": {
                "name": model_id
            },
            "spec": {
                "files": files_to_register
            }
        }
        conn = http.client.HTTPConnection(parsed_url.netloc)
        conn.request("POST", "/models", body=json.dumps(model_spec), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status not in (200, 201):
            log_queue.put(f"Failed to register model {model_id}: {resp.status} {resp.reason}")
        else:
            log_queue.put(f"Model {model_id} registered successfully.")
        resp.read()
        conn.close()
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
                # Load model and tokenizer
                log_queue.put(f"Loading tokenizer for {request.model_id}...")
                tokenizer = AutoTokenizer.from_pretrained(request.model_id)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                log_queue.put(f"Loading model {request.model_id}...")
                model = AutoModelForCausalLM.from_pretrained(
                    request.model_id,
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
