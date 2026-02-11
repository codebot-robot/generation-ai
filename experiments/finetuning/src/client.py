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
import finetuning_pb2
import finetuning_pb2_grpc
import argparse
import time
import sys

def run(host, model_id, dataset_id, dataset_text_field, max_steps):
    print(f"Connecting to {host}...")
    
    # We use a longer timeout for the initial connection
    channel = grpc.insecure_channel(host)
    stub = finetuning_pb2_grpc.FinetuningServiceStub(channel)
    
    request = finetuning_pb2.SFTRequest(
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_text_field=dataset_text_field,
        max_steps=max_steps
    )
    
    max_retries = 10
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            print(f"Requesting SFT for {model_id} (attempt {attempt + 1}/{max_retries})...")
            responses = stub.StartSFT(request)
            for response in responses:
                print(f"SERVER: {response.log_entry}")
            
            # If we reach here, the stream finished successfully
            print("Client finished successfully.")
            return
            
        except grpc.RpcError as e:
            print(f"gRPC error on attempt {attempt + 1}: {e.code()} - {e.details()}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Exiting.")
                sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost:50051')
    parser.add_argument('--model_id', type=str, default='facebook/opt-125m') # Small model for testing
    parser.add_argument('--dataset_id', type=str, default='timdettmers/openassistant-guanaco')
    parser.add_argument('--dataset_text_field', type=str, default='text')
    parser.add_argument('--max_steps', type=int, default=5)
    args = parser.parse_args()
    
    run(args.host, args.model_id, args.dataset_id, args.dataset_text_field, args.max_steps)
