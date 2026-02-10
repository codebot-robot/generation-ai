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

def run(host, model_id, dataset_id, dataset_text_field, max_steps):
    with grpc.insecure_channel(host) as channel:
        stub = finetuning_pb2_grpc.FinetuningServiceStub(channel)
        request = finetuning_pb2.SFTRequest(
            model_id=model_id,
            dataset_id=dataset_id,
            dataset_text_field=dataset_text_field,
            max_steps=max_steps
        )
        
        print(f"Requesting SFT for {model_id}...")
        for response in stub.StartSFT(request):
            print(f"SERVER: {response.log_entry}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost:50051')
    parser.add_argument('--model_id', type=str, default='facebook/opt-125m') # Small model for testing
    parser.add_argument('--dataset_id', type=str, default='timdettmers/openassistant-guanaco')
    parser.add_argument('--dataset_text_field', type=str, default='text')
    parser.add_argument('--max_steps', type=int, default=5)
    args = parser.parse_args()
    
    run(args.host, args.model_id, args.dataset_id, args.dataset_text_field, args.max_steps)
