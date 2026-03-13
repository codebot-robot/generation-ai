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
import time
import mlx.core as mx
from mlx_lm import load, generate

def main():
    parser = argparse.ArgumentParser(description="Run MLX inference benchmark")
    parser.add_argument("--model", type=str, default="mlx-community/functiongemma-270m-it-6bit", help="Model ID to use")
    parser.add_argument("--prompt", type=str, default="What is Rayleigh Scattering?", help="Prompt for inference")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device to use (cpu or gpu)")
    args = parser.parse_args()

    if args.device == "cpu":
        mx.set_default_device(mx.cpu)
        print("Using CPU device")
    else:
        mx.set_default_device(mx.gpu)
        print("Using GPU device")

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    prompt = args.prompt
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

    print(f"Prompt: {args.prompt}")
    print("Starting generation...")
    start_time = time.time()

    # mlx_lm.generate returns the response string
    response = generate(model, tokenizer, prompt=prompt, verbose=True)

    end_time = time.time()

    print("\nResponse:")
    print(response)

    duration = end_time - start_time
    print(f"\nDuration: {duration:.4f} seconds")
    
    # Estimate tokens
    tokens = tokenizer.encode(response)
    num_tokens = len(tokens)
    tokens_per_second = num_tokens / duration if duration > 0 else 0
    print(f"Estimated tokens: {num_tokens}")
    print(f"Tokens per second: {tokens_per_second:.2f} tokens/s")

if __name__ == "__main__":
    main()
