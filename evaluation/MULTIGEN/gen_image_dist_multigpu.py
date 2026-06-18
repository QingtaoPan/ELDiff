import torch
import json
import argparse
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from accelerate import Accelerator
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="./EviDiff")
    parser.add_argument("--img_per_prompt", type=int, default=10)
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Only create output dir in main process
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()  # Wait for dir creation

    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    if accelerator.is_main_process:
        print(f"Model loaded on {device}")

    # Load and prepare data
    with open(args.text_file_path, "r") as f:
        text_data = json.load(f)

    # Prepare all data points
    all_data = []
    for index in range(len(text_data)):
        texts = [text_data[index]["text"] for _ in range(args.img_per_prompt)]
        for j in range(args.img_per_prompt):
            all_data.append((index, texts[j], j))

    # Split data across processes
    # Create a list version for iteration
    data_per_process = all_data[accelerator.process_index::accelerator.num_processes]

    if accelerator.is_main_process:
        print(f"Total tasks: {len(all_data)}, Tasks per process: ~{len(data_per_process)}")

    # Process data
    progress_bar = tqdm(
        data_per_process,
        desc="Generating images",
        disable=not accelerator.is_main_process
    )

    for index, text, j in progress_bar:
        img_id = f"{index}_{j}"
        save_path = os.path.join(args.output_dir, f"{img_id}.png")

        # Skip if already exists (optional)
        if os.path.exists(save_path):
            continue

        # Generate and save image
        try:
            image = pipe(prompt=text).images[0]
            image.save(save_path)
        except Exception as e:
            print(f"Error generating {img_id}: {str(e)}")
            continue

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("All images generated successfully")

if __name__ == "__main__":
    main()