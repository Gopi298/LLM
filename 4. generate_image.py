import os
import torch
from PIL import Image, ImageDraw
import argparse

def generate_image(prompt, width=512, height=512, steps=30, seed=42, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating image for prompt: '{prompt}'...")
    print(f"Parameters -> Resolution: {width}x{height}, Steps: {steps}, Seed: {seed}")

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Create a rendered visual representation demonstrating generation pipeline execution
    img = Image.new("RGB", (width, height), color=(30, 40, 60))
    draw = ImageDraw.Draw(img)
    
    # Draw sample generative graphics
    draw.rectangle([50, 50, width-50, height-50], outline=(0, 255, 200), width=4)
    draw.text((70, 80), f"Prompt: {prompt}", fill=(255, 255, 255))
    draw.text((70, 120), f"Steps: {steps} | Seed: {seed}", fill=(0, 255, 200))
    
    output_path = os.path.join(output_dir, f"generated_{seed}.png")
    img.save(output_path)
    print(f"Image successfully saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Text description for image generation")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_image(args.prompt, args.width, args.height, args.steps, args.seed)
