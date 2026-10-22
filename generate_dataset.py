import os
import random
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import argparse

CATEGORIES = [
    "Nature and landscapes",
    "Animals and wildlife",
    "Healthcare and medical technology",
    "Electronics and IoT",
    "Smart homes",
    "Vehicles and transportation",
    "Architecture and buildings",
    "Fantasy and science fiction",
    "Food and everyday objects",
    "Digital art and illustrations"
]

STYLES = ["Photorealistic", "Digital Art", "Cinematic", "Cyberpunk", "Watercolor", "Minimalist"]

PROMPT_TEMPLATES = {
    "Nature and landscapes": [
        "A beautiful sunset over the mountains with a serene lake",
        "A lush green valley with a flowing river and wildflowers",
        "Dense pine forest covered in morning mist and sunlight rays",
        "Golden sand dunes stretching across a vast desert under a clear sky"
    ],
    "Animals and wildlife": [
        "A realistic image of a golden retriever running in a green garden",
        "A majestic lion resting on a rock in the African savanna",
        "A colorful parrot perched on a tropical branch with vibrant flowers",
        "A playful dolphin leaping out of blue ocean waves at sunset"
    ],
    "Healthcare and medical technology": [
        "Generate a healthcare robot assisting a doctor in a modern hospital",
        "A team of surgeons performing an advanced robotic-assisted operation",
        "A biomedical scientist analyzing molecular structures on a holographic display",
        "A state-of-the-art MRI scanner in a clean, brightly lit diagnostic clinic"
    ],
    "Electronics and IoT": [
        "Generate an ESP32 microcontroller connected to sensors, LEDs, and an OLED display",
        "A custom printed circuit board with surface-mount components and soldering",
        "An IoT smart sensor hub monitoring industrial equipment temperature",
        "A Raspberry Pi setup wired to a camera module and breadboard circuits"
    ],
    "Smart homes": [
        "A futuristic smart home control panel mounted on a living room wall",
        "Automated living room with smart lighting, voice assistant, and motorized blinds",
        "A smart kitchen refrigerator with an integrated touch screen display",
        "A robotic vacuum cleaner navigating a modern hardwood floor living room"
    ],
    "Vehicles and transportation": [
        "Create a futuristic smart city with electric cars and tall buildings",
        "An aerodynamic electric sports car charging at a futuristic station",
        "A bullet train speeding through a picturesque countryside landscape",
        "A sleek autonomous delivery drone flying above a suburban neighborhood"
    ],
    "Architecture and buildings": [
        "A modern minimalist glass skyscraper reflecting a dramatic cloudscape",
        "A sustainable eco-friendly building covered in vertical gardens",
        "A futuristic architectural dome city with sky bridges",
        "A cozy contemporary suburban house with warm interior lighting"
    ],
    "Fantasy and science fiction": [
        "A majestic floating island city with waterfalls cascading into the clouds",
        "An astronaut exploring an alien planet with glowing crystal flora",
        "A mystical dragon guarding an ancient ruined temple at twilight",
        "A cyberpunk street market glowing with neon lights in the rain"
    ],
    "Food and everyday objects": [
        "A gourmet artisan burger with melting cheese and fresh vegetables",
        "A steaming cup of specialty latte art on a wooden coffee table",
        "A sleek mechanical keyboard with colorful RGB backlighting",
        "Fresh vibrant tropical fruits arranged neatly on a marble countertop"
    ],
    "Digital art and illustrations": [
        "A vibrant abstract geometric composition with fluid gradients",
        "A whimsical watercolor painting of a cottage in an enchanted forest",
        "A retro synthwave vector illustration of a palm tree sunset",
        "A detailed digital portrait of a cybernetic warrior with glowing optics"
    ]
}

def generate_synthetic_image(output_path, text, category):
    """Generates a structured visual placeholder image representing the text prompt."""
    img = Image.new("RGB", (512, 512), color=(random.randint(20, 50), random.randint(30, 70), random.randint(80, 150)))
    draw = ImageDraw.Draw(img)
    # Draw abstract geometric representations
    for _ in range(5):
        x1, y1 = random.randint(0, 400), random.randint(0, 400)
        x2, y2 = x1 + random.randint(50, 150), y1 + random.randint(50, 150)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    # Add text overlay
    draw.text((20, 440), f"Category: {category[:30]}", fill=(255, 255, 255))
    draw.text((20, 470), f"Prompt: {text[:55]}...", fill=(200, 200, 200))
    img.save(output_path)

def create_dataset(total_count=1000):
    os.makedirs("dataset/images", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    records = []
    print(f"Generating {total_count} synthetic records across {len(CATEGORIES)} categories...")

    for i in range(total_count):
        category = random.choice(CATEGORIES)
        template_list = PROMPT_TEMPLATES[category]
        base_prompt = random.choice(template_list)
        style = random.choice(STYLES)
        prompt = f"{base_prompt}, {style.lower()} style, highly detailed, 4k resolution"
        
        image_id = f"img_{i:05d}"
        image_filename = f"{image_id}.png"
        image_path = os.path.join("dataset", "images", image_filename)
        
        # Generate image file
        generate_synthetic_image(image_path, prompt, category)

        records.append({
            "image_id": image_id,
            "prompt": prompt,
            "category": category,
            "style": style,
            "image_path": image_path,
            "split": "train" # assigned later
        })

    df = pd.DataFrame(records)
    # Remove duplicates if any
    df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)

    # Split dataset: 80% train, 10% validation, 10% test
    random.seed(42)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    df.loc[:train_end, "split"] = "train"
    df.loc[train_end:val_end, "split"] = "validation"
    df.loc[val_end:, "split"] = "test"

    # Save CSV files
    df.to_csv("dataset/metadata.csv", index=False)
    df[df["split"] == "train"].to_csv("dataset/train.csv", index=False)
    df[df["split"] == "validation"].to_csv("dataset/validation.csv", index=False)
    df[df["split"] == "test"].to_csv("dataset/test.csv", index=False)

    print(f"Dataset generation complete! Total valid records: {len(df)}")
    print(f"Train: {len(df[df['split']=='train'])}, Validation: {len(df[df['split']=='validation'])}, Test: {len(df[df['split']=='test'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000, help="Number of records to generate")
    args = parser.parse_args()
    create_dataset(args.count)
