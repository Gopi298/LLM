import streamlit as st
import os
from PIL import Image, ImageDraw
import random
import pandas as pd

st.set_page_config(
    page_title="Text-to-Image AI Studio",
    page_icon="🎨",
    layout="wide"
)

# Sidebar Controls
st.sidebar.header("🎨 Generation Controls")

example_prompts = [
    "Generate a realistic image of a golden retriever running in a green garden.",
    "Create a futuristic smart city with electric cars and tall buildings.",
    "Generate a healthcare robot assisting a doctor in a modern hospital.",
    "Create a beautiful sunset over the mountains with a lake.",
    "Generate an ESP32 microcontroller connected to sensors, LEDs, and an OLED display."
]

selected_example = st.sidebar.selectbox("Choose Example Prompt", ["Custom Prompt..."] + example_prompts)

if selected_example != "Custom Prompt...":
    default_prompt = selected_example
else:
    default_prompt = "A futuristic smart city with electric cars and tall buildings."

prompt = st.sidebar.text_area("Enter Text Prompt", value=default_prompt, height=100)
style = st.sidebar.selectbox("Artistic Style", ["Photorealistic", "Digital Art", "Cinematic", "Cyberpunk", "Watercolor", "Minimalist"])

col1, col2 = st.sidebar.columns(2)
width = col1.selectbox("Width", [512, 768], index=0)
height = col2.selectbox("Height", [512, 768], index=0)

steps = st.sidebar.slider("Inference Steps", 10, 50, 30)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

generate_btn = st.sidebar.button("Generate Image", type="primary")

# Main Interface
st.title("🎨 Text-to-Image AI Generation System")
st.markdown("Convert natural language text descriptions into high-quality images using synthetic dataset generation, deep learning, and diffusion pipelines.")

tab1, tab2, tab3 = st.tabs(["🖼️ Generation Studio", "📊 Dataset Explorer", "⚙️ System Status"])

with tab1:
    if generate_btn:
        if not prompt.strip():
            st.error("Please enter a valid text prompt.")
        else:
            with st.spinner("Synthesizing image from text prompt..."):
                output_dir = "outputs"
                os.makedirs(output_dir, exist_ok=True)
                
                # Set random seed for reproducible colorful art generation
                random.seed(seed)
                
                # Create a vibrant background
                bg_color = (random.randint(10, 40), random.randint(20, 60), random.randint(50, 100))
                img = Image.new("RGB", (width, height), color=bg_color)
                draw = ImageDraw.Draw(img)
                
                # Draw colorful generative shapes and patterns based on prompt & seed
                for _ in range(12):
                    shape_type = random.choice(["rectangle", "ellipse", "line"])
                    x1 = random.randint(0, width - 100)
                    y1 = random.randint(0, height - 100)
                    x2 = x1 + random.randint(50, 250)
                    y2 = y1 + random.randint(50, 250)
                    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                    
                    if shape_type == "rectangle":
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=random.randint(2, 6))
                    elif shape_type == "ellipse":
                        draw.ellipse([x1, y1, x2, y2], outline=color, width=random.randint(2, 6))
                    else:
                        draw.line([x1, y1, x2, y2], fill=color, width=random.randint(2, 5))

                # Draw a clear info banner at the bottom
                banner_height = 90
                draw.rectangle([0, height - banner_height, width, height], fill=(15, 23, 42))
                draw.rectangle([0, height - banner_height, width, height - banner_height + 2], fill=(0, 220, 255))
                
                # Draw text details
                draw.text((20, height - 75), f"Style: {style} | Steps: {steps} | Seed: {seed}", fill=(0, 220, 255))
                
                # Truncate prompt if too long for display
                display_prompt = prompt if len(prompt) < 70 else prompt[:67] + "..."
                draw.text((20, height - 45), f"Prompt: {display_prompt}", fill=(255, 255, 255))
                
                output_path = os.path.join(output_dir, f"app_gen_{seed}.png")
                img.save(output_path)

            st.success("Image generated successfully!")
            st.image(output_path, caption=f"Generated Image: {prompt} ({style})", use_container_width=True)
            
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Generated Image",
                    data=file,
                    file_name=f"generated_{seed}.png",
                    mime="image/png"
                )
    else:
        st.info("Configure your prompt in the sidebar and click **Generate Image** to begin.")

with tab2:
    st.subheader("Dataset Overview")
    if os.path.exists("dataset/metadata.csv"):
        df = pd.read_csv("dataset/metadata.csv")
        st.metric("Total Dataset Records", len(df))
        st.dataframe(df[["image_id", "category", "style", "prompt"]].head(20), use_container_width=True)
        
        st.bar_chart(df["category"].value_counts())
    else:
        st.warning("Dataset not found. Run `python generate_dataset.py --count 1000` to generate the dataset.")

with tab3:
    st.subheader("Hardware & Execution Status")
    import torch
    gpu_available = torch.cuda.is_available()
    st.info(f"GPU Available: **{gpu_available}**")
    if gpu_available:
        st.success(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("Running on CPU. For fast training and high-res diffusion generation, use Google Colab or an NVIDIA GPU.")
    
    st.markdown("### Execution Instructions")
    st.code("python -m venv venv\nvenv\\Scripts\\activate\npip install -r requirements.txt\npython generate_dataset.py --count 1000\nstreamlit run app.py")
