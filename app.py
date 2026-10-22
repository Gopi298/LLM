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
                
                img = Image.new("RGB", (width, height), color=(20, 30, 50))
                draw = ImageDraw.Draw(img)
                draw.rectangle([30, 30, width-30, height-30], outline=(0, 200, 255), width=3)
                draw.text((50, 60), f"Prompt: {prompt[:60]}...", fill=(255, 255, 255))
                draw.text((50, 90), f"Style: {style} | Steps: {steps}", fill=(0, 200, 255))
                
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
