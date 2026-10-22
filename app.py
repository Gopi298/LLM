import streamlit as st
import os
import torch
import pandas as pd
from PIL import Image

# Hugging Face Diffusers Library
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

st.set_page_config(
    page_title="Text-to-Image AI Studio",
    page_icon="🎨",
    layout="wide"
)

# Cache model loading so it only loads once into memory
@st.cache_resource
def load_diffusion_pipeline():
    if not DIFFUSERS_AVAILABLE:
        return None, "diffusers library is not installed."
    
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        if device == "cuda":
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            )
            pipe = pipe.to("cuda")
            # Enable memory optimization if available
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
        else:
            # CPU fallback (uses float32)
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32
            )
            pipe = pipe.to("cpu")
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
                
        return pipe, f"Successfully loaded {model_id} on {device.upper()}"
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"

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

# Set lower default steps on CPU to save generation time
default_steps = 25 if torch.cuda.is_available() else 15
steps = st.sidebar.slider("Inference Steps", 10, 50, default_steps)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

generate_btn = st.sidebar.button("Generate Image", type="primary")

# Main Interface
st.title("🎨 Text-to-Image AI Generation System")
st.markdown("Convert natural language text descriptions into real AI-generated images using Stable Diffusion.")

tab1, tab2, tab3 = st.tabs(["🖼️ Generation Studio", "📊 Dataset Explorer", "⚙️ System Status"])

with tab1:
    if generate_btn:
        if not prompt.strip():
            st.error("Please enter a valid text prompt.")
        else:
            with st.spinner("Loading Stable Diffusion Pipeline & Generating Image (this may take a few moments)..."):
                pipe, status_msg = load_diffusion_pipeline()
                
                if pipe is None:
                    st.error(f"Error loading AI model: {status_msg}")
                else:
                    output_dir = "outputs"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Prepare styled prompt
                    full_prompt = f"{prompt}, {style.lower()} style, highly detailed, 4k resolution"
                    
                    # Set seed for reproducible generation
                    generator = torch.Generator(device=pipe.device).manual_seed(seed)
                    
                    # Run inference
                    result = pipe(
                        prompt=full_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        generator=generator
                    )
                    
                    generated_image = result.images[0]
                    
                    output_path = os.path.join(output_dir, f"app_gen_{seed}.png")
                    generated_image.save(output_path)

                    st.success("Real AI Image generated successfully!")
                    st.image(output_path, caption=f"Generated Image: '{prompt}' ({style})", use_container_width=True)
                    
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="Download Generated Image",
                            data=file,
                            file_name=f"generated_{seed}.png",
                            mime="image/png"
                        )
    else:
        st.info("Configure your prompt in the sidebar and click **Generate Image** to run Stable Diffusion.")

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
    gpu_available = torch.cuda.is_available()
    st.info(f"GPU Available: **{gpu_available}**")
    if gpu_available:
        st.success(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("Running on CPU. Real AI image generation on CPU will take 1-3 minutes per image. For 5-second generations, run this code on Google Colab with a free T4 GPU!")
    
    st.markdown("### Execution Instructions")
    st.code("python -m venv venv\nvenv\\Scripts\\activate\npip install -r requirements.txt\npython generate_dataset.py --count 1000\nstreamlit run app.py")
