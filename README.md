# Text-to-Image AI Image Generation System

A complete Text-to-Image AI application featuring automated synthetic dataset generation, deep learning training pipelines, inference modules, and an interactive Streamlit web interface.

## Project Structure
- `dataset/`: Contains metadata CSVs and generated synthetic images.
- `models/`: Saved model weights and LoRA adapters.
- `outputs/`: Generated output images.
- `generate_dataset.py`: Automatic synthetic dataset generator.
- `train.py`: Model training and fine-tuning script.
- `generate_image.py`: Standalone CLI image generation module.
- `app.py`: Streamlit web application interface.
- `requirements.txt`: Python package dependencies.

## Installation & Execution Guide

### 1. Setup Virtual Environment
python -m venv venv
Windows: venv\Scripts\activate
macOS/Linux: source venv/bin/activate

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Generate Synthetic Dataset
python generate_dataset.py --count 20   # CPU-friendly test mode
python generate_dataset.py --count 1000 # Full dataset generation

### 4. Train / Fine-Tune Model
python train.py --epochs 1

### 5. Generate Image via CLI
python generate_image.py --prompt "A realistic healthcare robot assisting a doctor in a modern hospital"

### 6. Launch Web Application
streamlit run app.py
