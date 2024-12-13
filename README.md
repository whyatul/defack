# Deepfake Detection System

A comprehensive system for detecting deepfake images using multiple deep learning models. The system includes various state-of-the-art architectures and provides a user-friendly web interface for analysis.

## Features

- Multiple model architectures:
  - Xception
  - EfficientNet
  - Swin Transformer
  - Cross Attention
  - Two Stream
  - CNN Transformer

- Face detection using MediaPipe
- Real-time processing and analysis
- Confidence scoring with visual indicators
- User-friendly Streamlit interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. First, convert the trained models (required only once):
```bash
python convert_models.py
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

4. Upload an image through the web interface to analyze it

## Model Details

- **Input Size Requirements:**
  - Xception: 299x299
  - EfficientNet: 300x300
  - Other models: 224x224

- **Face Detection:**
  - Uses MediaPipe face detection
  - Includes 5% padding around detected faces
  - Automatic face cropping and preprocessing

- **Confidence Scoring:**
  - Red: ≥95% confidence
  - Orange: 90-95% confidence
  - Green: 70-90% confidence
  - Blue: <70% confidence

## File Structure

- `app.py`: Main Streamlit application
- `convert_models.py`: Script to convert MLflow models to PyTorch format
- `data_handler.py`: Data processing utilities
- `train_*.py`: Model architecture and training files
- `requirements.txt`: Python dependencies
- `mlruns/`: Directory containing trained models
- `converted_models/`: Directory containing converted PyTorch models

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, CPU-only mode supported)
- See requirements.txt for Python package dependencies

## Model Conversion

The system uses a two-step process for model deployment:
1. Models are initially trained and saved using MLflow
2. Models are converted to pure PyTorch format for inference
This approach ensures compatibility across different environments and hardware configurations.

## Troubleshooting

Common issues and solutions:

1. **No face detected:**
   - Ensure the image contains a clear, front-facing face
   - Try adjusting the image lighting or quality

2. **Model loading errors:**
   - Ensure you've run convert_models.py first
   - Check that all required model files are present in mlruns/

3. **Memory issues:**
   - Try processing smaller images
   - Close other resource-intensive applications

## Notes

- The system is designed to work with CPU-only systems
- Face detection includes a 5% padding for better results
- Models are loaded lazily to minimize memory usage
- Confidence scores are calculated using sigmoid activation

## License

[Your License Information]

## Acknowledgments

- MediaPipe for face detection
- Streamlit for the web interface
- PyTorch and MLflow for model handling 