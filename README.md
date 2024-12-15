# Deeplearning Approaches for Deepfake Detection

A comprehensive deepfake detection system implementing multiple deep learning architectures with a user-friendly web interface.

## Project Overview

This project implements various deep learning architectures for deepfake detection:
- Xception
- EfficientNet
- Swin Transformer
- Cross Attention
- CNN-Transformer

Each architecture is trained independently and can be used together in the web application for ensemble predictions.

## Dataset

The models are trained on a curated dataset of 20,000 face-cropped images:
- 10,000 real images
- 10,000 fake images
- Source datasets: DFDC, FF++, and CelebDF-v2
- Dataset Link: [3body-filtered-v2-10k](https://www.kaggle.com/datasets/ameencaslam/3body-filtered-v2-10k/settings)

## Training the Models

### Prerequisites
- Kaggle account with GPU access
- Dataset added to your Kaggle account

### Training Process
1. Use the provided training files (`train_*.py`) in Kaggle notebooks
2. Each architecture has its own training file with specific configurations
3. Common training parameters:
   ```python
   NUM_EPOCHS = 30  # Default, can be increased
   BATCH_SIZE = 32  # For most models
   IMAGE_SIZE = 224  # Varies by model (299 for Xception, 300 for EfficientNet)
   ```
4. Training configurations:
   - Learning rates vary by model and component
   - AdamW optimizer with weight decay
   - ReduceLROnPlateau scheduler
   - MLflow tracking integration

### Model Conversion
After training, models need to be converted for CPU compatibility:
1. Use the provided converter notebook: [DDP-V4-Converter](https://www.kaggle.com/code/ameencaslam/ddp-v4-converter)
2. Pre-converted models available: [DDP-V4-Models](https://www.kaggle.com/datasets/ameencaslam/ddp-v4-models)

## Web Application Setup

### Environment Setup
1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Application Structure
```
deepfake-detection-project-v4/
├── app.py                    # Main Streamlit application
├── video_processor.py        # Video processing utilities
├── feature_visualization.py  # Feature map visualization
├── data_handler.py          # Data processing utilities
├── train_*.py               # Model training files
├── requirements.txt
└── converted_models/        # Directory for converted model files
```

### Running the Application
1. Create `converted_models` directory and place converted model files
2. Start the application:
   ```bash
   streamlit run app.py
   ```

## Using the Web Application

### Image Analysis
1. Select "Image" mode
2. Upload an image containing a face
3. View:
   - Face detection visualization
   - Predictions from each model
   - Feature visualizations (for CNN models)

### Video Analysis
1. Select "Video" mode
2. Upload a video file
3. Choose number of frames to analyze (10-300)
4. View:
   - Model predictions
   - Detected faces from frames
   - Confidence scores and statistics

### Features
- Multi-model ensemble predictions
- Real-time face detection
- Feature map visualization
- Progress tracking
- Support for both image and video inputs
- Dark theme UI

## Model Architecture Details

Each architecture is designed for optimal deepfake detection:

- **Xception**: 
  - Three-flow architecture
  - Specialized in texture analysis
  - Input size: 299x299

- **EfficientNet**: 
  - B3 variant
  - Balanced efficiency-accuracy trade-off
  - Input size: 300x300

- **Swin Transformer**: 
  - Hierarchical feature learning
  - Shifted window attention
  - Input size: 224x224

- **Cross Attention**: 
  - Enhanced feature interaction
  - Dual attention mechanism
  - Input size: 224x224

- **CNN-Transformer**: 
  - Hybrid architecture
  - Combined local-global feature learning
  - Input size: 224x224

## Requirements

- Python 3.8+
- CUDA compatible GPU (for training)
- CPU with 8GB+ RAM (for inference)
- See requirements.txt for package details

## Notes

- Models should be converted before using in the web application
- Face detection uses MediaPipe with fallback confidence thresholds
- Feature visualization available only for CNN-based models
- Video processing time depends on the number of frames selected 