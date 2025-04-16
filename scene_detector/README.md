# CNN Image Classification with PyTorch

A deep learning project implementing a Convolutional Neural Network (CNN) using PyTorch for image classification.

## Technologies Used
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- NumPy

## Project Structure
The project is implemented as a Jupyter notebook (`cnn_train_and_inference.ipynb`) containing:
- Data preprocessing and transformations
- CNN model architecture
- Training pipeline
- Inference functionality

## Setup and Dependencies
```python
pip install torch torchvision numpy Pillow
```

## Features
- Custom dataset handling using PyTorch's Dataset class
- Image transformations including resizing to 150x150
- CNN architecture for image classification
- Training loop with Adam optimizer
- Model inference capabilities

## Usage
1. Open the Jupyter notebook `cnn_train_and_inference.ipynb`
2. Follow the cells in sequence for:
   - Data preparation
   - Model training
   - Making predictions

## Model Architecture
The CNN model includes:
- Convolutional layers
- Pooling layers
- Fully connected layers
- Adam optimizer for training

## Data Preprocessing
Images are preprocessed using the following transformations:
```python
transforms.Compose([
    transforms.Resize((150,150)),
    # Additional transforms as defined in the notebook
])
```

## Contributing
Feel free to fork this repository and submit pull requests for any improvements.

## License
[Add your chosen license here]
