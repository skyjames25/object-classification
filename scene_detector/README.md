# Scene Classification Model

This project implements a Convolutional Neural Network (CNN) for scene classification using PyTorch. The model can classify images into different scene categories.

## Features

- CNN-based scene classification
- Web interface using Streamlit
- Support for both training and inference
- GPU acceleration support
- Pre-trained model included

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Pillow
- Streamlit
- numpy

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install torch torchvision pillow streamlit numpy
```

## Project Structure

- `scene_train.py`: Contains the model architecture and training code
- `web_deployment.py`: Streamlit web interface for easy inference
- `best_checkpoint.model`: Pre-trained model weights
- `cnn_train_and_inference.ipynb`: Jupyter notebook with detailed training and inference examples

## Usage

### Training the Model

To train the model on your own dataset:

1. Organize your training data in the following structure:
```
seg_train/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
```

2. Run the training script:
```bash
python scene_train.py
```

### Using the Web Interface

1. Start the Streamlit web interface:
```bash
streamlit run web_deployment.py
```

2. Open your web browser and navigate to the provided local URL
3. Upload an image to get the scene classification prediction

### Using the Model Programmatically

```python
from scene_train import ConvNet, transformer, classes
import torch
from PIL import Image

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet(num_classes=6).to(device)
model.load_state_dict(torch.load('best_checkpoint.model', map_location=device))
model.eval()

# Make predictions
def predict_image(image_path):
    image = Image.open(image_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0).to(device)
    output = model(image_tensor)
    index = output.data.cpu().numpy().argmax()
    return classes[index]

# Example usage
prediction = predict_image('path_to_your_image.jpg')
print(f"Predicted scene: {prediction}")
```

## Model Architecture

The model uses a CNN architecture with:
- 3 convolutional layers
- Batch normalization
- ReLU activation
- Max pooling
- Fully connected layer for classification

## Performance

The model has been trained on a dataset of scene images and can classify images into 6 different scene categories. The pre-trained model achieves good accuracy on the test set.

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues and enhancement requests! 