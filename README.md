# Cat Breed Classification

This project is a deep learning-based image classification system that identifies different cat breeds using Convolutional Neural Networks (CNN) and TensorFlow. The model is trained on a dataset containing 888 images from five popular cat breeds: Bengal, Domestic Shorthair, Maine Coon, Ragdoll, and Siamese.

## 🔍 Project Overview

- **Framework**: TensorFlow & Keras
- **Model**: MobileNetV2 (transfer learning)
- **Dataset**: [Cat Breed Dataset (Kaggle)](https://www.kaggle.com/datasets/yapwh1208/cats-breed-dataset)
- **Classes**: Bengal, Domestic Shorthair, Maine Coon, Ragdoll, Siamese
- **Augmentation**: Random flip, rotation, and zoom
- **Evaluation**: Accuracy and loss on validation set
- **Deployment**: Flask web app for breed prediction

## Folder Structure

```
Cat-Breed-Classification/
├── static/uploads/         # Uploaded images
├── templates/              # HTML templates (Flask)
├── model_mobilenetv2_ras_kucing.keras  # Trained model
├── app.py                  # Flask web application
├── train_model.ipynb       # Model training notebook
└── README.md
```

## Getting Started

1. Clone this repo:

```bash
git clone https://github.com/yourusername/Cat-Breed-Classification.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the web app:

```bash
python app.py
```

4. Upload a cat image to get predictions.

## Model Training Highlights

- **Base Model**: MobileNetV2 (frozen)
- **Custom Layers**: GlobalAveragePooling → Dense → Dropout → Output
- **Optimizations**: Class weights, early stopping, learning rate reduction

## Results

The model achieves good accuracy on the validation dataset with low overfitting, thanks to augmentation and transfer learning.

## Author

- GitHub: [9Fik](https://github.com/9Fik)
- Email: taufik21ti@mahasiswa.pcr.ac.id
