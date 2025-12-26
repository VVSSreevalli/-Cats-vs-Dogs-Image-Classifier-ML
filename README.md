# Cats vs Dogs Image Classifier

## Description
This project classifies images as either a cat or a dog using a Convolutional Neural Network (CNN). The model is trained on a labelled dataset and can predict unseen images accurately, demonstrating practical applications of deep learning in computer vision.

## Dataset
- Source: [Kaggle Cats vs Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)  
- Contains labelled images of cats and dogs.  
- Preprocessing: Images are resized, normalized, and converted to tensors for training with PyTorch.

## Algorithms Used
- Convolutional Neural Network (CNN)  

## Tools & Libraries
- Python
- PyTorch
- Torchvision
- NumPy
- PIL

## How to Run
1. Clone the repository.
2. Ensure Python and required libraries are installed (`pip install torch torchvision pillow numpy`).
3. Run `train.py` to train the model.
4. Run `predict.py` to classify new images by updating the image path.

## Results
- The CNN model successfully learned to distinguish cats from dogs.  
- Training loss decreased over epochs, confirming effective learning.  
- Unseen images were correctly classified, demonstrating good generalization.

## Future Scope
- Implement advanced CNN architectures like VGG, ResNet, or MobileNet for higher accuracy.  
- Apply data augmentation to improve model generalization.  
- Extend the system to classify multiple animal categories.  
- Develop a web or mobile application for real-time image classification.
