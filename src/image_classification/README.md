# Continual Learning and Evaluation with PyTorch

This repository contains a PyTorch implementation for a continual learning and evaluation pipeline. The code includes training a neural network model, evaluating it, and adding new classes to the dataset. Below is an overview of the key components and their functionalities:

## Components

### 1. Data Preparation

The data is loaded and preprocessed using PyTorch's `ImageFolder` and `transforms`. Data augmentation techniques like random horizontal flip, random rotation, and color jitter are applied to improve model performance.

### 2. Model Selection

A deep learning model is selected from a predefined list of models, such as AlexNet, VGG, ResNet, and more. The selected model is loaded and prepared for training.

### 3. Training

The selected model is trained on the dataset. The training process includes defining loss functions, optimizers, and learning rate schedules. The code allows for resuming training from a saved model or starting from scratch.

### 4. Evaluation

Model evaluation is performed to assess its performance. This includes calculating loss, accuracy, confusion matrices, and precision-recall per class. Misclassified images are also visualized for further analysis.

### 5. Single Image Prediction

The trained model can be used to make predictions on single images. Given an image file, the model predicts its class and provides a certainty score.

### 6. Continual Learning

The code supports adding new classes to the dataset. This is useful for scenarios where the model needs to adapt to changing or expanding datasets. The new dataset is loaded, and the model is extended to accommodate the new classes.

## Dependencies

You can find the list of required dependencies in the 'requirements.txt' file. To install these dependencies, run the following command:

```bash
pip install -r requirements.txt
```
## Dataset Directory Structure

The dataset directory should be structured as follows:
```bash
 dataset_dir/
 ├── class_1/
 │   ├── image1.jpg
 │   ├── image2.jpg
 │   └── ...
 ├── class_2/
 │   ├── image1.jpg
 │   ├── image2.jpg
 │   └── ...
 └── ...
 ```
 

## Usage

To utilize this project, follow these steps:

1. Clone this repository to your local machine.

2. Install the necessary dependencies as mentioned in the 'Dependencies' section.

### Model Selection

The code provides a list of model options to choose from. You can add more models to the `model_dict` dictionary as needed.

The script supports the following models:

- AlexNet
- VGG11
- VGG16
- VGG19
- ResNet18
- ResNet50
- Inception V3
- MobileNet V2
- GoogLeNet
- ShuffleNet V2 x1.0



### Training

To train a model, specify the following parameters:
- `retrain`: True to start training from scratch, False to load a pre-trained model.
- `num_epochs`: The number of training epochs.
- `DATA_DIR_PATH`: The path to the dataset.
- `model_name`: The name of the model to use.
- `save_weights`: True to save model weights after training.
- `pred_image_path`: Path to an image for single image prediction.
- `eveluate`: True to perform model evaluation.
- `add_class_dir`: Path to a directory containing new classes to add to the dataset.

```bash
example:
python repair_classifier.py --retrain True --num_epochs 4 --datadir ./dataset --model resnet18 --save_weights True
```

### Single Image Prediction

To make predictions on a single image, you should provide the following parameter:

- `pred_image_path`: The file path to the image you wish to predict.

**Important: Ensure that the model and its associated weights are saved before making predictions.**

```bash
example:
python repair_classifier.py --load_model ./resnet18.pth --predict ./dataset3/Sunflower/Sunflower_Downy_mildew/downymildew\(100\).jpeg
```


### Continual Learning

To add new classes to the dataset and extend the model:
- Set `add_class_dir` to the path of the directory containing the new classes.
- The code will automatically load the new data, extend the dataset, and train a model that includes the new classes.
```bash
example:
python repair_classifier.py --add_classes_dir ./dataset3 --datadir ./dataset --load_model ./resnet18.pth --num_epochs 1 --save_weights True
```

## Contributing

We welcome contributions to this project.