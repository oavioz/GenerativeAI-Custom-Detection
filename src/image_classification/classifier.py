# In[1]
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


# # imports

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import os
import time
import copy
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import pandas as pd
from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve, precision_score, recall_score, accuracy_score
import seaborn as sns
import argparse


# # specifics

torch.manual_seed(42)
batch_size = 8
h_dim = 800
z_dim = 35
x_sigma2 = 0.0015
learn_rate = 0.0002
betas = (0.99, 0.999)
num_epochs = 2
class_certainty_threshold = 0.95  
sub_class_certainty_threshold = 0.75
momentum = 0.9
step_size =  7
gamma = 0.1
eveluate = True

#after adding classes
learning_rate_first_layers = 0.001
learning_rate_final_layer = 0.01


# # General

PROJECT_PATH = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model_dict = {
    'alexnet': models.alexnet,
    'vgg11': models.vgg11,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'resnet18': models.resnet18,
    'resnet50': models.resnet50,
    'inception_v3': models.inception_v3,
    'mobilenet_v2': models.mobilenet_v2,
    "googlenet": models.googlenet,
    "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,

}

def create_model_instance(model_name):
    if model_name in list(model_dict.keys()):
        print(f"Creating {model_name}")
        return model_dict.get(model_name.lower())(pretrained=True)
    
def all_models():
    my_models = {key: create_model_instance(key) for key in model_dict.keys()}
    for model_name, model in my_models.items():
        print(f"###### Model: {model_name}")
        if 'classifier' in model._modules:
            # Modify the classifier layers for models like VGG
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, len(dataset.main_classes_set.keys()))
        elif 'fc' in model._modules:
            # Modify the fully connected layer for models like AlexNet
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(dataset.main_classes_set.keys()))

        model = model.to(device)

def rename_file(root, filename):
    base, ext = os.path.splitext(filename)
    if ext == '.ini':
        print(f"Skipping {os.path.join(root, filename)}")
        return
    if " " in base or base.count(".") >= 1:
        # Remove spaces and dots
        new_base = base.replace(" ", "").replace(".", "")
        new_filename = f"{new_base}{ext}"
        old_path = os.path.join(root, filename)
        new_path = os.path.join(root, new_filename)

        # Rename the file
        os.rename(old_path, new_path)
        filename = new_filename
    return filename

def create_dataset_from_directory(root_directory):
    main_classes = []
    samples = []
    main_classes_dict = {}

    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            filename = rename_file(root, filename)
            
            relative_path = root.split(os.path.sep)
            main = str(relative_path[-2]) + "_" + str(relative_path[-1])
            if main not in main_classes_dict.keys():
                main_classes_dict[main] = len(main_classes_dict.keys())
            
            main_classes.append(main_classes_dict[main])
            
            samples.append(os.path.join(root, filename))
                        
            
    return torch.tensor(main_classes), samples, main_classes_dict


# Define custom PyTorch dataset
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, start_inx=0):
            self.main_class_labels, self.image_paths, self.main_classes_set = create_dataset_from_directory(data_dir)
            self.transform = transform
            if start_inx > 0:
                self.main_class_labels += start_inx
                self.main_classes_set = {key: value+start_inx for key, value in self.main_classes_set.items()}



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        main_class = self.main_class_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, main_class #, sub_class
    
    def get_classes_names(self):
        classes = {v: k for k, v in self.main_classes_set.items()}
        return classes
    
    def concat_datasets(self, new_dataset):
        self.main_class_labels = torch.cat((self.main_class_labels, new_dataset.main_class_labels))
        self.image_paths = self.image_paths + new_dataset.image_paths
        old_num_classes = len(self.main_classes_set.keys())
        num_new_classes = 0
        for key, val in new_dataset.main_classes_set.items():
            if key not in self.main_classes_set.keys():
                print(f"Adding {key} with {val} to the dataset")
                self.main_classes_set[key] = val
                num_new_classes += 1
            else:
                print(f"Skipping {key} with {val} because it already exists")
                
        return num_new_classes


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Lists to store training and validation statistics for plotting
    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        i=0
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            with tqdm(dataloaders[phase], total=len(dataloaders[phase])) as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            scheduler.step()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    pbar.update(1)
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.cpu().numpy())
            else:
                valid_loss_history.append(epoch_loss)
                valid_acc_history.append(epoch_acc.cpu().numpy())

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Plot the training and validation statistics
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(valid_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.plot(valid_acc_history, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
                

def save_model(filename, model_name, state_dict, class_labels):
    # Save the best model weights to a file
    filename = f'{filename}.pth'
    torch.save({'model_name': model_name,
                'num_classes': len(class_labels.keys()),
                'state_dict': state_dict,
                'class_labels': class_labels},
                 filename)

    print('best_weights saved')

    return model_ft.state_dict()


def save_layers_except_last(model_ft, model_name):
    # Create a new dictionary to hold the state_dict of the layers you want to save
    filtered_state_dict = {}

    # Copy the state_dict of the layers you want to save into the new dictionary
    for layer_name, weights in model_ft.state_dict().items():
        # print(layer_name)
        if 'fc' not in layer_name:
            filtered_state_dict[layer_name] = weights
        else:
            print(f"Excluding {layer_name}")

    # Save the filtered state_dict to a file
    torch.save(filtered_state_dict, model_name + 'best_weights_except_last.pth')
    print('best_weights_except_last saved')

    return filtered_state_dict

def evaluate_model_with_limited_misclassified(model, dataloader, criterion, device, class_labels):
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    predictions = []
    actuals = []
    all_images = []  # To store the images
    misclassified_images = []  # To store misclassified images
    max_misclassified=9

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc='Evaluation Progress') as pbar:
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                probabilities, _ = F.softmax(outputs, dim=1).max(dim=1)

                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                predictions.extend(predicted.cpu().numpy())
                actuals.extend(labels.cpu().numpy())
                all_images.extend(inputs.cpu())  # Store all images

                for pred, label, image, prob in zip(predicted, labels, inputs, probabilities):
                    if len(misclassified_images) < max_misclassified and pred != label:
                        misclassified_images.append((image.cpu(), pred, label, prob))

                pbar.update(1)

    # Calculate loss and accuracy
    evaluation_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    # Calculate confusion matrix
    cm = confusion_matrix(actuals, predictions)

    # Calculate precision and recall per class
    precision_per_class = precision_score(actuals, predictions, average=None)
    recall_per_class = recall_score(actuals, predictions, average=None)

    print(f'Evaluation Loss: {evaluation_loss:.4f}')
    print(f'Evaluation Accuracy: {accuracy * 100:.2f}')

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Precision and Recall Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(class_labels))
    bar_width = 0.35

    ax.bar(x, precision_per_class, width=bar_width, label='Precision', color='b', align='center')
    ax.bar(x, recall_per_class, width=bar_width, label='Recall', color='g', align='edge')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45)
    ax.set_xlabel('Class')
    ax.set_ylabel('Value')
    ax.set_title('Precision and Recall per Class')
    ax.legend()

    plt.tight_layout()
    plt.show()

    data = [{'Class': label, 'Precision': precision, 'Recall': recall} for label, precision, recall in zip(class_labels, precision_per_class, recall_per_class)]

    df = pd.DataFrame(data)

    print(df)
    
    # Plot samples from misclassified images
    rows,cols = 3,3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle('Model Test Misclassified Results', fontsize=16)

    for i, (image, pred, actual, certainty) in enumerate(misclassified_images):
        if i >= max_misclassified:
            break

        ax = axes[i // 3, i % 3]
        ax.imshow(image.permute(1, 2, 0))  # Assuming images are in the format (C, H, W)
        ax.set_title(f"Pred: {class_labels[pred]}\nActual: {class_labels[actual]}\n"
                    f"Certainty: {certainty:.2f}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def predict_single_image(model, image_path, device, classes_names):
    model.eval()
    
    # Load the image and apply the same transformations as in your dataloader
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, prediction = torch.max(output, 1)
        certainty = probabilities.max(dim=1)[0].item()
        label = classes_names[prediction.item()]

    print(f"Predicted Label: {label}, Certainty: {certainty}")
    
    return label, certainty



if __name__ == "__main__":
    # # Parse command line arguments
    
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--train", type=bool, default=False, help="Train")
    parser.add_argument("--add_classes", type=bool, default=False, help="Add new classes")
    parser.add_argument("--predict", type=str, help="Predict")

    parser.add_argument("--retrain", type=bool, default=False, help="Retrain the model")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--datadir", type=str, help="Data directory")
    parser.add_argument("--model", type=str, default="resnet18", help="Model type")
    parser.add_argument("--load_model", type=str, help="Load weights")
    parser.add_argument("--save_weights", type=bool, default=False, help="Save weights")
    
    parser.add_argument("--evaluate", type=bool, default=False, help="Evaluate")
    parser.add_argument("--add_classes_dir", type=str, help="Add new classes")
    


    args = parser.parse_args()

    # Update your variables with the parsed arguments
    retrain = args.retrain
    num_epochs = args.num_epochs
    DATA_DIR_PATH = args.datadir
    model_name = args.model
    save_weights = args.save_weights
    pred_image_path = args.predict
    eveluate = args.evaluate
    add_classes_dir = args.add_classes_dir
    load_model = args.load_model
    
    # datasets and loaders

    # ## prepare labels

    if (retrain or add_classes_dir) and not DATA_DIR_PATH:
        print("Please provide a data directory")
        exit()


    # ## datasets

    if DATA_DIR_PATH:

        # Define data transformations including data augmentation
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        # Create the custom dataset
        dataset = CustomDataset(DATA_DIR_PATH, transform=transform)

        # Split the dataset into train, validation, and test subsets
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
        ds = {'train': train_data, 'valid': val_data, 'test': test_data}


        # ## data loaders

        dataloaders = {key: DataLoader(ds[key], batch_size=batch_size, shuffle=True) for key in ds.keys()} 
        dataset_sizes = {
            'train': len(train_data),
            'valid': len(val_data)
        }

    # # Model

    # ## chose model

    if load_model:
        checkpoint = torch.load(load_model)
        model_name = checkpoint['model_name']
        num_classes = checkpoint['num_classes']
        model_state_dict = checkpoint['state_dict']
        class_labels = checkpoint['class_labels']
    elif DATA_DIR_PATH:
        num_classes = len(dataset.main_classes_set.keys())
    else:
        print("Please provide a data directory or a model to load")

    model_ft = create_model_instance(model_name) 
    if 'classifier' in model_ft._modules:
                    # Modify the classifier layers for models like VGG
                    num_ftrs = model_ft.classifier[-1].in_features
                    model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    elif 'fc' in model_ft._modules:
                    # Modify the fully connected layer for models like AlexNet
                    num_ftrs = model_ft.fc.in_features
                    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    if load_model:
        model_ft.load_state_dict(model_state_dict)

    # Transfer the model to GPU
    model_ft = model_ft.to(device)

    if pred_image_path:
        predict_single_image(model_ft, pred_image_path, device, class_labels)
        exit()


    # ## params

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer 
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.AdamW(model_ft.parameters(), betas=betas)
    # Learning rate decay
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)


    # ## find best params for model


    # Optimizer 
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=learn_rate, betas=betas)
    # Learning rate decay
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)


    # # Train


    # ## fit/train
          

    if retrain:
        # Train the model
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        dataloaders, dataset_sizes, device, num_epochs=num_epochs)
    else:
        # Load weights
        model_ft.load_state_dict(torch.load('best_weights.pth'))


    # ## save weights

    if save_weights:
        classes_dict = dataset.get_classes_names()
        save_model(model_name, model_name, model_ft.state_dict(), classes_dict)
        save_layers_except_last(model_ft, model_name)

    # # Evaluation


    if eveluate:
        evaluate_model_with_limited_misclassified(model_ft, dataloaders['test'], criterion, device, list(dataset.get_classes_names().values()))



    # # Continual Learning - Add new classes

    # ## prepare new datasets and dataloaders

    if add_classes_dir:

        new_data_dir = add_classes_dir
        # Create a new dataset with data for the new class
        new_class_dataset = CustomDataset(new_data_dir, transform=transform, start_inx=len(dataset.main_classes_set.keys()))
        print(f'{len(new_class_dataset)} samples in the new dataset')

        own_concat_dataset = copy.copy(dataset)
        num_new_classes = own_concat_dataset.concat_datasets(new_class_dataset)
        print(f'{num_new_classes} new classes added to the dataset')

        total_size = len(own_concat_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        own_concat_train_data, own_concat_val_data, own_concat_test_data = random_split(own_concat_dataset, [train_size, val_size, test_size])
        own_concat_dataset_ds = {'train': own_concat_train_data, 'valid': own_concat_val_data, 'test': own_concat_test_data}
        own_concat_class_dls = {key: DataLoader(own_concat_dataset_ds[key], batch_size=batch_size, shuffle=True) for key in own_concat_dataset_ds.keys()}
        own_concat_class_dataset_sizes = {
            'train': len(own_concat_dataset_ds['train']),
            'valid': len(own_concat_dataset_ds['valid'])
        }


        # ## new model

        new_model = copy.deepcopy(model_ft)
        if 'classifier' in new_model._modules:
            # Modify the classifier layers for models like VGG
            num_ftrs = new_model.classifier[-1].in_features
            new_model.classifier[-1] = nn.Linear(num_ftrs, new_model.classifier[-1].out_features+num_new_classes)
        elif 'fc' in new_model._modules:
            # Modify the fully connected layer for models like AlexNet
            num_ftrs = new_model.fc.in_features
            new_model.fc = nn.Linear(num_ftrs, model_ft.fc.out_features+num_new_classes)

        new_model.to(device)


        first_layers_params = []
        final_layer_params = []

        for name, param in new_model.named_parameters():
            if 'fc' not in name:
                first_layers_params.append(param)
            else:
                final_layer_params.append(param)
            
        # Create separate parameter groups with different learning rates
        param_groups = [
            {'params': first_layers_params, 'lr': 0},
            {'params': final_layer_params, 'lr': learning_rate_final_layer}
        ]

        add_class_optimizer = optim.AdamW(param_groups, lr=0.001, weight_decay=0.01)



        own_concat_dataset_model = train_model(new_model, criterion, add_class_optimizer, exp_lr_scheduler, own_concat_class_dls, own_concat_class_dataset_sizes, device, num_epochs=num_epochs)

        if save_weights:
            classes_dict = own_concat_dataset.get_classes_names()
            save_model(model_name+'_after_add_class', model_name, own_concat_dataset_model.state_dict(), classes_dict)
            save_layers_except_last(own_concat_dataset_model, model_name)

        if eveluate:
            evaluate_model_with_limited_misclassified(own_concat_dataset_model, own_concat_class_dls['test'], criterion, device, list(own_concat_dataset.get_classes_names().values()))
# %%