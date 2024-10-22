import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm
from enum import Enum
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2 as transformsV2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter
import seaborn as sn
import pandas as pd
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix
import itertools
import random
from IPython.display import clear_output
# from NetworkFineTuning import MultiModel, FineTuneMode, Hotdog_NotHotdog


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)

# Function to create all combinations of hyperparameters


def create_combinations(hyperparameter_grid):
    keys, values = zip(*hyperparameter_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

# Function to randomly sample hyperparameters


def sample_hyperparameters(hyperparameter_grid, num_samples):
    samples = []
    for _ in range(num_samples):
        sample = {}
        for key, values in hyperparameter_grid.items():
            sample[key] = random.choice(values)
        samples.append(sample)
    return samples


# TODO: Test out this as the numbers are the same every time for some reason
def check_accuracy(model, dataloader, device, batch_size):
    model.eval()
    num_correct = 0
    num_pixels = 0
    y_true = []
    y_pred = []

    image,label = [],[]

    with torch.no_grad():
        for data in dataloader:
            
            if len(data) == 3:
                image, label = data[0], data[1]
            else:
                image, label = data
            label = label.type(torch.LongTensor)

            image = image.to(device)
            label = label.to(device)

            scores = model(image)

            _, predictions = scores.max(1)

            num_correct += (predictions == label).sum()
            
            num_pixels += label.numel()
            y_pred.extend(predictions.cpu().tolist())  # Save Prediction
            label = label.data.cpu().numpy()
            y_true.extend(label)  # Save Truth
    num_pixels *= batch_size
    accuracy = float(num_correct)/float(num_pixels)
    print(
        f"Got {num_correct}/{num_pixels} with accuracy {accuracy * 100:.3f}%\n\n")
    model.train()
    return accuracy


def train_mod(model, logger, hyper_parameters, modeltype, device, loss_function, dataloader_train, dataloader_validation, directory):

    optimizer, scheduler = set_optimizer_and_scheduler(hyper_parameters, model)


    epochs = hyper_parameters["epochs"]
    all_train_losses = []
    all_val_losses = []
    all_accuracies = []
    validation_loss = 0

    images,labels = [],[]

    for epoch in range(epochs):  # loop over the dataset multiple times

        """    Train step for one batch of data    """
        training_loop = create_tqdm_bar(
            dataloader_train, desc=f'Training Epoch [{epoch+1}/{epochs}]')

        training_loss = 0
        model.train()  # Set the model to training mode
        train_losses = []
        accuracies = []
        
        for train_iteration, batch in training_loop:
            optimizer.zero_grad()  # Reset the parameter gradients for the current minibatch iteration

            if len(batch) == 3:
                images, labels = batch[0], batch[1]
            else:
                images, labels = batch
            labels = labels.type(torch.LongTensor)

            labels = labels.to(device)
            images = images.to(device)

            # Forward pass, backward pass and optimizer step
            predicted_labels = model(images)
            loss_train = loss_function(labels, predicted_labels)
            loss_train.backward()
            optimizer.step()

            # Accumulate the loss and calculate the accuracy of predictions
            training_loss += loss_train.item()
            train_losses.append(loss_train.item())

            # Running train accuracy
            _, predicted = predicted_labels.max(1)
            num_correct = (predicted == labels).sum()
            train_accuracy = float(num_correct)/float(images.shape[0])
            accuracies.append(train_accuracy)

            training_loop.set_postfix(train_loss="{:.8f}".format(
                training_loss / (train_iteration + 1)), val_loss="{:.8f}".format(validation_loss))

            logger.add_scalar(f'Train loss', loss_train.item(
            ), epoch*len(dataloader_train)+train_iteration)
            logger.add_scalar(f'Train accuracy', train_accuracy, epoch*len(dataloader_train)+train_iteration)
        all_train_losses.append(sum(train_losses)/len(train_losses))
        all_accuracies.append(sum(accuracies)/len(accuracies)) 

        """    Validation step for one batch of data    """
        val_loop = create_tqdm_bar(
            dataloader_validation, desc=f'Validation Epoch [{epoch+1}/{epochs}]')
        validation_loss = 0
        val_losses = []
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for val_iteration, batch in val_loop:
                
                if len(batch) == 3:
                    images, labels = batch[0], batch[1]
                else:
                    images, labels = batch
                    
                labels = labels.type(torch.LongTensor)

                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                output = model(images)

                # Calculate the loss
                loss_val = loss_function(labels, output)

                validation_loss += loss_val.item()
                val_losses.append(loss_val.item())

                val_loop.set_postfix(val_loss="{:.8f}".format(
                    validation_loss/(val_iteration+1)))

                # Update the tensorboard logger.
                logger.add_scalar(f'Validation loss', validation_loss/(
                    val_iteration+1), epoch*len(dataloader_validation)+val_iteration)
            all_val_losses.append(sum(val_losses)/len(val_losses))

        # This value is for the progress bar of the training loop.
        validation_loss /= len(dataloader_validation)

        logger.add_scalars(f'Combined', {'Validation loss': validation_loss,
                                                 'Train loss': training_loss/len(dataloader_train)}, epoch)
        if scheduler is not None:
            scheduler.step()
            print(f"Current learning rate: {scheduler.get_last_lr()}")

    if scheduler is not None:
        logger.add_hparams(
            {f"Step_size": scheduler.step_size, f'Batch_size': hyper_parameters["batch size"], f'Optimizer': hyper_parameters["optimizer"], f'Scheduler': hyper_parameters["scheduler"]},
            {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
                f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
                f'Avg val loss': sum(all_val_losses)/len(all_val_losses)}
        )
    else:
        logger.add_hparams(
            {f"Step_size": "None", f'Batch_size': hyper_parameters["batch size"], f'Optimizer': hyper_parameters["optimizer"], f'Scheduler': hyper_parameters["scheduler"]},
            {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
                f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
                f'Avg val loss': sum(all_val_losses)/len(all_val_losses)}
        )
    
    
    # Check accuracy and save model
    accuracy = check_accuracy(model, dataloader_validation, device, hyper_parameters['batch size'])
    save_dir = os.path.join(directory, f'accuracy_{accuracy:.3f}.pth')
    torch.save(model.state_dict(), save_dir)  # type: ignore

    return accuracy




def set_optimizer_and_scheduler(new_hp, model):
    if new_hp["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=new_hp["learning rate"],
                                     betas=(new_hp["beta1"],
                                            new_hp["beta2"]),
                                     weight_decay=new_hp["weight decay"],
                                     eps=new_hp["epsilon"])
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=new_hp["learning rate"],
                                    momentum=new_hp["momentum"],
                                    weight_decay=new_hp["weight decay"])
    if new_hp["scheduler"] == "Yes":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=new_hp["step size"], gamma=new_hp["gamma"])
    else:
        scheduler = None
    return optimizer, scheduler


def hyperparameter_search(model, modeltype, loss_function, device, dataset_train, dataset_validation, dataset_test, hyperparameter_grid, missing_hp, run_dir):
    # Initialize with a large value for minimization problems
    best_performance = 0
    best_hyperparameters = None
    run_counter = 0
    modeltype_directory = os.path.join(run_dir, f'{modeltype}')
    for hyper_parameters in hyperparameter_grid:
        # Empty memory before start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Current hyper parameters: {hyper_parameters}")
        hyper_parameters.update(missing_hp)
        # Initialize model, optimizer, scheduler, logger, dataloader
        dataloader_train = DataLoader(
            dataset_train, batch_size=hyper_parameters["batch size"], shuffle=True, num_workers=hyper_parameters["number of workers"], drop_last=False)
        print(f"Created a new Dataloader for training with batch size: {hyper_parameters['batch size']}")
        dataloader_validation = DataLoader(
            dataset_validation, batch_size=hyper_parameters["batch size"], shuffle=False, num_workers=hyper_parameters["number of workers"], drop_last=False)
        print(f"Created a new Dataloader for validation with batch size: {hyper_parameters['batch size']}")
        dataloader_test = DataLoader(
            dataset_test, batch_size=hyper_parameters["batch size"], shuffle=False, num_workers=hyper_parameters["number of workers"], drop_last=False)
        print(f"Created a new Dataloader for testing with batch size: {hyper_parameters['batch size']}")

        log_dir = os.path.join(modeltype_directory, f'run_{str(run_counter)}_{hyper_parameters["network name"]}_{hyper_parameters["optimizer"]}_Scheduler_{hyper_parameters["scheduler"]}')
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir)

        # accuracy = automatic_fine_tune(logger, hyper_parameters, modeltype, device,
        #                                loss_function, dataloader_train, dataloader_validation, dataloader_test, log_dir)
        accuracy = train_mod(model, logger, hyper_parameters, modeltype, device,
                             loss_function, dataloader_train, dataloader_validation, log_dir)

        run_counter += 1

        # Update best hyperparameters if the current model has better performance
        if accuracy > best_performance:
            best_performance = accuracy
            best_hyperparameters = hyper_parameters

        logger.close()
    print(f"\n\n############### Finished hyperparameter search! ###############")

    return best_hyperparameters

'''

train_transform = transformsV2.Compose([transformsV2.Resize(hyperparameters["image size"]),
                                        transformsV2.RandomVerticalFlip(p=0.5),
                                        transformsV2.RandomHorizontalFlip(
                                            p=0.5),
                                        transformsV2.RandomAdjustSharpness(
                                            sharpness_factor=2, p=0.5),
                                        transformsV2.RandomAutocontrast(p=0.5),
                                        transformsV2.ColorJitter(
                                            brightness=0.25, saturation=0.20),
                                        # Replace deprecated ToTensor()
                                        transformsV2.ToImage(),
                                        transformsV2.ToDtype(torch.float32, scale=True)])
test_transform = transformsV2.Compose([transformsV2.Resize(hyperparameters["image size"]),
                                       # Replace deprecated ToTensor()
                                       transformsV2.ToImage(),
                                       transformsV2.ToDtype(torch.float32, scale=True)])

os.environ['TORCH_HOME'] = hyperparameters["torch home"]
os.makedirs(hyperparameters["torch home"], exist_ok=True)

'''