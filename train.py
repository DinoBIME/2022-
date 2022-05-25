"""This is a template that I rewrite at 2021/9/26"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transform
from efficientnet_pytorch import EfficientNet
import config as cfg

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

### set device
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

### set the hyperparameters
VERSION = cfg.num_train
# PATH
TRAIN_DATA = cfg.train_data
TEST_DATA = cfg.test_data
LOG_DIR = cfg.log_dir
TRAINING_LOG = cfg.training_log
TESTING_LOG = cfg.testing_log
SAVE_MODEL = cfg.save_model

# PARAMETERS
INPUT_SIZE = cfg.input_size
BATCH_SIZE = cfg.batch_size
EPOCH = cfg.epochs
LR = cfg.learning_rate
MOMENTUM = cfg.momentum

NUM_CLASSES = cfg.num_classes
NUM_WORKERS = cfg.num_workers

PRETRAINED = cfg.use_pretrained
LOG_INTERVAL = cfg.log_interval

### Make the directory to save all the model
n = 'history_model_{}'.format(VERSION)
os.makedirs(os.path.join(LOG_DIR,n),exist_ok=True)

### Define the model
def efficientnet_model(model_name):
    model = EfficientNet.from_pretrained(model_name, advprop=True)
    num_ftr = model._fc.in_features
    model._fc = nn.Linear(num_ftr, NUM_CLASSES)

    # freeze the layers
    for param_stem in model._conv_stem.parameters():
        param_stem.requires_grad = False
        print("_conv_stem is freezed")

    for param_bn0 in model._bn0.parameters():
        param_bn0.requires_grad = False
        print("bn0 is freezed")

    child_counter = 0
    for child in model._blocks.children():
        if child_counter < 36:
            for param in child.parameters():
                param.requires_grad = False
            print("model._blocks child {} is freezed".format(child_counter))
        child_counter += 1

    print(model)

    model = model.to(device)

    print("have dropout? {}".format(model._global_params.include_top))
    return model


### Function of training

def train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epoch):

    test_accuracy = 0
    best_model = model

    for epoch in range(num_epoch):
        total_loss = 0
        accuracy = 0
        count = 0
        
        for batch_idx, (data, label) in enumerate(train_loader):

            # move data to device
            data = data.to(device=device)
            label = label.to(device=device)

            # get the forward result
            result = model(data)
            loss = criterion(result, label)

            _, predicted_label = torch.max(result.data, 1)
            count += len(data)
            accuracy += (predicted_label == label).sum().item()
            total_loss += loss.item() * len(label)

            # do backward propgation
            optimizer.zero_grad() # This is very important!! we just wanna focus on current batch
            loss.backward()

            # update the weights by optimizer
            optimizer.step()

        # Using Scheduler
        scheduler.step()

        # [Print accuracy] =================================
        print("[{}/{}] | Loss: {} | Accuracy: {}".format(epoch, num_epoch, total_loss/count, accuracy/count))
        # write in the log file
        with open(TRAINING_LOG, "a+") as train_log:
            print("{} is batch size?".format(count))
            train_log.write("Epoch: {} \n".format(num_epoch))
            train_log.write("Training Loss: {} \n".format(total_loss / count))
            train_log.write("Training Accuracy: {} \n\n".format(accuracy / count))

        # [Find if the current model is the best] ==========
        if (epoch % LOG_INTERVAL == 0):
            temp_accuracy = test(model, test_loader, criterion, epoch)
            if (temp_accuracy > test_accuracy):
                test_accuracy = temp_accuracy
                best_model = model
            
        
        torch.save(model, os.path.join(LOG_DIR, "history_model_{}".format(VERSION), "model_{}".format(epoch)))
        # [Save the model] ===========
        torch.save(best_model, SAVE_MODEL)

    return model

### check accuracy
def test(model, test_loader, criterion, num_epoch):
    model.eval() # the most important part!!it will skip dropout, batch norm. etc.
    total_loss = 0
    accuracy = 0
    count = 0

    for data, label in test_loader:  # label is for prediction, target is the answer(my definition)
        # move data to device
        data = data.to(device=device)
        label = label.to(device=device)
        # data = data.reshape(data.shape[0], -1)

        # forward
        result = model(data)
        # loss
        loss = criterion(result, label)
        # get the model predict result -> max score of target vector
        _,predicted_label = torch.max(result.data, 1)

        ## note: label is the vector of correct answer, and predicted_label is a vector that model predict
        count += len(data)
        accuracy += (predicted_label==label).sum().item()
        total_loss += loss.item()*len(label)

    # write in the log file
    with open(TESTING_LOG, "a+") as test_log:
        test_log.write("Epoch: {} \n".format(num_epoch))
        test_log.write("Testing Loss: {} \n".format(total_loss / count))
        test_log.write("Testing Accuracy: {} \n\n".format(accuracy / count))

    return (accuracy / count) # just return the accuracy when calling this function (we will need it in training function)

### Define the own dataset if you need
def dataset():
    pass

def main():
    ### Data Augmantation
    train_transform = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Lambda(lambda img: img * 2.0 - 1.0)
    	])

    test_transform = transform.Compose([
        transform.ToTensor(),
        transform.Lambda(lambda img: img * 2.0 - 1.0)
    	])

    ### Instance the dataset and dataloader
    train_dataset = datasets.ImageFolder(TRAIN_DATA, train_transform)
    test_dataset = datasets.ImageFolder(TEST_DATA, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    ### Instance the model object
    model = efficientnet_model("efficientnet-b7")

    ### Optimizer and Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.0000000001)

    ### Train the model
    model_fit = train(model, train_loader, test_loader,
                      criterion, optimizer, scheduler, EPOCH)

if __name__ == "__main__":
    main()

