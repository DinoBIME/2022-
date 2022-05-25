import itertools
import numpy as np
import matplotlib.pyplot as plt
import PIL

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transform

from efficientnet_pytorch import EfficientNet

import config as cfg

import csv
from torchvision.io import read_image

### set device
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

### set the hyperparameters
# PATH
TRAIN_DATA = cfg.train_data
TEST_DATA = cfg.test_data
LOG_DIR = cfg.log_dir
TRAINING_LOG = cfg.training_log
TESTING_LOG = cfg.testing_log
SAVE_MODEL = cfg.save_model
LOAD_MODEL = cfg.load_model

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

CMT_PATH = cfg.cmt_path


### Define the model
def efficientnet_model(model_name):
    model = EfficientNet.from_pretrained(model_name)
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
        if child_counter < 19:
            print(f"model._blocks child {child_counter} is freezed")
        child_counter += 1

    model = model.to(device)

    print("have dropout? {}".format(model._global_params.include_top))
    return model

    # {    ,  'soybean': 11, 'sugarcane': 12, 'tamato': 13}

def to_labelname(label):

    label_name = ""

    if label == "0":
        label_name='banana'
    elif label == "1":
        label_name='bareland'
    elif label == "2":
        label_name='carrot'
    elif label == "3":
        label_name='corn'
    elif label == "4":
        label_name='dragonfruit'
    elif label == "5":
        label_name='garlic'
    elif label == "6":
        label_name='guava'
    elif label == "7":
        label_name='peanut'
    elif label == "8":
        label_name='pineapple'
    elif label == "9":
        label_name='pumpkin'
    elif label == "10":
        label_name='rice'
    elif label == "11":
        label_name='soybean'
    elif label == "12":
        label_name='sugarcane'
    elif label == "13":
        label_name='tomato'
    
    return label_name


### check accuracy
def test(model, test_loader):
    model.eval() # the most important part!!it will skip dropout, batch norm. etc.
    total_loss = 0
    accuracy = 0
    count = 0

    csvfile = open('result.csv', 'w', newline='')
    writer = csv.writer(csvfile)

    writer.writerow(["image_filename", "label"])

    for i, (path,data) in enumerate(test_loader, 0):  # label is for prediction, target is the answer(my definition)

        data = data.to(device=device)

        # fname = test_loader.dataset.samples
        batch_size = 10
        # fname_batch = fname[i*batch_size:i*batch_size+10]
        # print(path)
        # print(len(path))

        result = model(data)
        _,predicted_label = torch.max(result.data, 1)

        for t in range(len(predicted_label)):
            temp_label_name = to_labelname(str(predicted_label[t].item()))
            # print(predicted_label[t].item())
            # print(temp_label_name)
            writer.writerow([path[t], temp_label_name])

    csvfile.close()
        
    # return (accuracy / count) # just return the accuracy when calling this function (we will need it in training function)

# write by Lin Yun
def create_cmt(val_loader, model_ft, classes):
    model_ft = model_ft.eval()

    class_correct = list(0. for i in range(NUM_CLASSES))
    class_total = list(0. for i in range(NUM_CLASSES))
    val_cmt = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

    fname = val_loader.dataset.samples

    l = []
    p = []

    f = open("/home/ubuntu/crop_identification/version1_VanillaEfficientNEt/crop_model_6/log/misclassification/miss.txt", "a")

    for k, (inputs, labels) in enumerate(val_loader, 0):
        batch_size = 10
        fname_batch = fname[k*batch_size:k*batch_size+10]
        # inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        _, predicted = torch.max(outputs, 1)

        l.extend(labels)
        p.extend(predicted)

        c = (predicted == labels).squeeze()

        for r in range(len(c)):
            if c[r] == False:
                f.write(fname_batch[r][0])
                f.write("    predict"+"{}".format(predicted[r])+"\n")

        for i in range(len(labels)):
            label = int(labels[i])
            val_cmt[label, predicted[i]] += 1
            class_correct[label] += c[i].item()
            class_total[label] += 1

    plot_confusion_matrix(val_cmt, classes, normalize=True, title='Confusion Matrix', cmap=plt.cm.YlGn)

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.style.use("default")
#     print(cm)
    if normalize:
        cm = cm.numpy()
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=8)
        
    # save_path = cmt_path
    # os.makedirs(save_path, exist_ok=True)
        
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(CMT_PATH)

class val_dataset(Dataset):
    def __init__(self, path, trans):
        self.root = path
        self.lst = os.listdir(path)
        self.trans = trans

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        img_path = self.lst[idx]
        data = PIL.Image.open(self.root + img_path)
        data = self.trans(data)

        return img_path, data


def main():
    ### Data Augmantation
    test_transform = transform.Compose([
        # transform.Resize((INPUT_SIZE, INPUT_SIZE)),
        transform.ToTensor(),
        transform.Lambda(lambda img: img * 2.0 - 1.0)
    	])

    ### Instance the dataset and dataloader
    test_dataset = datasets.ImageFolder("/home/nas/Research_Group/Personal/Dino/dataset_3/process_test/", test_transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=4)

    ### Instance the model object
    model = torch.load(LOAD_MODEL).to(device)
    model = nn.DataParallel(model, device_ids=[0,1])

    ### Classes list
    classes = ["banana","bareland","carrot","corn","dragonfruit","garlic","guava","peanut","pineapple","pumpkin", 'rice', 'soybean', 'sugarcane', 'tamato']

    test(model, test_loader)

    # create_cmt(test_loader, model, classes)

if __name__ == "__main__":
    main()