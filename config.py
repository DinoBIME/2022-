import os

"""Training 編號"""
num_train = 8888
"""About Path"""
# root path
data_dir = "/home/nas/Research_Group/Personal/Dino/dataset_4_1"
log_dir = "/home/ubuntu/crop_identification/version1_VanillaEfficientNEt/crop_model_6/log"
model_dir = "/home/ubuntu/crop_identification/version1_VanillaEfficientNEt/crop_model_6/model"

# I use ImageFolder so there is no label csv file
train_data = os.path.join(data_dir, "process_train")
test_data = os.path.join(data_dir, "process_test")

training_log = os.path.join(log_dir, "traing_log_{}.txt".format(num_train))
testing_log = os.path.join(log_dir, "testing_log_{}.txt".format(num_train))

save_model = os.path.join(model_dir, "model_{}.pth".format(num_train))
load_model = "/home/ubuntu/crop_identification/version1_VanillaEfficientNEt/crop_model_6/log/history_model_8888/model_10"

eval_path = os.path.join(log_dir, "eval_{}.txt".format(num_train))

cmt_path = "/home/ubuntu/crop_identification/version1_VanillaEfficientNEt/crop_model_6/log/confusion_matrix/confusion_matrix_{}.svg".format(num_train)

"""About Hyperparameters"""
input_size = 1024
batch_size = 8
num_workers = 4
num_classes = 14
learning_rate = 0.0005
momentum = 0.9
use_pretrained = True
epochs = 100
log_interval = 1
