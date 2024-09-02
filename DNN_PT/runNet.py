import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet101, resnet50, vgg16, alexnet, inception_v3, resnet152
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import time
import os
import random
import argparse
import csv
import torch.nn.functional as F
from torch.utils.data import Subset

class LeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_random_subset_indices(num_samples, dataset_size):
    return random.sample(range(dataset_size), num_samples)


def create_dataset(model_name):
    # Define your transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Load the full CIFAR-100 dataset
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',  # or any other directory you prefer
        train=True,
        download=True,
        transform=train_transform
    )
    subset_size=1000
    # If subset_size is specified, return a subset of the dataset
    if subset_size is not None and subset_size < len(train_dataset):
        subset_indices = list(range(subset_size))
        train_dataset = Subset(train_dataset, subset_indices)

    return train_dataset




def build_model(model_name,GPU_selection,share):
    if model_name == "ResNet-101":
        model = resnet101(pretrained=None)
    elif model_name == "ResNet-50":
        model = resnet50(pretrained=None)
    elif model_name == "ResNet-152":
        model = resnet152(pretrained=None)
    elif model_name == "VGG-16":
        model = vgg16(pretrained=None)
    elif model_name == "AlexNet":
        model = alexnet(pretrained=None)
    elif model_name == "LeNet":
        model = LeNet()
    elif model_name == "Inception-V3":
        model = inception_v3(pretrained=None, aux_logits=True)
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, 1000)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1000)
    else:
        raise ValueError("Invalid model name")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    device = torch.device("cuda:" + str(GPU_selection[0]) if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    if share==1:
        model = nn.DataParallel(model, device_ids=[int(gpu) for gpu in GPU_selection])   
    else:
        model = nn.DataParallel(model)
    return model, criterion, optimizer, device


def create_dataLoader(model_name, batch_size, num_workers):
    return DataLoader(create_dataset(model_name), batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)



def select_device(selected_gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))


def train_one_epoch(model_name, model, criterion, optimizer, data_loader, device):
    model.train()
    total_images = 0
    start_time = time.time()
    start_time_dataLoad = time.time()
    end_time_dataLoad = 0
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)


        optimizer.zero_grad()

        if model_name == "Inception-V3":
            outputs, aux_outputs = model(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels.to(outputs.device))


        loss.backward()
        optimizer.step()
        total_images += inputs.size(0)

    end_time = time.time()
    images_per_second = total_images / (end_time - start_time)
    dataLoad_time = end_time_dataLoad - start_time_dataLoad
    return int(images_per_second), end_time - start_time



def train(model_name, batch_size=256, GPU_selection=[0, 1], epoch=5, num_workers=8, output="default", share=0):
    num_epochs = epoch
    train_loader = create_dataLoader(model_name, batch_size, num_workers)
    
    model, criterion, optimizer, device = build_model(model_name, GPU_selection=GPU_selection, share=share)
   
    images_per_second_list = []
    for epoch in range(num_epochs):
        images_per_second, epoch_duration = train_one_epoch(model_name, model, criterion, optimizer, train_loader, device)
        if epoch > 0:  # Skip the first epoch
            images_per_second_list.append(images_per_second)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Duration: {epoch_duration:.2f}s, Images/s: {images_per_second}")

    avg_images_per_second = sum(images_per_second_list) / len(images_per_second_list)

    # Save the results to a CSV file
    num_gpus = len(GPU_selection)
    result = [model_name, batch_size, num_workers, num_gpus, avg_images_per_second]

    csv_file = "../result/" + output
    if not os.path.isfile(csv_file):
        with open(csv_file, "w", newline="") as f:
            csv_writer = csv.writer(f)
            header = ["Model", "Batch_Size", "Num_Workers", "Num_GPUs", "Images/s"]
            csv_writer.writerow(header)

    # Append the results to the CSV file
    with open(csv_file, "a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(result)




def main(args):
    
#     num_cpu_cores = 4

#     os.environ['OMP_NUM_THREADS'] = str(num_cpu_cores)
#     os.environ['OPENBLAS_NUM_THREADS'] = str(num_cpu_cores)
    
    model_name = args.model_name
    batch_size = args.batch_size
    num_workers = args.number_worker
    output = args.output
    GPU_selection = [int(gpu) for gpu in args.GPU_selection.split(",")]
    node_share = args.share
    epoch = args.epoch
    
    if(node_share==0):
        select_device(GPU_selection)

    train(model_name, batch_size=batch_size, GPU_selection=GPU_selection, num_workers=num_workers, output=output,share=node_share, epoch=epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ResNet-50", help="Choose the model: Inception-V3, ResNet-101, ResNet-50, VGG-16, AlexNet, LeNet")
    parser.add_argument("--batch_size", type=int, default=400, help="Batch size for training (default: 256)")
    parser.add_argument("--number_worker", type=int, default=4, help="Number of workers for data loading (default: 8)")
    parser.add_argument("--GPU_selection", type=str, default="0", help="Comma-separated list of GPU indices to use (default: 0,1)")
    parser.add_argument("--output", type=str, default="result", help="speficy the csv output file")
    parser.add_argument("--share", type=int, default=1, help="Node-sharing")
    parser.add_argument("--epoch", type=int, default=10, help="epochs")

    args = parser.parse_args()
    main(args)
