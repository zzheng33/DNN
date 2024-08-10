import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import time
import os
from torch.utils.data import Subset
import random
import torch.nn.functional as F
import argparse

# Define the LeNet architecture
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

# Update the build_model function
def build_model():
    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = nn.DataParallel(model)
    return model, criterion, optimizer, device

# Update the data augmentation and normalization for training
def create_dataset():
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.ImageFolder(os.path.join("/lus/eagle/projects/datascience/ImageNet/ILSVRC/Data/CLS-LOC", "train"), transform=train_transform)

    num_samples = 10000

    train_indices = get_random_subset_indices(num_samples, len(train_dataset))
    small_train_dataset = Subset(train_dataset, train_indices)
    return small_train_dataset


def get_random_subset_indices(num_samples, dataset_size):
    return random.sample(range(dataset_size), num_samples)


def create_dataLoader(batch_size=256, workers=8):
    return DataLoader(create_dataset(), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

def select_device(selected_gpus):
     os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    
    total_images = 0
    start_time = time.time()
    start_time_dataLoad = time.time()
    end_time_dataLoad = 0
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_images += inputs.size(0)

    end_time = time.time()
    images_per_second = total_images / (end_time - start_time)
    dataLoad_time = end_time_dataLoad - start_time_dataLoad
    return int(images_per_second), end_time-start_time

import csv

def train(batch_size=256, GPU_selection=[0, 1], epoch=5, num_workers=8):
    
    num_epochs = epoch  # Adjust this value according to your needs
    train_loader = create_dataLoader(batch_size, num_workers)
    select_device(GPU_selection)
    model, criterion, optimizer, device = build_model()
    
    images_per_second_list = []
    for epoch in range(num_epochs):
        images_per_second, epoch_duration = train_one_epoch(model, criterion, optimizer, train_loader, device)
        if epoch > 0:  # Skip the first epoch
            images_per_second_list.append(images_per_second)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Duration: {epoch_duration:.2f}s, Images/s: {images_per_second}")

    avg_images_per_second = sum(images_per_second_list) / len(images_per_second_list)

    # Save the results to a CSV file
    model_name = "ResNet-50"
    num_gpus = len(GPU_selection)
    result = [model_name, batch_size, num_workers, num_gpus, avg_images_per_second]

    # Check if the CSV file exists and create it if it doesn't
    csv_file = "./result/training_results.csv"
    if not os.path.isfile(csv_file):
        with open(csv_file, "w", newline="") as f:
            csv_writer = csv.writer(f)
            header = ["Model", "Batch Size", "Num Workers", "Num GPUs", "Average Images/s"]
            csv_writer.writerow(header)

    # Append the results to the CSV file
    with open(csv_file, "a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(result)
        
def main(args):
    batch_size = args.batch_size
    num_workers = args.number_worker
    GPU_selection = [int(gpu) for gpu in args.GPU_selection.split(",")]

    train(batch_size=batch_size, GPU_selection=GPU_selection, num_workers=num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LeNet on ImageNet")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training (default: 256)")
    parser.add_argument("--number_worker", type=int, default=8, help="Number of workers for data loading (default: 8)")
    parser.add_argument("--GPU_selection", type=str, default="0,1", help="Comma-separated list of GPU indices to use (default: 0,1)")

    args = parser.parse_args()
    main(args)