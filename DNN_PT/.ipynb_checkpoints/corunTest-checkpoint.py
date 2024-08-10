import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet101, resnet50, vgg16, alexnet, inception_v3,resnet152
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import time
import os
import random
import argparse
import csv
import torch.nn.functional as F



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


def create_dataset(model_name, dataset):
    if dataset == "cifar100":
        data_dir = "/lus/eagle/projects/datascience/ImageNet/ILSVRC/Data/CLS-LOC"
    else:
        data_dir = "/lus/eagle/projects/datascience/ImageNet/ILSVRC/Data/CLS-LOC"

    if model_name == "LeNet":
        input_size = 32
    elif model_name == "Inception-V3":
        input_size = 299
    else:
        input_size = 224

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if dataset == "cifar100":
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)

    else:
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)

    num_samples = 10000
    train_indices = get_random_subset_indices(num_samples, len(train_dataset))
    small_train_dataset = Subset(train_dataset, train_indices)
    return small_train_dataset



def build_model(model_name, num_classes, GPU_selection, share):
    if model_name == "ResNet-101":
        model = resnet101(weights=None)
    elif model_name == "ResNet-50":
        model = resnet50(weights=None)
    elif model_name == "ResNet-152":
        model = resnet152(weights=None)
    elif model_name == "VGG-16":
        model = vgg16(weights=None)
    elif model_name == "AlexNet":
        model = alexnet(weights=None)
    elif model_name == "LeNet":
        model = LeNet(num_classes=num_classes)
    elif model_name == "Inception-V3":
        model = inception_v3(weights=None, aux_logits=True)
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
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


def create_dataLoader(model_name, dataset, batch_size=256, workers=8):
    return DataLoader(create_dataset(model_name, dataset), batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True)




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



def train(model_name, dataset, batch_size=256, GPU_selection=[0, 1], epoch=5, num_workers=8, output="default", share=0):
    num_epochs = epoch
    train_loader = create_dataLoader(model_name, dataset, batch_size, num_workers)

    if dataset == "cifar100":
        num_classes = 1000
    else:
        num_classes = 1000

    model, criterion, optimizer, device = build_model(model_name, num_classes=num_classes, GPU_selection=GPU_selection, share=share)

   
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
    model_name = args.model_name
    batch_size = args.batch_size
    num_workers = args.number_worker
    output = args.output
    GPU_selection = [int(gpu) for gpu in args.GPU_selection.split(",")]
    node_share = args.share
    dataset = args.dataset
    
    if(node_share==0):
        select_device(GPU_selection)

    train(model_name, dataset=dataset, batch_size=batch_size, GPU_selection=GPU_selection, num_workers=num_workers, output=output, share=node_share, epoch=args.epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="LeNet", help="Choose the model: Inception-V3, ResNet-101, ResNet-50, VGG-16, AlexNet, LeNet")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 256)")
    parser.add_argument("--number_worker", type=int, default=8, help="Number of workers for data loading (default: 8)")
    parser.add_argument("--GPU_selection", type=str, default="0,1", help="Comma-separated list of GPU indices to use (default: 0,1)")
    parser.add_argument("--output", type=str, default="result", help="speficy the csv output file")
    parser.add_argument("--share", type=int, default=1, help="")
    parser.add_argument("--epoch", type=int, default=5, help="")
    parser.add_argument("--dataset", type=str, default="imagenet", help="Choose the dataset: imagenet, cifar100")

    

    args = parser.parse_args()
    main(args)
