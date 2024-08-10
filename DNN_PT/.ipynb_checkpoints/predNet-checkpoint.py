import os
import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, Subset

data_dir = "/lus/eagle/projects/datascience/ImageNet/ILSVRC/Data/CLS-LOC/train"
model_dir = "/lus/eagle/projects/datascience/zzheng/model"

num_images = 10000  # Set the number of images you want to predict

def download_and_save_model(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'resnet50_imagenet.pth')
    
    if os.path.exists(model_path):
        model = models.resnet50()
        model.load_state_dict(torch.load(model_path))
    else:
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        torch.save(model.state_dict(), model_path)
    
    model.eval()
    return model

def load_model(model_dir):
    model_path = os.path.join(model_dir, 'resnet50_imagenet.pth')
    model = models.resnet50()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def main():
    # Download and save the pre-trained model to the specified directory
    model = download_and_save_model(model_dir)

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create a test dataset and DataLoader
    full_test_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=transform)
    test_dataset = Subset(full_test_dataset, range(num_images))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Perform inference on the test dataset
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.tolist())

    # print(f"Predicted class indices: {all_preds}")

if __name__ == "__main__":
    main()

