import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel, GATConv, global_mean_pool
import time
import argparse


def create_dataset():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '/lus/eagle/projects/datascience/zzheng/gnn_data', 'MNIST')
    dataset = MNISTSuperpixels(path, transform=T.Cartesian()).shuffle()
    dataset = dataset[:len(dataset) // 3]
    return dataset


def build_model(dataset, GPU_selection):
    class GATnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(dataset.num_features, 32, heads=8, dropout=0.6)
            self.conv2 = GATConv(32 * 8, dataset.num_classes, heads=1, concat=False, dropout=0.6)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            x = global_mean_pool(x, data.batch)
            return F.log_softmax(x, dim=1)
        
    
    
    model = GATnet()
    GPU_selection = [int(gpu) for gpu in args.GPU_selection.split(",")]
    device = torch.device("cuda:" + str(GPU_selection[0]) if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if GPU_selection:
        model = DataParallel(model, device_ids=[int(gpu) for gpu in GPU_selection])   
    else:
        model = DataParallel(model)
        
    return model


def train(args):
    dataset = create_dataset()
    loader = DataListLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.number_worker)
    model = build_model(dataset, args.GPU_selection)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(args.number_epochs):
        start_time = time.time()
        for data_list in loader:
            optimizer.zero_grad()
            output = model(data_list)
            y = torch.cat([data.y for data in data_list]).to(output.device)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        end_time = time.time()
        throughput = len(loader.dataset) / (end_time - start_time)
        print(f"Epoch {epoch + 1} completed. Elapsed time: {end_time - start_time:.2f} Throughput: {throughput:.2f} samples/s")


def main(args):
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT Multi-GPU Training")
    parser.add_argument("--GPU_selection", type=str, default="2,3", help="Select GPUs to run on (default: '0')")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (default: 1024)")
    parser.add_argument("--number_worker", type=int, default=4, help="Number of workers (default: 4)")
    parser.add_argument("--number_epochs", type=int, default=5, help="Number of epochs (default: 5)")

    args = parser.parse_args()
    main(args)
