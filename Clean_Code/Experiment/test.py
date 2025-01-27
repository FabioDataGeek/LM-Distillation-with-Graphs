from dataloader import *
from torch_geometric.data import DataLoader as GeometricDataLoader
from torch_geometric.utils import add_remaining_self_loops, add_self_loops
from torch_geometric.data import Batch
from tqdm import tqdm

""" from torch.utils.data import DataLoader as GeometricDataLoader """

""" def my_collate(data_list):
    # Separate the attributes for each graph
    x1_list = [data.x1 for data in data_list]
    edge_index1_list = [data.edge_index1 for data in data_list]
    x2_list = [data.x2 for data in data_list]
    edge_index2_list = [data.edge_index2 for data in data_list]
    y_list = [data.y for data in data_list]

    # Create a Batch object for each graph
    batch1 = Batch.from_data_list([Data(x=x1, edge_index=edge_index1) for x1, edge_index1 in zip(x1_list, edge_index1_list)])
    batch2 = Batch.from_data_list([Data(x=x2, edge_index=edge_index2) for x2, edge_index2 in zip(x2_list, edge_index2_list)])

    # Return a tuple of the two Batch objects and the target tensor
    return batch1, batch2, torch.stack(y_list) """



dataset = Dataset_RGNN(root='/usrvol/processed_tensors/SNLI/dev/semantic/bert-base-uncased', 
                       files_path='/usrvol/processed_tensors/SNLI/dev/semantic/bert-base-uncased/raw', 
                       semantic=True)

""" # check if Nan or Inf values in tensors
for batch in dataset:
    for minibatch in batch:
        if torch.isnan(minibatch.x1).any() or torch.isinf(minibatch.x1).any():
            print("NaN or Inf found in x1")
        if torch.isnan(minibatch.edge_index1).any() or torch.isinf(minibatch.edge_index1).any():
            print("NaN or Inf found in edge_index1")
        if torch.isnan(minibatch.x2).any() or torch.isinf(minibatch.x2).any():
            print("NaN or Inf found in x2")
        if torch.isnan(minibatch.edge_index2).any() or torch.isinf(minibatch.edge_index2).any():
            print("NaN or Inf found in edge_index2")
        if torch.isnan(minibatch.y).any() or torch.isinf(minibatch.y).any():
            print("NaN or Inf found in y") """
        

""" # check if there are None values
for batch in dataset:
    for minibatch in batch:
        if minibatch.x1 is None:
            print("None found in x1")
        if minibatch.edge_index1 is None:
            print("None found in edge_index1")
        if minibatch.x2 is None:
            print("None found in x2")
        if minibatch.edge_index2 is None:
            print("None found in edge_index2")
        if minibatch.y is None:
            print("None found in y") """

""" # check __len__ method
print(len(dataset)) """

""" # check indices of graphs
for batch in dataset:
    for minibatch in batch:
        if minibatch.edge_index1.numel() != 0:
            if minibatch.edge_index1.min() < 0 or minibatch.edge_index1.max() >= minibatch.x1.size(0):
                print("Invalid indices in edge_index1")
        if minibatch.edge_index2.numel() != 0:
            if minibatch.edge_index2.min() < 0 or minibatch.edge_index2.max() >= minibatch.x2.size(0):
                print("Invalid indices in edge_index2") """


""" # check the datatypes of tensors
for batch in dataset:
    for minibatch in batch:
        if minibatch.x1.dtype != torch.float32:
            print("Invalid datatype in x1")
        if minibatch.edge_index1.dtype != torch.int64:
            print("Invalid datatype in edge_index1")
        if minibatch.x2.dtype != torch.float32:
            print("Invalid datatype in x2")
        if minibatch.edge_index2.dtype != torch.int64:
            print("Invalid datatype in edge_index2")
        if minibatch.y.dtype != torch.int64:
            print("Invalid datatype in y") """


""" #Add self-loop to all the Data items
for file in range(len(dataset)):
    for data in range(len(dataset[file])):
        dataset[file][data].edge_index1, _ = add_self_loops(dataset[file][data].edge_index1, num_nodes=dataset[file][data].x1.size(0))
        dataset[file][data].edge_index2, _ = add_self_loops(dataset[file][data].edge_index2, num_nodes=dataset[file][data].x2.size(0)) """
""" dataloader = GeometricDataLoader(dataset[0], batch_size=1, shuffle=True, collate_fn=my_collate)"""
for file in tqdm(dataset):
    dataloader = HeteroGeneousDataLoader(file, batch_size=16, shuffle=True)
    for batch1, batch2, y in dataloader:
        continue