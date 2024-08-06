import torch
from torch_geometric.loader import DataLoader
from dataset import MoleculeDataset
from tqdm import tqdm
import numpy as np
import mlflow.pytorch
from utils import (count_parameters, gvae_loss, 
        slice_edge_type_from_edge_feats, slice_atom_type_from_node_feats,
        evaluate_generated_mols)
from gvae import GVAE
from config import DEVICE as device

# Specifically suppress MLflow warnings, you can set the logging level for MLflow:
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)

# Load data
# train_dataset = MoleculeDataset(root="data/", filename="HIV_train_oversampled.csv")[:10000]
# test_dataset = MoleculeDataset(root="data/", filename="HIV_test.csv", test=True)[:1000]
# train_dataset = MoleculeDataset(root="data/", filename="Log_P_modified_with_effective.csv")[:800]
# test_dataset = MoleculeDataset(root="data/", filename="Log_P_modified_with_effective.csv", test=True)[801:]

train_dataset = MoleculeDataset(root="data/", filename="Log_P_modified_with_effective.csv")
train_dataset = [data for data in train_dataset if data.y == 1]
test_dataset = MoleculeDataset(root="data/", filename="Log_P_modified_with_effective.csv", test=True)
test_dataset = [data for data in test_dataset if data.y == 1]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Load model
model = GVAE(feature_size=train_dataset[0].x.shape[1])
model = model.to(device)
print("Model parameters: ", count_parameters(model))

# Define loss and optimizer
loss_fn = gvae_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
kl_beta = 0.001  #0.5

# Train function
def run_one_epoch(data_loader, type, epoch, kl_beta):
    # Store per batch loss and accuracy 
    all_losses = []
    all_kldivs = []

    # Iterate over data loader
    for _, batch in enumerate(tqdm(data_loader)):
        # Some of the data points have invalid adjacency matrices 
        try:
            # Use GPU
            batch.to(device)  
            # Reset gradients
            optimizer.zero_grad() 
            # Call model
            triu_logits, node_logits, mu, logvar = model(batch.x.float().to(device), 
                                                        batch.edge_attr.float().to(device),
                                                        batch.edge_index.to(device), 
                                                        batch.batch.to(device)) 
            # Calculate loss and backpropagate
            edge_targets = slice_edge_type_from_edge_feats(batch.edge_attr.float().to(device))
            node_targets = slice_atom_type_from_node_feats(batch.x.float().to(device), as_index=True)
            loss, kl_div = loss_fn(triu_logits, node_logits,
                                   batch.edge_index, edge_targets, 
                                   node_targets, mu, logvar, 
                                   batch.batch, kl_beta)
            if type == "Train":
                loss.backward()  
                optimizer.step() 
            # Store loss and metrics
            all_losses.append(loss.detach().cpu().numpy())
            #all_accs.append(acc)
            all_kldivs.append(kl_div.detach().cpu().numpy())
        except IndexError as error:
            # For a few graphs the edge information is not correct
            # Simply skip the batch containing those
            print("Error: ", error)
    
    # Perform sampling
    if type == "Test":
        generated_mols, num_valid = model.sample_mols(num=100, device=device)
        valid_ratio = num_valid / 100
        print(f"Generated {generated_mols} valid molecules out of 100 attempts.")
        mlflow.log_metric(key=f"Valid Generation Ratio", value=valid_ratio, step=epoch)
    

    print(f"{type} epoch {epoch} loss: ", np.array(all_losses).mean())
    mlflow.log_metric(key=f"{type} Epoch Loss", value=float(np.array(all_losses).mean()), step=epoch)
    mlflow.log_metric(key=f"{type} KL Divergence", value=float(np.array(all_kldivs).mean()), step=epoch)
    mlflow.pytorch.log_model(model, "model")

# Run training
with mlflow.start_run() as run:
    for epoch in range(100):
        model.train()
        run_one_epoch(train_loader, type="Train", epoch=epoch, kl_beta=kl_beta)
        if epoch % 5 == 0:
            print("Start evaluation...")
            model.eval()
            generated_mols, _ = model.sample_mols(num=100, device=device)
            eval_metrics = evaluate_generated_mols(generated_mols, [data.smiles for data in train_dataset])
            mlflow.log_metrics(eval_metrics, step=epoch)
            if test_loader:
                run_one_epoch(test_loader, type="Test", epoch=epoch, kl_beta=kl_beta)
            else:
                print("Skipping test evaluation due to lack of test data.")