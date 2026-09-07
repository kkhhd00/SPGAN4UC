import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import math

def seed_everything(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.benchmark = True

seed_everything(42)

class SimpleSample:
    def __init__(self, x: torch.Tensor, y: torch.Tensor):

        self.x = x
        self.y = y

class SimpleBatch:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, num_graphs: int):

        self.x = x
        self.y = y
        self.num_graphs = num_graphs

    def to(self, device: torch.device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self

def collate_samples(batch: list):

    xs = [b.x for b in batch]
    ys = [b.y for b in batch]
    x_stack = torch.stack(xs, dim=0)
    y_stack = torch.stack(ys, dim=0)
    return SimpleBatch(x_stack, y_stack, num_graphs=len(batch))

def load_and_preprocess_data(tieline_path, thermal_path, renewable_path, combined_data_path, target_type='thermal'):

    print(f"Loading and preprocessing data for target type: {target_type}...")

    print("Reading thermal unit data...")
    thermal_df = pd.read_excel(thermal_path, header=0, index_col=0)

    thermal_data = thermal_df.values.reshape(91, 254, 96).transpose(0, 2, 1)

    print("Reading tie-line data...")
    tieline_df = pd.read_excel(tieline_path, header=0, index_col=0)
    print(tieline_df.shape)

    tieline_data = tieline_df.values.reshape(91, 4, 96).transpose(0, 2, 1)

    print("Reading target data...")

    df_from_excel = pd.read_excel(combined_data_path, index_col=[0, 1])

    dates = df_from_excel.index.get_level_values(0).unique()

    data_3d = np.array([df_from_excel.loc[date].values for date in dates])
    print(f"3D target data shape: {data_3d.shape}")

    if target_type == 'dispatch':
        indices = list(range(10, 20))
        target_data = data_3d[:, indices, :].transpose(0, 2, 1)
    elif target_type == 'thermal':
        indices = list(range(21, 32 + 1)) + list(range(34, 73 + 1)) + list(range(75, 80 + 1))
        target_data = data_3d[:, indices, :].transpose(0, 2, 1)
        print("target", target_data.shape)
    elif target_type == 'hydro':
        indices = list(range(83, 97 + 1))
        target_data = data_3d[:, indices, :].transpose(0, 2, 1)
    elif target_type == 'wind':
        indices = list(range(99, 231 + 1))
        target_data = data_3d[:, indices, :].transpose(0, 2, 1)
    elif target_type == 'solar':
        indices = list(range(233, 461 + 1))
        target_data = data_3d[:, indices, :].transpose(0, 2, 1)
    elif target_type == 'storage':
        indices = list(range(472, 504 + 1))
        target_data = data_3d[:, indices, :].transpose(0, 2, 1)
    else:
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 33, 74, 81, 82, 98, 232, 462, 463, 464, 465, 466, 467, 468, 469,
                   470, 471]
        target_data = data_3d[:, indices, :].transpose(0, 2, 1)

    print(f"Target data shape: {target_data.shape}")

    indices_new = list(range(99, 231 + 1)) + list(range(233, 461 + 1))
    renewable_new = data_3d[:, indices_new, :].transpose(0, 2, 1)

    print(f"Thermal unit data shape: {thermal_data.shape}")

    print(f"Target data shape: {target_data.shape}")

    target_scaler = 1
    return tieline_data, thermal_data, renewable_new, target_data, target_scaler

class PowerGraphDataset:
    def __init__(self, tieline_data, thermal_data, renewable_data, labels):
        self.thermal_data = torch.FloatTensor(thermal_data)
        self.renewable_data = torch.FloatTensor(renewable_data)
        self.tieline_Data = torch.FloatTensor(tieline_data)
        self.labels = torch.FloatTensor(labels)
        self.num_samples = thermal_data.shape[0] * thermal_data.shape[1]

        self.num_thermal_nodes = thermal_data.shape[2]
        self.num_renewable_nodes = renewable_data.shape[2]
        self.total_nodes = self.num_thermal_nodes + self.num_renewable_nodes

        edge_index = self._create_connected_edges(self.total_nodes)
        self.edge_index = edge_index
        print("______________________________", edge_index.shape)

        print(f"Edge index shape: {self.edge_index.shape}")

    def _create_connected_edges(self, num_nodes):

        node_df = pd.read_csv('nodes.csv', encoding='gb2312')
        load_df = pd.read_csv('loads.csv', encoding='gb2312')
        connection_df = pd.read_csv('connections.csv', encoding='gb2312')

        all_nodes = []
        all_nodes.extend(node_df['node_id'].tolist())
        all_nodes.extend(connection_df['source_node_id'].tolist())
        all_nodes.extend(connection_df['target_node_id'].tolist())
        all_nodes.extend(load_df['connected_node_id'].tolist())
        unique_nodes = sorted(list(set(all_nodes)))

        node_mapping = {old: i + 1 for i, old in enumerate(unique_nodes)}

        node_df['new_node_id'] = node_df['node_id'].map(node_mapping)
        load_df['new_connected_node_id'] = load_df['connected_node_id'].map(node_mapping)
        connection_df['new_source_node_id'] = connection_df['source_node_id'].map(node_mapping)
        connection_df['new_target_node_id'] = connection_df['target_node_id'].map(node_mapping)

        total_nodes = len(unique_nodes)
        new_nodes = [total_nodes + i + 1 for i in range(len(load_df))]
        load_connections = list(zip(new_nodes, load_df['new_connected_node_id']))

        original_connections = list(zip(connection_df['new_source_node_id'], connection_df['new_target_node_id']))
        final_connections = original_connections + load_connections

        result_matrix = [[], []]
        for start, end in final_connections:
            result_matrix[0].append(start)
            result_matrix[1].append(end)

        result_matrix = np.array(result_matrix)
        edge_index = torch.tensor(result_matrix, dtype=torch.long).t().contiguous()
        return edge_index.T

    def get_data(self, time_step):
        data_list = []
        for day in range(91):
            for time in range(96 - time_step + 1):
                thermal_features = self.thermal_data[day, time:time + time_step]
                renewable_features = self.renewable_data[day, time:time + time_step]
                tieline_features = self.tieline_Data[day, time:time + time_step]

                x_ts = torch.cat([thermal_features, renewable_features, tieline_features], dim=1)
                zeros = torch.zeros((time_step, 303), dtype=x_ts.dtype, device=x_ts.device)
                x_ts = torch.cat([zeros, x_ts], dim=1)

                y_ts = self.labels[day, time:time + time_step]
                sample = SimpleSample(x=x_ts, y=y_ts)
                data_list.append(sample)
        return data_list, self.edge_index

def forward_model(model, batch: SimpleBatch, edge_index: torch.Tensor, time_steps: int):

    return model(batch.x, edge_index)

def train_model(model, edge_index, train_loader, optimizer, device, time_steps):
    model.train()
    total_loss = 0
    total_l1_loss = 0

    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()

        torch.autograd.set_detect_anomaly(True)
        out = forward_model(model, data, edge_index.to(device), time_steps)

        y_out = data.y

        loss = F.mse_loss(out, y_out)

        l1loss = F.l1_loss(out, y_out)

        l1loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_l1_loss += l1loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset), total_l1_loss / len(train_loader.dataset)

def evaluate_model(model, edge_index, loader, device, time_steps):
    model.eval()
    total_loss = 0
    total_loss_L1 = 0

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            out = forward_model(model, data, edge_index.to(device), time_steps)

            y_out = data.y

            loss = F.mse_loss(out, y_out)
            loss_L1 = F.l1_loss(out, y_out)

            total_loss += loss.item() * data.num_graphs
            total_loss_L1 += loss_L1.item() * data.num_graphs

    return total_loss / len(loader.dataset), total_loss_L1 / len(loader.dataset)

def main(target_type='thermal', time_steps=12):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    hidden_dim = 128
    learning_rate = 0.001
    epochs = 100
    time_steps = time_steps

    thermal_path = "thermal_data.xlsx"
    renewable_path = "renewable_data.xlsx"
    combined_data_path = "target_data.xlsx"
    tieline_path = 'tieline_data.xlsx'

    model_dir = "model_outputs"
    model_name = "iTransformer_GAT"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    tieline_data, thermal_data, renewable_data, target_data, target_scaler = load_and_preprocess_data(
        tieline_path, thermal_path, renewable_path, combined_data_path, target_type
    )

    dataset, edge_index = PowerGraphDataset(tieline_data, thermal_data, renewable_data, target_data).get_data(
        time_step=time_steps)

    import random

    days = [dataset[i * 85:(i + 1) * 85] for i in range(91)]

    random_test_days = random.sample(range(91), 10)

    print("Selected test days:", sorted(random_test_days))

    test_dataset = [day for i, day in enumerate(days) if i in random_test_days]

    train_dataset = [day for i, day in enumerate(days) if i not in random_test_days]

    train_dataset = [sample for day in train_dataset for sample in day]
    test_dataset = [sample for day in test_dataset for sample in day]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_samples)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_samples)

    for data in train_loader:
        print("——————————————————————————————————————————————————————————————\n")
        print("x shape:", data.x.shape)
        print("y shape:", data.y.shape)
        print("Example x[0,0,:3]:", data.x[0, 0, :3])
        print("Example y[0,0,:3]:", data.y[0, 0, :3])
        print("——————————————————————————————————————————————————————————————\n")
        break

    input_dim = 1
    output_dim = target_data.shape[-1]

    from tsfmodel2.test_model.iTransformer_gat_dl import Model
    from tsfmodel2.get_config import get_config
    config = get_config('iTransformer')
    model = Model(config).to(device)

    print(f"- Input dimension: {input_dim} (features per node)")
    print(f"- Hidden dimension: {hidden_dim}")
    print(f"- Output dimension: {output_dim}")
    print(f"- Device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        train_loss, train_loss_L1 = train_model(model, edge_index, train_loader, optimizer, device, time_steps)
        test_loss, test_loss_L1 = evaluate_model(model, edge_index, test_loader, device, time_steps)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(
            f'Epoch: {epoch:03d}, train MSE: {train_loss:.4f}, train MAE: {train_loss_L1:.4f}, test MSE: {test_loss:.4f}, test MAE: {test_loss_L1:.4f}')

        scheduler.step(test_loss)

    model_path = os.path.join(model_dir, f'power_prediction_{target_type}_{model_name}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'target_scaler': target_scaler,
        'target_type': target_type,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim
    }, model_path)
    print(f"Model saved to {model_path}")

    model.to(device)
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Predicting test set"):
            data = data.to(device)
            out = forward_model(model, data, edge_index.to(device), time_steps)
            y_out = data.y

            predictions.append(out.cpu().numpy())
            labels.append(y_out.cpu().numpy())

    predictions_np = np.concatenate(predictions, axis=0)
    labels_np = np.concatenate(labels, axis=0)

    np.save(os.path.join(model_dir, f'test_predictions_{target_type}_{model_name}.npy'), predictions_np)
    np.save(os.path.join(model_dir, f'test_labels_{target_type}_{model_name}.npy'), labels_np)

    print(f"Test predictions saved to {os.path.join(model_dir, f'test_predictions_{target_type}_{model_name}.npy')}")
    print(f"Test labels saved to {os.path.join(model_dir, f'test_labels_{target_type}_{model_name}.npy')}")

    mae = mean_absolute_error(labels_np.reshape(-1), predictions_np.reshape(-1))
    mse = mean_squared_error(labels_np.reshape(-1), predictions_np.reshape(-1))

    print(f"Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}")

    print(predictions_np.shape, labels_np.shape)

if __name__ == "__main__":

    target_type = 'thermal'
    time_steps = 12
    nodenum = 620 + 303
    predictnum = 58
    main(target_type, time_steps)
