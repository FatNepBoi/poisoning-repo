import numpy as np
import pandas as pd
import glob
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
GET STUFF FROM FILES
"""

def build_arrays_from_single_file(file, system_order):
    try:
        df = pd.read_csv(file)
    except ValueError as e:
        print("Error: ", e)
    
    setpoint_rows = df.iloc[:,0].values
    controller_rows = df.iloc[:,1].values
    plant_rows = df.iloc[:,2].values

    N = len(plant_rows)

    X_rows = []
    y_rows = []
    # for each k starting at N (so previous outputs exist)
    for k in range(system_order, N):
        #x = [setpoint_rows[k], controller_rows[k]]
        x = [controller_rows[k]]
        for i in range(1,system_order+1):
            x.append(plant_rows[k-i])
        y = plant_rows[k]
        X_rows.append(x)
        y_rows.append(y)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    return X, y

def build_arrays_from_file_list(file_list, system_order):
    X_list = []
    y_list = []
    for f in file_list:
        X_file, y_file = build_arrays_from_single_file(f, system_order)
        X_list.append(X_file)
        y_list.append(y_file)

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    return X_all, y_all

def get_columns_from_single_file(file):
    try:
        df = pd.read_csv(file)
    except ValueError as e:
        print("Error: ", e)
    
    #setpoint_rows = df.iloc[:,0].values
    plant_rows = df.iloc[:,2].values
    controller_rows = df.iloc[:,1].values

    y = np.array(plant_rows, dtype=np.float32)
    u = np.array(controller_rows, dtype=np.float32)
    return y, u

def get_columns_from_file_list(file_list):
    y_list = []
    u_list = []
    for f in file_list:
        y_file, u_file = get_columns_from_single_file(f)
        y_list.append(y_file)
        u_list.append(u_file)
    y_all = np.concatenate(y_list)
    u_all = np.concatenate(u_list)
    return y_all, u_all

"""
TESTING ADVERSARIAL ATTACK FUNCTIONS
"""

def get_gradients_from_files(model, binary_files, order):
    grads = []
    y, u = get_columns_from_file_list(binary_files)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    u = torch.as_tensor(u, dtype=torch.float32, device=device)
    T = len(y)
    model.eval()

    for i in range(order, T):
        prev_y = y[i-order:i].flip(0)
        current_u = u[i].unsqueeze(0)
        true = y[i].unsqueeze(0)

        x_k = torch.cat([current_u, prev_y]).unsqueeze(0)
        x_k = x_k.clone().detach().requires_grad_(True)

        loss_function = nn.MSELoss(reduction="mean")

        y_pred = model(x_k)

        loss = loss_function(y_pred, true)
        loss = -loss

        model.zero_grad()
        loss.backward()

        grad = x_k.grad
        grads.append(grad.tolist())
    return grads

def get_gradients_from_segment(model, segment, order):
    grads = []
    y = torch.as_tensor(segment[:, 2], dtype=torch.float32, device=device)
    u = torch.as_tensor(segment[:, 1], dtype=torch.float32, device=device)
    T = len(y)
    model.eval()

    for i in range(order, T):
        prev_y = y[i-order:i].flip(0)
        current_u = u[i].unsqueeze(0)
        true = y[i].unsqueeze(0)

        x_k = torch.cat([current_u, prev_y]).unsqueeze(0)
        x_k = x_k.clone().detach().requires_grad_(True)

        loss_function = nn.MSELoss(reduction="mean")

        y_pred = model(x_k)

        loss = loss_function(y_pred, true)
        loss = -loss

        model.zero_grad()
        loss.backward()

        grad = x_k.grad
        grads.append(grad.tolist())
    return grads

def get_gradients_from_segment_test(model, segment, order):
    grads = []
    y = torch.as_tensor(segment[:, 2], dtype=torch.float32, device=device)
    u = torch.as_tensor(segment[:, 1], dtype=torch.float32, device=device)
    T = len(y)
    model.eval()

    for i in range(order, T):
        prev_y = y[i-order:i].flip(0)
        current_u = u[i].unsqueeze(0)
        true = y[i].unsqueeze(0)

        x_k = torch.cat([current_u, prev_y]).unsqueeze(0)
        x_k = x_k.clone().detach().requires_grad_(True)

        loss_function = nn.MSELoss(reduction="mean")

        y_pred = model(x_k)

        loss = loss_function(y_pred, true)

        model.zero_grad()
        loss.backward()

        grad = x_k.grad
        grads.append(grad.tolist())
    return grads

def aggregate_y_gradients(grads, mode=0):
    max_grad = defaultdict(float)
    sums = defaultdict(float)
    counts = defaultdict(int)      

    for i, row in enumerate(grads):
        row = row[0]
        y_grads = row[1:]

        for j, g in enumerate(y_grads):
            y_index = i + j
            sums[y_index] += g
            counts[y_index] += 1
            
            if abs(g) > abs(max_grad[y_index]):
                max_grad[y_index] = g

    final_grad_dict = {}
    all_indices = sums.keys()

    if mode == 0:  # Max Absolute
        final_grad_dict = max_grad

    elif mode == 1:  # Mean
        for idx in all_indices:
            final_grad_dict[idx] = sums[idx] / counts[idx]

    elif mode == 2:  # Summation
        final_grad_dict = sums

    else:
        raise ValueError("mode must be 0 (max), 1 (mean), 2 (sum)")

    # Construct the final dense numpy array
    if not final_grad_dict:
        return np.array([])
        
    max_idx = max(final_grad_dict.keys())
    flat = np.zeros(max_idx + 1)

    for i in range(max_idx + 1):
        flat[i] = final_grad_dict.get(i, 0.0)

    return flat


def aggregate_u_gradients(grads):

    max_grad = defaultdict(float)

    for i, row in enumerate(grads):
        row = row[0]
        u_grads = row[:1]

        for j, g in enumerate(u_grads):
            u_index = i + j
            if abs(g) > abs(max_grad[u_index]):
                max_grad[u_index] = g

    max_idx = max(max_grad.keys())
    flat = np.zeros(max_idx + 1)

    for i in range(max_idx + 1):
        flat[i] = max_grad[i]

    return flat

def topk_mask(flat_grad, block_size=3001, top_percent=0.1):
    n = len(flat_grad)
    masked = np.zeros_like(flat_grad)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = flat_grad[start:end]

        k = max(1, int(len(block) * top_percent))
        top_indices = np.argsort(block)[-k:]
        masked[start:end][top_indices] = block[top_indices]

    return masked

"""
MAKE TENSORS OUT OF FILE COLLECTIONS
"""
class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_tensors(train_files, val_files, test_files, order, batch_size):

    X_train, y_train = build_arrays_from_file_list(train_files,order)
    X_val,   y_val = build_arrays_from_file_list(val_files,order)
    X_test,  y_test = build_arrays_from_file_list(test_files,order)

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1,1))

    X_train_s = scaler_X.transform(X_train)
    y_train_s = scaler_y.transform(y_train.reshape(-1,1)).reshape(-1)

    X_val_s   = scaler_X.transform(X_val)
    y_val_s   = scaler_y.transform(y_val.reshape(-1,1)).reshape(-1)

    X_test_s  = scaler_X.transform(X_test)
    y_test_s  = scaler_y.transform(y_test.reshape(-1,1)).reshape(-1)

    train_ds = ArrayDataset(X_train_s, y_train_s)
    val_ds   = ArrayDataset(X_val_s, y_val_s)
    test_ds  = ArrayDataset(X_test_s, y_test_s)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return [train_loader, val_loader, test_loader, order, test_files, scaler_X, scaler_y]

def create_loader(files, order, scaler_x, scaler_y):
    X_new, y_new = build_arrays_from_file_list(files,order)

    X_new_s = scaler_x.transform(X_new)
    y_new_s = scaler_y.transform(y_new.reshape(-1,1)).reshape(-1)

    batch_size = 64
    ds  = ArrayDataset(X_new_s, y_new_s)

    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def get_transition_mask(df, window_size, tol=1e-6):
    setpoint = df.iloc[:, 0].values
    labels = np.zeros(len(df), dtype=np.int8)
    positions = np.full(len(df), -1, dtype=np.int32)  # -1 = not in any transition window

    diff = np.diff(setpoint)
    rising_edges = np.where(np.isclose(diff, 1.0, atol=tol))[0] + 1
    falling_edges = np.where(np.isclose(diff, -1.0, atol=tol))[0] + 1

    if np.isclose(setpoint[0], 1.0, atol=tol):
        rising_edges = np.concatenate(([0], rising_edges))

    for idx in rising_edges:
        end_idx = min(idx + window_size, len(df))
        length = end_idx - idx
        labels[idx:end_idx] = 1
        positions[idx:end_idx] = np.arange(length)  # 0, 1, 2, ..., length-1

    for idx in falling_edges:
        end_idx = min(idx + window_size, len(df))
        length = end_idx - idx
        labels[idx:end_idx] = -1
        positions[idx:end_idx] = np.arange(length)

    return labels, positions

def build_arrays_from_single_file_protocol2(file, system_order, window_size):
    df = pd.read_csv(file)
    controller_rows = df.iloc[:,1].values
    plant_rows = df.iloc[:,2].values

    transition_mask, position_mask = get_transition_mask(df, window_size)

    X_rows = []
    y_rows = []
    transition_rows = []
    position_rows = []

    for k in range(system_order, len(df)):
        x = [controller_rows[k]]
        for i in range(1, system_order + 1):
            x.append(plant_rows[k-i])

        X_rows.append(x)
        y_rows.append(plant_rows[k])
        transition_rows.append(transition_mask[k])
        position_rows.append(position_mask[k])

    return (np.array(X_rows, dtype=np.float32),np.array(y_rows, dtype=np.float32),np.array(transition_rows, dtype=np.int32),np.array(position_rows, dtype=np.int32))

def build_arrays_from_file_list_protocol2(file_list, system_order, window_size):
    X_list = []
    y_list = []
    transition_list = []
    position_list = []

    for f in file_list:
        X_file, y_file, trans_file, pos_file = build_arrays_from_single_file_protocol2(f, system_order, window_size)

        X_list.append(X_file)
        y_list.append(y_file)
        transition_list.append(trans_file)
        position_list.append(pos_file)

    return (np.vstack(X_list),np.concatenate(y_list),np.concatenate(transition_list),np.concatenate(position_list))

def create_tensors_protocol2(train_files, val_files, test_files, order, batch_size, window_size):
    (X_train,y_train,trans_train,pos_train) = build_arrays_from_file_list_protocol2(train_files, order, window_size)
    (X_val,y_val,trans_val,pos_val) = build_arrays_from_file_list_protocol2(val_files, order, window_size)

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1,1))

    X_train_s = scaler_X.transform(X_train)
    y_train_s = scaler_y.transform(y_train.reshape(-1,1)).reshape(-1)

    X_val_s = scaler_X.transform(X_val)
    y_val_s = scaler_y.transform(y_val.reshape(-1,1)).reshape(-1)

    train_ds = GradientDataset(X_train_s,y_train_s,trans_train,pos_train)
    val_ds = GradientDataset(X_val_s,y_val_s,trans_val,pos_val)

    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_ds,batch_size=batch_size,shuffle=False)

    return (train_loader,val_loader)

class GradientDataset(Dataset):

    def __init__(self,X,y,transitions,positions):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.transitions = torch.tensor(transitions,dtype=torch.long)
        self.positions = torch.tensor(positions,dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx],self.y[idx],self.transitions[idx],self.positions[idx])
    
def aggregate_y_gradients_protocol2(grads, mode="mean"):
    accumulator = defaultdict(list)

    for i, row in enumerate(grads):
        y_grads = row[1:]
        for j, g in enumerate(y_grads):
            y_index = i + j
            accumulator[y_index].append(g)

    max_idx = max(accumulator.keys())
    flat = np.zeros(max_idx + 1)

    for idx, values in accumulator.items():
        if mode == "max":
            flat[idx] = max(values,key=lambda x: abs(x))
        elif mode == "sum":
            flat[idx] = np.sum(values)
        else:  # mean
            flat[idx] = np.mean(values)
    return flat

"""
MODEL
"""
class CustomOrderNetwork(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1+order, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)