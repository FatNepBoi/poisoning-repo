import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    
    controller_rows = df.iloc[:,1].values
    plant_rows = df.iloc[:,2].values

    N = len(plant_rows)

    X_rows = []
    y_rows = []
    # for each k starting at N (so previous outputs exist)
    for k in range(system_order, N):
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
    
    plant_rows = df.iloc[:,2].values
    controller_rows = df.iloc[:,1].values

    y = np.array(plant_rows, dtype=np.float32)
    u = np.array(controller_rows, dtype=np.float32)
    return y, u

"""
ADVERSARIAL ATTACK FUNCTIONS
"""

def get_gradients_from_segment_test(model, segment, order, scaler_X, scaler_y):
    grads = []
    y = torch.as_tensor(segment[:, 2], dtype=torch.float32, device=device)
    u = torch.as_tensor(segment[:, 1], dtype=torch.float32, device=device)
    T = len(y)

    mu_X = torch.tensor(scaler_X.mean_,  dtype=torch.float32, device=device)
    sd_X = torch.tensor(scaler_X.scale_, dtype=torch.float32, device=device)
    mu_y = float(scaler_y.mean_[0])
    sd_y = float(scaler_y.scale_[0])

    loss_function = nn.MSELoss(reduction="mean")
    model.eval()

    for i in range(order, T):
        prev_y = y[i-order:i].flip(0)
        current_u = u[i].unsqueeze(0)
        true = y[i].unsqueeze(0)                           # physical, (1,)

        x_k = torch.cat([current_u, prev_y]).unsqueeze(0)
        x_k = x_k.clone().detach().requires_grad_(True)    # physical input

        y_pred = mu_y + sd_y * model((x_k - mu_X) / sd_X)  # physical, (1,)
        loss = loss_function(y_pred, true)

        model.zero_grad()
        loss.backward()
        grads.append(x_k.grad.tolist())
    return grads

def aggregate_y_gradients(grads, order):
    if not grads:
        return np.array([])

    n_rows = len(grads)
    length = n_rows + order - 1              # covers y[0] ... y[T-2]
    sums = np.zeros(length)

    for i, row in enumerate(grads):
        y_grads = row[0][1:]
        for j, g in enumerate(y_grads):
            idx = i + order - 1 - j          # index fix
            sums[idx] += g

    return sums

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

def fit_scalers(train_files, order):
    X_train, y_train = build_arrays_from_file_list(train_files, order)
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    return scaler_X, scaler_y

"""
MODEL
"""
class CustomOrderNetwork(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1+order, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


"""
TRAINING LOOP
"""
def training_loop(model, train_loader, val_loader, learning_rate=1e-3, num_epochs=50, patience=7):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss(reduction="mean")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    train_metrics = {"MSE": [], "RMSE": [], "MAE": []}
    val_metrics = {"MSE": [], "RMSE": [], "MAE": []}

    best_val_mse = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        # -------------------- TRAIN --------------------
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # -------------------- EVAL --------------------
        model.eval()
        train_sse = 0.0
        train_sae = 0.0
        n_train_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                train_sse += torch.sum((y_pred - y_batch) ** 2).item()
                train_sae += torch.sum(torch.abs(y_pred - y_batch)).item()
                n_train_samples += X_batch.size(0)

        train_mse = train_sse / n_train_samples
        train_rmse = train_mse ** 0.5
        train_mae = train_sae / n_train_samples

        train_metrics["MSE"].append(train_mse)
        train_metrics["RMSE"].append(train_rmse)
        train_metrics["MAE"].append(train_mae)

        # -------------------- VALIDATION EVAL --------------------
        val_sse = 0.0
        val_sae = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                val_sse += torch.sum((y_pred - y_batch) ** 2).item()
                val_sae += torch.sum(torch.abs(y_pred - y_batch)).item()
                n_val_samples += X_batch.size(0)

        val_mse = val_sse / n_val_samples
        val_rmse = val_mse ** 0.5
        val_mae = val_sae / n_val_samples

        val_metrics["MSE"].append(val_mse)
        val_metrics["RMSE"].append(val_rmse)
        val_metrics["MAE"].append(val_mae)

        scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        # -------------------- LOGGING --------------------
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:3d} | "
            f"LR: {current_lr:.2e} | "
            f"Train MSE: {train_mse:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Best Val MSE: {best_val_mse:.6f}"
        )

        # -------------------- EARLY STOP --------------------
        if epochs_without_improvement >= patience:
            print(f"\\nEarly stopping triggered at epoch {epoch}")
            break;

    # -------------------- RESTORE BEST MODEL --------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"\nBest Validation MSE: {best_val_mse:.6f}")

    return (model, train_metrics, val_metrics, best_val_mse)

"""
FIND TRANSITIONS IN DATA
"""
def extract_transitions(file, window_size=100):
    data = pd.read_csv(file).values
    rising_segments = []
    falling_segments = []
    setpoint = data[:, 0]

    diff = np.diff(setpoint)
    rising_edges = np.where(np.isclose(diff, 1.0, atol=1e-6))[0] + 1
    falling_edges = np.where(np.isclose(diff, -1.0, atol=1e-6))[0] + 1

    if setpoint[0] == 1:
        rising_edges = np.insert(rising_edges, 0, 0)

    for i in rising_edges:
        if i + window_size <= len(data):
            segment = data[i:i+window_size]
            rising_segments.append({
                "start_idx": i,
                "segment": segment
            })

    for i in falling_edges:
        if i + window_size <= len(data):
            segment = data[i:i+window_size]
            falling_segments.append({
                "start_idx": i,
                "segment": segment
            })

    return rising_segments, falling_segments

"""
TESTING LOOP + AUXILIARY FUNCTION
"""
def testing_loop(model, y_true, u, order, scaler_X, scaler_y, device):
    model.eval()
    T = len(y_true)
    y_pred = np.zeros(T)
    y_pred[:order] = y_true[:order]

    with torch.no_grad():
        for k in range(order, T):
            past_outputs = y_pred[k-order:k][::-1]
            x = np.concatenate(([u[k]], past_outputs))

            max_abs_x = np.max(np.abs(x))
            if max_abs_x > 1e20:
                print(f"\nHuge input detected at timestep k={k}")
                print("max abs x =", max_abs_x)
                print("x =", x)
                break

            x_scaled = scaler_X.transform(x.reshape(1, -1))
            if not np.isfinite(x_scaled).all():
                print(f"\nNon-finite scaled input at timestep k={k}")
                print("x_scaled =", x_scaled)
                break

            x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
            pred_scaled = model(x_tensor).cpu().numpy()

            if not np.isfinite(pred_scaled).all():
                print(f"\nNon-finite scaled prediction at timestep k={k}")
                print("pred_scaled =", pred_scaled)
                break

            if np.abs(pred_scaled).max() > 5:
                print(
                    f"\nLarge scaled prediction at k={k}: "
                    f"{pred_scaled}"
                )

            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1))[0,0]

            if not np.isfinite(pred):
                print(f"\nNon-finite real prediction at timestep k={k}")
                print("pred =", pred)
                break
            
            if abs(pred) > 1e10:
                print(f"\nPrediction exploded at timestep k={k}")
                print("pred =", pred)
                break
            y_pred[k] = pred
    return y_pred

def evaluate_testing_loop(model, file_list, order, scaler_X, scaler_y, device):
    all_y_true = []
    all_y_pred = []
    all_errors = []
    transition_errors = []

    for file in file_list:
        y_true, u = get_columns_from_single_file(file)
        y_pred = testing_loop(model,y_true,u,order,scaler_X,scaler_y,device)

        error = y_true - y_pred
        all_errors.append(error)
        rising_segments, falling_segments = extract_transitions(file)

        for item in rising_segments:
            start = item["start_idx"]
            end = start + len(item["segment"])
            transition_errors.extend(error[start:end])

        for item in falling_segments:
            start = item["start_idx"]
            end = start + len(item["segment"])
            transition_errors.extend(error[start:end])

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    all_errors = np.concatenate(all_errors)
    transition_errors = np.array(transition_errors)

    mse = np.mean(all_errors**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_errors))

    transition_mse = np.mean(transition_errors**2)
    transition_rmse = np.sqrt(transition_mse)
    transition_mae = np.mean(np.abs(transition_errors))

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "transition_mse": transition_mse,
        "transition_rmse": transition_rmse,
        "transition_mae": transition_mae
    }

    return metrics, y_pred_all, y_true_all