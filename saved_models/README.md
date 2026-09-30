## Cada modelo desta pasta é inicializado com PyTorch da seguinte maneira

### Atenção: Existem 3 variáveis cuja criação não esta definida neste documento:
###          train_files, val_files, test_files

```python

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


# train_files, val_files e test_files são as variaveis criadas apos carregar os ficheiros
config_train = train_files
config_val = val_files
config_test = test_files[:3]

configs = {}
batch_size = 32

configs["normal_model"] = create_tensors(
    config_train,
    config_val,
    config_test,
    2,
    batch_size
)

for epsilon in epsilons:

    configs[f"{epsilon}_calculated_model"] = create_tensors(
        calculated_train_files[epsilon],
        calculated_val_files[epsilon],
        config_test,
        2,
        batch_size
    )

    configs[f"{epsilon}_gaussian_model"] = create_tensors(
        gaussian_train_files[epsilon],
        gaussian_val_files[epsilon],
        config_test,
        2,
        batch_size
    )

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

            # Detect divergence
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

        y_pred = testing_loop(
            model,
            y_true,
            u,
            order,
            scaler_X,
            scaler_y,
            device
        )

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

topk_folder = "top25"
save_dir = os.path.join("saved_models", topk_folder)
experiments = {}

for filename in sorted(os.listdir(save_dir)):
        if not filename.endswith(".pth"):
            continue
        name = filename[:-4]
        parts = name.split("_")

        run = int(parts[1])
        order = int(parts[3])
        config = "_".join(parts[4:])

        path = os.path.join(save_dir, filename)
        model = CustomOrderNetwork(order)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        experiments[(run, order, config)] = {"model": model}

testing_files = test_files[-1:] # Testing files que serão usados

for key, data in experiments.items():

    run, order, config = key

    model = data["model"]

    test_loader = configs[config][2]
    scaler_X = configs[config][5]
    scaler_y = configs[config][6]

    metrics, all_preds, all_targets = evaluate_testing_loop(
        model,
        testing_files,
        order,
        scaler_X,
        scaler_y,
        device
    )

    experiments[key]["test_metrics"] = metrics
    experiments[key]["predictions"] = all_preds
    experiments[key]["targets"] = all_targets   # valores reais
```