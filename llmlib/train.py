import torch
import numpy as np
import time
from torch import nn


def train_loop(model, dl_train, dl_val, dl_test, plotter: PlotterION, epochs, optimizer, criterion, use_lr_scheduler=False, use_smoothing=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("-------------cuda run-----------------")
    else:
        device = torch.device("cpu")
        print("--------------cpu run-----------------")

    model.to(device)
    model.train()
    train_loss_list = []
    train_std_list = []
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA_LRS)
    if use_smoothing:
        ema_mae = torch.ones(model.n_features, device=device)
    start_time = time.time()
    for i in range(epochs):
        train_loss = 0
        val_loss = 0
        test_loss = 0
        batch_train_losses = []
        batch_val_losses = []
        batch_test_losses = []
        batch_i = 0
        teachf_p = 1.0 * np.exp(-i / (epochs * TEACHER_DROP_OFF))
        # teachf_p = 0

        model.train()
        for x, y in dl_train:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x, y, teachf_p)
            if np.isnan(y_pred.detach().cpu()).any() or np.isinf(y_pred.detach().cpu()).any():
                print(f"y_pred contains NaN or inf! (train)")
            # - weighted L1Loss ----------------------------------------------------
            if USE_PFL:
                abs_error = torch.abs(y_pred - y) ** 2  # Absolute Error, add '**2' to use mse-loss
                per_feature_mae = abs_error.mean(dim=(0, 1)).detach() + 1e-6  # AE to MAE, and add a little to prevent 1/0 (1e-6 is limit of decimal-precision for float32). this is per feature loss.
                # use expon. moving average to give more weight to recent losses, but include older information too.
                if use_smoothing:
                    alpha = 0.2
                    ema_mae = (1 - alpha) * ema_mae + alpha * per_feature_mae
                    feature_weights = ema_mae / ema_mae.sum()
                else:
                    feature_weights = per_feature_mae / per_feature_mae.sum()  # normalize feature weights to sum to 1
                # Clamp minimum to counteract numerical instability
                feature_weights = torch.clamp(feature_weights, min=0.001)
                # Renormalize to ensure weights still sum to 1
                feature_weights = feature_weights / feature_weights.sum()
                weighted_error = abs_error * feature_weights  # weigh the loss with the feature weights
                loss = weighted_error.mean()
            else:
                loss = criterion(y_pred, y)
            # ----------------------------------------------------------------------
            try:
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward()
            except RuntimeError as e:
                print(e)
                print("input:")
                print(x.cpu())
                print("target:")
                print(y.cpu())
                print("predicted:")
                print(y_pred.cpu())
                print("feature_weights:")
                print(feature_weights)
                print("per_feature_mae:")
                print(per_feature_mae)
                print("absolute error:")
                print(abs_error)
                con_val = input("Continue? y/N")
                if con_val == "y":
                    continue
                else:
                    stop = True
                    break
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            repr_train_loss = criterion(y_pred, y).item()  # Make sure to keep general L1-Loss for performance metric plotting
            train_loss += repr_train_loss
            batch_train_losses.append(repr_train_loss)
            batch_i += 1
            if (batch_i % 10) == 0:
                if use_lr_scheduler:
                    print(
                        f"Epoch {i + 1}, Train-Batch {batch_i}/{len(dl_train)}, Learning Rate: {optimizer.param_groups[0]['lr']}, Time: {(time.time() - start_time) / 60}")
                else:
                    print(
                        f"Epoch {i + 1}, Train-Batch {batch_i}/{len(dl_train)}, Time: {(time.time() - start_time) / 60}")
        print("Last Prediction (train):")
        print(y_pred)
        print("Last Label: (train)")
        print(y)
        if USE_PFL:
            print("Last Batch Per-Feature-Loss: (train)")
            print(str(per_feature_mae.cpu()))

        train_loss = train_loss / len(dl_train)
        train_loss_list.append(train_loss)
        train_std_list.append(torch.std(torch.tensor(batch_train_losses)).item())
    return train_loss_list, train_std_list

def val_loop(model):
    model.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("-------------cuda val-----------------")
    else:
        device = torch.device("cpu")
        print("--------------cpu val-----------------")
    for x, y in dl_val:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred = model(x)
        if np.isnan(y_pred.detach().cpu()).any() or np.isinf(y_pred.detach().cpu()).any():
            print(f"y_pred contains NaN or inf! (val)")
        loss = criterion(y_pred, y)
        batch_val_losses.append(loss.item())
        val_loss += loss.item()
        batch_i += 1
        if (batch_i % 10) == 0:
            print(f"Epoch {i + 1}, Valid-Batch {batch_i}/{len(dl_val)}, Time: {(time.time() - start_time) / 60}")
    print(f"val_loss_sum: {val_loss}, len ds: {len(dl_val)}")
    val_loss = val_loss / len(dl_val)
    val_loss_list.append(val_loss)
    val_std_list.append(torch.std(torch.tensor(batch_val_losses)).item())
    print("Last Prediction (val):")
    print(y_pred)
    print("Last Label (val):")
    print(y)
    plotter.plot_ion_step_3x2('val', i, teachf_p, x, y, y_pred)
    for x, y in dl_test:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred = model(x)
        loss = criterion(y_pred, y)
        batch_test_losses.append(loss.item())
        test_loss += loss.item()
    test_loss = test_loss / len(dl_test)
    test_loss_ls.append(test_loss)
    test_std_list.append(torch.std(torch.tensor(batch_test_losses)).item())
    plotter.plot_ion_step_3x2('test', i, teachf_p, x, y, y_pred)
    print(
        f"Epoch {i + 1} Done. Train Loss: {train_loss} | Val Loss: {val_loss} | Test Loss: {test_loss} | LR: {optimizer.param_groups[0]['lr']} | Time:{(time.time() - start_time) / 60}")
    cp_saver.checkpoint(model, i + 1, train_loss, val_loss, test_loss)
    if use_lr_scheduler:
        scheduler.step()
    return cp_saver.kpi_list, cp_saver.cp_list, train_loss_list, val_loss_list, test_loss_ls, train_std_list, val_std_list, test_std_list, model


def test_loop(model, criterion, test_data_loader):
    model.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("-------------cuda test----------------")
    else:
        device = torch.device("cpu")
        print("--------------cpu test----------------")
    test_loss = []
    for x, y in test_data_loader:
        if np.isnan(x).any():
            print(f"data contains NaN!(test)")
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        test_loss.append(loss.item())
    print(f"Train Loss Avg. Test Run2: {sum(test_loss) / len(test_data_loader)}")
    return test_loss