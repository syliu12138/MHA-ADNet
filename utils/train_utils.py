import torch
import torch.nn as nn
import math


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, y_pred, y_true):
        loss = 0
        n_dim = y_pred.dim()
        for i in range(len(self.quantiles)):
            q = self.quantiles[i]
            if n_dim == 3:
                error = y_pred[:, :, i].reshape(1, -1) - y_true.reshape(1, -1)
            elif n_dim == 4:
                error = y_pred[:, :, :, i].reshape(1, -1) - y_true.reshape(1, -1)
            # 如果预测值大于等于实际值，即 y_pred >= y_true， error> 0 则损失为 q * (y_pred - y_true)
            pinball_loss = torch.where(error > 0, (1 - q) * error, -q * error)
            loss += torch.mean(pinball_loss)
        return loss


def loss_func(y_pred, y_true, interval, kind="mse"):
    if kind == "mse":
        mse_per_point = ((y_pred - y_true) ** 2).mean(dim=0)
        total_loss = mse_per_point.mean()
    elif kind == "quantile":
        loss = QuantileLoss(interval)
        total_loss = loss(y_pred, y_true)
    return total_loss


def train_epoch(dataloader, model, optim, loss_kind, interval=None, device="cuda"):
    model.train()
    total_loss = 0.
    device = torch.device(device)
    idx = 0
    while len(dataloader) > 0:
        station_idx, X, y = dataloader.get_batch()
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X.unsqueeze(-1), station_idx)
        l = loss_func(y_pred, y, interval, loss_kind)
        total_loss += l.item()
        optim.zero_grad()
        l.backward()
        optim.step()
        idx += 1
    dataloader.reset()
    return total_loss / idx


def train_multi_site_transformer(dataloader, model, loss_kind, lr, max_iter, earlystopping, alpha, path, interval=None,
                         device="cuda", epsilon=0.):
    cnt = 0
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)
    min_loss = math.inf
    loss_list = []
    model.train()
    print("Start training.")
    for epoch in range(max_iter):
        l = train_epoch(dataloader, model, optim, loss_kind, interval, device)
        loss_list.append(l)
        if not epoch % 10:
            print(f"Epoch {epoch}: loss={l:.3f}")

        if min_loss - l > epsilon:
            cnt = 0
            min_loss = l
            torch.save(model, path)
            # torch.save(model.state_dict(), path)
        else:
            cnt += 1

        if cnt == earlystopping:
            break

    print("Finish training.")

