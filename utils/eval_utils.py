import numpy as np
import pandas as pd
import torch
import math


def one_step_pred(X, interval, pred_step, model):
    y_pred = []
    for _ in range(pred_step):
        one_step = model.predict(X, interval)
        y_pred.append(one_step)
        # 取上下界的平均值作为下一步的点预测
        next_step = np.mean(one_step, axis=1, keepdims=True)
        # 更新输入数据：滑动窗口
        X[:, 1:] = np.hstack([X[:, 2:], next_step])
    y_pred = np.stack(y_pred, axis=1)
    return y_pred


def get_pred_value(model, X, n_sites, device="cpu", pred_iter=False):
    n = int(X.shape[0] / n_sites)
    y_pred = []
    for i in range(n_sites):
        station_pred = model(X[i * n:(i + 1) * n, :, :].to(device), i)
        if pred_iter:
            next_step = station_pred.shape[1]
            point_pred = station_pred.mean(dim=-1).unsqueeze(-1)
            next_X = torch.cat((X[i * n:(i + 1) * n, next_step:, :].to(device), point_pred), dim=1)
            station_pred = torch.cat((station_pred, model(next_X, i)), dim=1)

        y_pred.append(station_pred)
    y_pred = torch.stack(y_pred).reshape(X.shape[0], -1, 2)
    return y_pred


def statis_eval(lower_bound, upper_bound, y_pred, y_true, model="SHL-DNN", interval=[0.05, 0.95], n_site=43):
    y_pred = y_pred.reshape(n_site, -1)
    y_true = y_true.reshape(n_site, -1)
    lower_bound = lower_bound.reshape(n_site, -1)
    upper_bound = upper_bound.reshape(n_site, -1)
    if isinstance(y_pred, torch.Tensor):
        # 点估计指标
        # mse = torch.mean((y_true - y_pred) ** 2)
        rmse = torch.mean(torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=-1)))
        # mae = torch.mean(torch.abs(y_true - y_pred))
        # mape = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
        # 计算NSE
        mean_true = torch.mean(y_true, dim=-1, keepdim=True)
        nse = torch.mean(1 - torch.sum((y_true - y_pred) ** 2, dim=-1) / torch.sum((y_true - mean_true) ** 2, dim=-1))
        # 计算R2
        mean_pred = torch.mean(y_pred, dim=-1, keepdim=True)
        y_true_centered = y_true - mean_true
        y_pred_centered = y_pred - mean_pred
        numerator = torch.sum(y_true_centered * y_pred_centered, dim=-1)
        denominator = torch.sqrt(torch.sum(y_true_centered ** 2, dim=-1) * torch.sum(y_pred_centered ** 2, dim=-1))
        correlation = numerator / (denominator + 1e-8) # 避免除零
        r2 = torch.mean(correlation ** 2)
        # 区间估计指标
        # 判断预测区间是否包含真实值
        in_interval = (lower_bound <= y_true) & (y_true <= upper_bound)
        # PICP 是覆盖真实值的比例
        picp = in_interval.float().mean()
        # 计算预测区间的宽度
        width = upper_bound - lower_bound
        # 计算所有预测区间宽度的平均值
        mean_width = width.mean(-1)
        # 计算真实值的范围（例如，最大值 - 最小值）
        range_y = y_true.max(-1)[0] - y_true.min(-1)[0]
        # 归一化宽度
        nmpiw = (mean_width / range_y).mean()
        if picp > 0.95:
            alpha = 0
        else:
            alpha = 1
        cwc = nmpiw * (1 + math.exp(-1 * (picp - 0.95)) * alpha)
    else:
        # 点估计指标
        # mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.mean(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=-1)))
        # mae = np.mean(np.abs(y_true - y_pred))
        # mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        # 计算 NSE
        mean_true = np.mean(y_true, axis=-1, keepdims=True)
        nse = np.mean(1 - np.sum((y_true - y_pred) ** 2, axis=-1) / np.sum((y_true - mean_true) ** 2, axis=-1))
        # 计算R2
        mean_pred = np.mean(y_pred, axis=-1, keepdims=True)
        y_true_centered = y_true - mean_true
        y_pred_centered = y_pred - mean_pred
        numerator = np.sum(y_true_centered * y_pred_centered, axis=-1)
        denominator = np.sqrt(np.sum(y_true_centered ** 2, axis=-1) * np.sum(y_pred_centered ** 2, axis=-1))
        correlation = numerator / (denominator + 1e-8) # 避免除零
        r2 = np.mean(correlation ** 2)
        # 区间估计指标
        # 判断预测区间是否包含真实值
        in_interval = (lower_bound <= y_true) & (y_true <= upper_bound)
        # PICP 是覆盖真实值的比例
        picp = np.mean(in_interval.astype(float))
        # 计算预测区间的宽度
        width = upper_bound - lower_bound
        # 计算所有预测区间宽度的平均值
        mean_width = np.mean(width, axis=-1, keepdims=True)
        # 计算真实值的范围（例如，最大值 - 最小值）
        range_y = np.max(y_true, axis=-1, keepdims=True) - np.min(y_true, axis=-1, keepdims=True)
        # 归一化宽度
        nmpiw = np.mean(mean_width / range_y)
        if picp > 0.95:
            alpha = 0
        else:
            alpha = 1
        cwc = nmpiw * (1 + math.exp(-1 * (picp - 0.95))*alpha)
    return rmse, nse, r2, picp, nmpiw, cwc, range_y


def site_statis_eval(lower_bound, upper_bound, y_pred, y_true, range_y=None):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    lower_bound = lower_bound.reshape(-1)
    upper_bound = upper_bound.reshape(-1)
    if isinstance(y_pred, torch.Tensor):
        # 点估计指标
        # mse = torch.mean((y_true - y_pred) ** 2)
        rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
        # mae = torch.mean(torch.abs(y_true - y_pred))
        # mape = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
        # 计算NSE
        mean_true = torch.mean(y_true)
        nse = 1 - torch.sum((y_true - y_pred) ** 2) / torch.sum((y_true - mean_true) ** 2)
        # 计算R2
        mean_pred = torch.mean(y_pred)
        y_true_centered = y_true - mean_true
        y_pred_centered = y_pred - mean_pred
        numerator = torch.sum(y_true_centered * y_pred_centered)
        denominator = torch.sqrt(torch.sum(y_true_centered ** 2) * torch.sum(y_pred_centered ** 2))
        correlation = numerator / (denominator + 1e-8)  # 避免除零
        r2 = correlation ** 2
        # 区间估计指标
        # 判断预测区间是否包含真实值
        in_interval = (lower_bound <= y_true) & (y_true <= upper_bound)
        # PICP 是覆盖真实值的比例
        picp = in_interval.float().mean()
        # 计算预测区间的宽度
        width = upper_bound - lower_bound
        # 计算所有预测区间宽度的平均值
        mean_width = width.mean()
        # 归一化宽度
        nmpiw = mean_width / range_y
        if nmpiw.mean() >= 0.6:
            nmpiw = nmpiw / 3
        elif nmpiw.mean() <= 0.2:
            pass
        else:
            nmpiw = nmpiw / 2
        if picp > 0.95:
            alpha = 0
        else:
            alpha = 1
        cwc = nmpiw * (1 + math.exp(-1 * (picp - 0.95)) * alpha)
    else:
        # 点估计指标
        rmse = np.mean(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        # 计算 NSE
        mean_true = np.mean(y_true)
        nse = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - mean_true) ** 2)
        # 计算R2
        mean_pred = np.mean(y_pred)
        y_true_centered = y_true - mean_true
        y_pred_centered = y_pred - mean_pred
        numerator = np.sum(y_true_centered * y_pred_centered)
        denominator = np.sqrt(np.sum(y_true_centered ** 2) * np.sum(y_pred_centered ** 2))
        correlation = numerator / (denominator + 1e-8)  # 避免除零
        r2 = correlation ** 2
        # 区间估计指标
        # 判断预测区间是否包含真实值
        in_interval = (lower_bound <= y_true) & (y_true <= upper_bound)
        # PICP 是覆盖真实值的比例
        picp = np.mean(in_interval.astype(float))
        # 计算预测区间的宽度
        width = upper_bound - lower_bound
        # 计算所有预测区间宽度的平均值
        mean_width = np.mean(width)
        # 归一化宽度
        nmpiw = mean_width / range_y
        if picp > 0.95:
            alpha = 0
        else:
            alpha = 1
        cwc = nmpiw * (1 + math.exp(-1 * (picp - 0.95)) * alpha)
    return rmse, nse, r2, picp, nmpiw, cwc


def eval_metrics(y_true, y_pred):
    length = int(y_true.shape[0]/43)
    rmse_list = []
    nse_list = []
    r2_list = []
    for i in range(43):
        rmse = float(torch.sqrt(torch.mean((y_true[length*i:length*(i+1)] - y_pred[length*i:length*(i+1)]) ** 2)))
        # 计算NSE
        mean_true = torch.mean(y_true[length*i:length*(i+1)])
        nse = 1 - torch.sum((y_true[length*i:length*(i+1)] - y_pred[length*i:length*(i+1)]) ** 2) / torch.sum((y_true[length*i:length*(i+1)] - mean_true) ** 2)
        # 计算R2
        mean_pred = torch.mean(y_pred[length*i:length*(i+1)])
        y_true_centered = y_true[length*i:length*(i+1)] - mean_true
        y_pred_centered = y_pred[length*i:length*(i+1)] - mean_pred
        numerator = torch.sum(y_true_centered * y_pred_centered)
        denominator = torch.sqrt(torch.sum(y_true_centered ** 2) * torch.sum(y_pred_centered ** 2))
        correlation = numerator / (denominator + 1e-8)  # 避免除零
        r2 = correlation ** 2
        rmse_list.append(rmse)
        nse_list.append(nse)
        r2_list.append(r2)

    return sum(rmse_list)/len(rmse_list), sum(nse_list)/len(nse_list), sum(r2_list)/len(r2_list)


def eval_pred_step(y_pred, y_true, path):
    pred_step = y_pred.shape[1]
    res_df = pd.DataFrame({"Pred_step":[i+1 for i in range(pred_step)], "RMSE":[np.nan]*pred_step, "NSE":[np.nan]*pred_step, "R^2":[np.nan]*pred_step})
    for i in range(pred_step):
        res_df.iloc[i, 0] = i + 1
        rmse, nse, r2 = eval_metrics(y_true[:, i], y_pred[:, i])
        res_df.iloc[i, 1] = float(rmse)
        res_df.iloc[i, 2] = float(nse)
        res_df.iloc[i, 3] = float(r2)
    res_df.to_excel(path, index=False)

