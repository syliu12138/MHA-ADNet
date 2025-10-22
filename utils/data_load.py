import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")


def data_preprocess(data_path, train_path, test_path, output_path, scale_path=None, step=1):
    df = pd.read_excel(data_path)
    # 计算特征矩阵
    feature_data = df.iloc[2:, :]
    feature_data.rename(columns={feature_data.columns[0]: "Date"}, inplace=True)
    for col in feature_data.columns[1:]:
        # 计算均值和标准差
        mean = feature_data[col].mean()
        std = feature_data[col].std()
        # 应用3σ准则，替换异常值为NaN
        feature_data[col] = feature_data[col].apply(lambda x: x if (mean - 3 * std <= x <= mean + 3 * std) else np.nan)
    feature_data = feature_data.interpolate(method='linear', limit_direction='both')
    print("开始正则化")
    mean_values = feature_data.iloc[:, 1:].mean()
    std_values = feature_data.iloc[:, 1:].std()
    feature_data.iloc[:, 1:] = (feature_data.iloc[:, 1:] - mean_values) / std_values
    scales = pd.DataFrame({"Station_ID": feature_data.columns[1:], "Mean": mean_values, "Std": std_values})
    train_df = feature_data.iloc[:-365, :]
    test_df = feature_data.iloc[-365 - step:, :]
    train_df.to_excel(train_path, index=False)
    test_df.to_excel(test_path, index=False)
    scales.to_excel(scale_path, index=False)
    feature_data.to_excel(output_path, index=False)


class StationDataset(Dataset):
    def __init__(self, data, batch_size, seq_len=15, pred_len=7, if_aggregate=False, step=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.if_aggregate = if_aggregate
        self.step = step
        self.batch_size = batch_size

        # Aggregate data to monthly if required
        if self.if_aggregate:
            data = self._aggregate_monthly(data)

        self.data = data.iloc[:, 1:].values  # Remove the date column, keep station data only
        self.num_stations = self.data.shape[1]
        self.samples_per_station = self.data.shape[0] - seq_len - pred_len + 1
        self.indices = {i: list(range(self.samples_per_station)) for i in range(self.num_stations)}

        # Initialize used and unused indices
        self.unused_indices = {i: [] for i in range(self.num_stations)}  # Tracks unused indices
        self.used_indices = {i: [] for i in range(self.num_stations)}  # Tracks used indices
        self.reset()

    def _aggregate_monthly(self, data):
        """
        Convert daily data into monthly data by sampling every 30 days.
        :param data: Original daily data DataFrame.
        :return: Monthly data DataFrame.
        """
        monthly_data = data.iloc[::self.step].reset_index(drop=True)
        return monthly_data

    def reset(self):
        """
        Reset unused sample indices while retaining previously used ones.
        """
        for station_idx in self.indices:
            # Randomly shuffle the indices for unused samples
            self.unused_indices[station_idx] = np.random.permutation(self.indices[station_idx]).tolist()
            self.used_indices[station_idx] = []

    def __len__(self):
        return sum(len(v) for v in self.unused_indices.values())

    def __getitem__(self, station_idx):
        if len(self.unused_indices[station_idx]) == 0:
            raise IndexError(f"No more samples available for station {station_idx}.")

        idx = self.unused_indices[station_idx].pop()
        self.used_indices[station_idx].append(idx)  # Mark this sample as used
        x = self.data[idx:idx + self.seq_len, station_idx]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, station_idx]
        return station_idx, x, y

    def get_batch(self):
        """
        Get a batch of data.
        :param batch_size: Batch size.
        :return: Station index, historical data (x), future data (y).
        """
        batch_size = self.batch_size
        while True:
            available_stations = [k for k, v in self.unused_indices.items() if len(v) > 0]
            if not available_stations:
                self.reset()  # Reset indices if all samples are used
                available_stations = [k for k, v in self.unused_indices.items() if len(v) > 0]

            station_idx = np.random.choice(available_stations)
            if len(self.unused_indices[station_idx]) < batch_size:
                batch_size = len(self.unused_indices[station_idx])

            batch = [self.__getitem__(station_idx) for _ in range(batch_size)]
            _, xs, ys = zip(*batch)
            return station_idx, torch.Tensor(np.array(xs)), torch.Tensor(np.array(ys))


def check_and_trim_shapes(data_list):
    shapes = [arr.shape for arr in data_list]
    if len(set(shapes)) > 1:
        # print("Shape mismatch detected:", shapes)
        data_list = data_list[:-1]
        # shapes = [arr.shape for arr in data_list]
        # print("Shape mismatch detected:", shapes)
    return data_list


def get_samples(df, look_back, look_forward, step=1, if_iter=False, n_iter=1):
    data = df.iloc[::step, 1:]
    samples_X = []
    samples_y = []
    if if_iter:
        pred_step = n_iter * look_forward
    else:
        pred_step = look_forward
    for i in range(data.shape[1]):
        for j in range(0, data.shape[0] - look_forward - look_back + 1, pred_step):
            samples_X.append(data.iloc[j:j + look_back, i].values)
            samples_y.append(data.iloc[j + look_back:j + look_back + pred_step, i].values)
            samples_X = check_and_trim_shapes(samples_X)
            samples_y = check_and_trim_shapes(samples_y)
    samples_X = torch.Tensor(samples_X)
    samples_y = torch.Tensor(samples_y)
    return samples_X.unsqueeze(-1), samples_y


def get_samples4qrf(df, look_back, look_forward, step=1):
    data = df.iloc[::step, 1:]
    samples_X = []
    samples_y = []
    for i in range(data.shape[1]):
        for j in range(0, data.shape[0] - look_forward - look_back + 1, look_forward):
            samples_X.append(data.iloc[j:j + look_back, i].values)
            samples_y.append(data.iloc[j + look_back:j + look_back + look_forward, i].values)
    samples_X = torch.Tensor(samples_X)
    samples_y = torch.Tensor(samples_y)
    return samples_X.unsqueeze(-1), samples_y
