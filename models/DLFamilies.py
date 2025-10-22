import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # Add positional encoding to each time step
        return x


class FullConnectLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, neg_slope):
        super(FullConnectLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.neg_slope = neg_slope
        self.output = nn.Linear(hidden_dim, output_dim)
        # self.output = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.dropout1(self.fc1(x)), self.neg_slope)
        x = self.output(x)
        return x


class AttentionSeparateLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_head, dropout):
        super(AttentionSeparateLayer, self).__init__()
        self.encode = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True  # Ensures input shape is (batch_size, time_stamp, embed_dim)
        )
        self.output = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.encode(x)
        x = self.output(x)
        return x


class MultiSiteTransformer(nn.Module):
    def __init__(self, n_sites, look_forward, input_dim, embed_dim, hidden_dim1, hidden_dim2, output_dim, n_heads, n_layers,
                 dropout, neg_slope):
        super(MultiSiteTransformer, self).__init__()
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.token_embedding = nn.Linear(input_dim, embed_dim)
        self.embed_dropout = nn.Dropout(p=dropout)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim1,
            dropout=dropout,
            batch_first=True  # Ensures input shape is (batch_size, time_stamp, embed_dim)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.look_forward = look_forward
        self.separate_layers = nn.ModuleList([FullConnectLayer(input_dim=embed_dim,
                                                               hidden_dim=hidden_dim2,
                                                               output_dim=output_dim,
                                                               dropout=dropout,
                                                               neg_slope=neg_slope) for _ in range(n_sites)])

        # self.separate_layers = nn.ModuleList([AttentionSeparateLayer(input_dim=embed_dim,
        #                                                              hidden_dim=hidden_dim1,
        #                                                              output_dim=output_dim,
        #                                                              n_head=1,
        #                                                              dropout=dropout) for _ in range(n_sites)])

    def forward(self, x, station_idx):
        x = self.positional_encoding(x)
        # x = self.token_embedding(x)
        x = self.embed_dropout(x)
        # Pass through Transformer encoder
        x = self.encoder(x)
        x = x[:, -1*self.look_forward:, :]
        output = self.separate_layers[station_idx](x)
        return output


# -----------------------------------------------SHL-DNN-------------------------------------------------------------
class SHLDNNOutputLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, neg_slope, if_quantile=True):
        super(SHLDNNOutputLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.neg_slope = neg_slope
        self.output1 = nn.Linear(hidden_dim, output_dim)
        self.output2 = nn.Linear(hidden_dim, output_dim)
        self.if_quantile = if_quantile

    def forward(self, x):
        x = F.leaky_relu(self.dropout1(self.fc1(x)), self.neg_slope)
        x1 = self.output1(x).permute(0, 2, 1)
        if self.if_quantile:
            x2 = self.output2(x).permute(0, 2, 1)
            return torch.cat([x1, x2], dim=-1)
        else:
            return x1


class SHLDNN(nn.Module):
    def __init__(self, n_sites, look_forward, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4,
                 dropout, neg_slope, if_quantile=True):
        super(SHLDNN, self).__init__()
        self.input = nn.Linear(input_dim, hidden_dim1)
        self.fc1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, hidden_dim3)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.look_forward = look_forward
        self.neg_slope = neg_slope
        self.separate_layers = nn.ModuleList([SHLDNNOutputLayer(input_dim=hidden_dim3,
                                                                hidden_dim=hidden_dim4,
                                                                output_dim=look_forward,
                                                                dropout=dropout,
                                                                neg_slope=neg_slope,
                                                                if_quantile=if_quantile) for _ in range(n_sites)])

    def forward(self, x, station_idx):
        # 为了和MST共用相同格式的输入
        x = x.permute(0, 2, 1)
        x = F.leaky_relu(self.input(x), self.neg_slope)
        # x = self.dropout1(x)
        x = F.leaky_relu(self.fc1(x), self.neg_slope)
        # x = self.dropout2(x)
        x = F.leaky_relu(self.fc2(x), self.neg_slope)
        # x = self.dropout3(x)
        output = self.separate_layers[station_idx](x)
        return output


# --------------------------------------------------QLSTM------------------------------------------------------------
class QLSTM(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, look_forward, n_layers=1, device="cuda"):
        super(QLSTM, self).__init__()
        self.embed = nn.Linear(1, embed_dim)
        self.lstm = nn.LSTM(input_dim+embed_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
        self.look_forward = look_forward
        self.device = device

    def forward(self, x, station_id):
        batch_size, time_step, embed_dim = x.shape
        station_embed = self.embed(torch.Tensor([station_id]).to(self.device))
        station_embed = station_embed.repeat(batch_size, time_step, 1)
        x = torch.cat([station_embed, x], dim=-1)
        x, _ = self.lstm(x)
        x = x[:, -self.look_forward:, :]
        x = self.fc(x)
        return x


# --------------------------------------------------QFormer----------------------------------------------------------
class QTrans(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout, n_heads, n_points, hidden_dim, output_dim, look_forward, n_layers=1):
        super(QTrans, self).__init__()
        # Positional encoding
        # self.positional_encoding = PositionalEncoding(embed_dim)
        self.token_embedding = nn.Linear(input_dim, embed_dim)
        self.embed_dropout = nn.Dropout(p=dropout)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim*n_points,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.look_forward = look_forward
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # Input x: (Batch, Length, Num Points, Embed Dim)
        B, L, N, D= x.size()
        # x = self.positional_encoding(x)
        x = self.token_embedding(x)
        x = self.embed_dropout(x)
        x = x.reshape(B, L, -1)
        # Pass through Transformer encoder
        x = self.encoder(x)
        x = x.reshape(B, L, N, -1)
        x = x[:, -1 * self.look_forward:, :, :]
        output = self.fc(x)
        return output



if __name__ == "__main__":
    import pandas as pd
    import torch
    from utils.data_load import StationDataset

    data = pd.read_excel("../outputs/filtered_feature_data.xlsx")
    dataset = StationDataset(data, 256, 15, 7, False)
    # model = MultiSiteTransformer(43, 7, 1, 64, 512, 32, 2,
    #                              4, 1, 0.1, 0.1)
    model = SHLDNN(43, 7, 15, 32, 32, 64, 32,
                 0.1, 0.1, if_quantile=True)
    batch = dataset.get_batch()
    station_ids, x, y = batch
    print(model(x.unsqueeze(-1), 0).shape)



