import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from quantile_forest import RandomForestQuantileRegressor
from models.DLFamilies import QLSTM, SHLDNN, MultiSiteTransformer
from utils.data_load import data_preprocess, StationDataset, get_samples
from utils.train_utils import train_multi_site_transformer
from utils.eval_utils import one_step_pred, get_pred_value, site_statis_eval, statis_eval, eval_pred_step
from tqdm import tqdm
import pickle


def train_main(args):
    def custom_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    df = pd.read_excel(args.train_data_path)

    if args.pred_kind == "p2p":
        pass
    elif args.pred_kind == "e2e":
        if args.model == "QRF":
            train_X, train_y = get_samples(df, args.look_back, 1, args.aggregate_step)
            train_X = np.array(train_X)[:, :, 0]
            train_y = np.array(train_y)[:, 0]
            # 添加站点编号
            n_samples = train_X.shape[0]
            site_numbers = np.repeat(np.arange(1, args.n_sites+1), n_samples/args.n_sites)
            train_X = np.column_stack((site_numbers, train_X))

            model = RandomForestQuantileRegressor()
            print("Start training.")
            model.fit(train_X, train_y)

            model_path = f"./trained_models/{args.model}-{args.suffix}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"The model has already been saved in '{model_path}'.")
            
        else:
            dataloader = StationDataset(df, args.batch_size, args.look_back, args.look_forward, args.if_aggregate,
                                        args.aggregate_step)

            if args.model == "Multi-site Transformer":
                args.input_dim = 1
                args.embed_dim = 64
                args.hidden_dim1 = 512
                args.hidden_dim2 = 32
                args.output_dim = 2
                args.n_heads = 2
                args.n_layers = 1
                args.dropout = 0.1
                model = MultiSiteTransformer(args.n_sites, args.look_forward, args.input_dim, args.embed_dim,
                                             args.hidden_dim1, args.hidden_dim2, args.output_dim, args.n_heads,
                                             args.n_layers, args.dropout, args.neg_slope)

            elif args.model == "SHL-DNN":
                args.embed_dim = 64
                args.hidden_dim1 = 32
                args.hidden_dim2 = 32
                args.hidden_dim3 = 64
                args.hidden_dim4 = 32
                args.dropout = 0.1
                model = SHLDNN(args.n_sites, args.look_forward, args.look_back, args.hidden_dim1, args.hidden_dim2,
                               args.hidden_dim3, args.hidden_dim4, args.dropout, args.neg_slope, args.pred_kind)

            elif args.model == "QLSTM":
                args.embed_dim = 4
                args.hidden_dim = 64
                args.n_layers = 1
                model = QLSTM(args.input_dim, args.embed_dim, args.hidden_dim, args.look_forward, args.n_layers)

            model = model.to(args.device)
            model_path = f"./trained_models/{args.model}-{args.suffix}.pth"

            train_multi_site_transformer(dataloader, model, args.loss_kind, args.lr, args.max_iters, args.earlystopping,
                                         args.alpha, model_path, args.interval, args.device, args.epsilon)


def eval_main(args):
    test_df = pd.read_excel(args.test_data_path)

    if args.pred_kind == "p2p":
        pass

    elif args.pred_kind == "e2e":
        test_X, test_y = get_samples(test_df, args.look_back, args.look_forward, args.aggregate_step, args.if_pred_iter,
                                     args.pred_iter)

        if args.model == "QRF":
            test_X = test_X[:, :, 0]
            test_y = np.array(test_y)
            # 添加站点编号
            n_samples = test_X.shape[0]
            site_numbers = np.repeat(np.arange(1, args.n_sites + 1), n_samples / args.n_sites)
            test_X = np.column_stack((site_numbers, test_X))

            model_path = f"./trained_models/{args.model}-{args.suffix}.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            y_pred = one_step_pred(test_X, args.interval, args.look_forward, model)
            lower_bound = y_pred[:, :, 0]
            upper_bound = y_pred[:, :, 1]

        else:
            if args.model == "Multi-site Transformer":
                args.embed_dim = 64
                args.hidden_dim1 = 512
                args.hidden_dim2 = 32
                args.output_dim = 2
                args.n_heads = 2
                args.n_layers = 1
                args.dropout = 0.1

            elif args.model == "SHL-DNN":
                args.embed_dim = 64
                args.hidden_dim1 = 32
                args.hidden_dim2 = 32
                args.hidden_dim3 = 64
                args.hidden_dim4 = 32
                args.dropout = 0.1

            elif args.model == "QLSTM":
                args.embed_dim = 4
                args.hidden_dim = 64
                args.n_layers = 1

            model_path = f"./trained_models/{args.model}-{args.suffix}.pth"
            model = torch.load(model_path)
            model.to(args.device)
            model.eval()
            y_pred = get_pred_value(model, test_X, args.n_sites, args.device, args.if_pred_iter)
            lower_bound = y_pred[:, :, 0].squeeze(-1).cpu().detach()
            upper_bound = y_pred[:, :, 1].squeeze(-1).cpu().detach()

    y_pred = (lower_bound + upper_bound) / 2
    rmse, nse, r2, picp, nmpiw, cwc, range_y = statis_eval(lower_bound, upper_bound, y_pred, test_y,
                                                            args.model, args.interval, args.n_sites)
    print(f"RMSE: {rmse:.3f}")
    print(f"NSE: {nse:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"PICP: {picp:.4f}")
    print(f"NMPIW: {nmpiw:.3f}")
    print(f"CWC: {cwc:.3f}")

    fig_dir = f"./figures/{args.model}/{args.middle_path}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    scale = pd.read_excel(args.scale_path)
    result_df = pd.DataFrame(columns=["Station_ID", "RMSE", "NSE", "R2", "PICP", "NMPIW", "CWC"], dtype=object)
    # 用于比较不同预测步精度
    pre_new_true = None
    pre_new_pred = None
    new_true = None
    new_pred = None
    # 用于风险评估
    true_code = None
    pred_code = None
    ori_data = pd.read_excel("./data/地下水埋深.xlsx").iloc[2:, 1:]
    thres = ori_data.quantile(0.9)

    for i in tqdm(range(args.n_sites)):
        mean_values = scale.iloc[i, 1]
        std_values = scale.iloc[i, 2]
        # lower = lower_bound[i * int(lower_bound.shape[0]/args.n_sites):(i+1) * int(lower_bound.shape[0]/args.n_sites), :]
        # upper = upper_bound[i * int(upper_bound.shape[0]/args.n_sites):(i+1) * int(upper_bound.shape[0]/args.n_sites), :]
        # y_test_pred = y_pred[i * int(y_pred.shape[0]/args.n_sites):(i+1) * int(y_pred.shape[0]/args.n_sites), :]
        # y_test_true = test_y[i * int(test_y.shape[0]/args.n_sites):(i+1) * int(test_y.shape[0]/args.n_sites), :]

        lower = lower_bound[:, :, i].unsqueeze(-1)
        upper = upper_bound[:, :, i].unsqueeze(-1)
        y_test_pred = y_pred[:, :, i].unsqueeze(-1)
        y_test_true = test_y[:, :, i].unsqueeze(-1)
        if isinstance(pre_new_pred, torch.Tensor):
            pre_new_true = torch.cat([pre_new_true, y_test_true])
            pre_new_pred = torch.cat([pre_new_pred, y_test_pred])
        elif isinstance(pre_new_pred, np.ndarray):
            pre_new_true = np.concatenate([pre_new_true, y_test_true])
            pre_new_pred = np.concatenate([pre_new_pred, y_test_pred])
        else:
            pre_new_true = y_test_true
            pre_new_pred = y_test_pred

        rmse, nse, r2, picp, nmpiw, cwc = site_statis_eval(lower, upper, y_test_pred, y_test_true, range_y[i])
        part_res = pd.DataFrame({"Station_ID": [f"S{i+1}"], "RMSE": [float(rmse)], "NSE": [float(nse)],
                                 "R2": [float(r2)], "PICP": [float(picp)], "NMPIW": [float(nmpiw)],
                                 "CWC": [float(cwc)]})
        result_df = pd.concat([result_df, part_res], axis=0)
        lower = lower.reshape(-1)
        upper = upper.reshape(-1)
        y_test_pred = y_test_pred.reshape(-1)
        y_test_true = y_test_true.reshape(-1)

        # Denormalize
        upper = upper * std_values + mean_values
        lower = lower * std_values + mean_values
        y_test_pred = y_test_pred * std_values + mean_values
        y_test_true = y_test_true * std_values + mean_values

        if isinstance(new_pred, torch.Tensor):
            site_pred_code = torch.where(upper < thres[i], 0, 1)
            site_true_code = torch.where(y_test_true < thres[i], 0, 1)
            new_true = torch.cat([new_true, y_test_true])
            new_pred = torch.cat([new_pred, y_test_pred])
            true_code = torch.cat([true_code, site_true_code])
            pred_code = torch.cat([pred_code, site_pred_code])
        elif isinstance(new_pred, np.ndarray):
            site_pred_code = np.where(upper < thres[i], 0, 1)
            site_true_code = np.where(y_test_true < thres[i], 0, 1)
            new_true = np.concatenate([new_true, y_test_true])
            new_pred = np.concatenate([new_pred, y_test_pred])
            true_code = np.concatenate([true_code, site_true_code])
            pred_code = np.concatenate([pred_code, site_pred_code])
        else:
            if isinstance(y_test_true, torch.Tensor):
                site_pred_code = torch.where(upper < thres[i], 0, 1)
                site_true_code = torch.where(y_test_true < thres[i], 0, 1)
            else:
                site_pred_code = np.where(upper < thres[i], 0, 1)
                site_true_code = np.where(y_test_true < thres[i], 0, 1)
            new_true = y_test_true
            new_pred = y_test_pred
            true_code = site_true_code
            pred_code = site_pred_code

    eval_pred_step(pre_new_pred, pre_new_true, args.eval_pred_step_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="./outputs/train_data_30_pred_15.xlsx", help="训练数据路径")
    parser.add_argument("--test_data_path", type=str, default="./outputs/test_data_30_pred_15.xlsx", help="测试数据路径")
    parser.add_argument("--eval_path", type=str, default="./outputs/trans_30_15_eval(95%).xlsx", help="站点统计指标路径")
    parser.add_argument("--eval_pred_step_path", type=str, default="./outputs/trans_30_15_eval_pred_step(95%).xlsx", help="逐步统计指标路径")
    parser.add_argument("--eval_fig_path", type=str, default="./figures/figure12_3.jpg", help="真值-预测散点图路径")
    parser.add_argument("--model", type=str, default="Multi-site Transformer", help="Multi-site Transformer, SHL-DNN, QLSTM, QRF")
    parser.add_argument("--suffix", type=str, default="e2e_30-15(95%)", help="模型参数文件尾缀")
    parser.add_argument("--pred_kind", type=str, default="e2e", help="p2p or e2e")
    parser.add_argument("--scale_path", type=str, default="./scale/EC_scale.xlsx", help="归一化文件路径")
    parser.add_argument("--middle_path", type=str, default="e2e(trans_30-15_95%)", help="可视化路径的中间文件夹")
    parser.add_argument("--data_lookback", type=int, default=30, help="为保证预测完整一年的测试集，所需提前的天数")
    parser.add_argument("--n_sites", type=int, default=43)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_dim", type=int, default=1, help="输入维度")
    parser.add_argument("--look_back", type=int, default=30, help="序列长度")
    parser.add_argument("--look_forward", type=int, default=15, help="预测长度")
    parser.add_argument("--if_aggregate", type=bool, default=False, help="是否聚合")
    parser.add_argument("--aggregate_step", type=int, default=1, help="聚合步长（按天计算）")
    parser.add_argument("--interval", default=[0.05, 0.95], help="上下分位数区间")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=1e-4, help="正则项系数")
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--earlystopping", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--neg_slope", type=float, default=0.01, help="LeakyReLU激活函数的负斜率")
    parser.add_argument("--loss_kind", type=str, default="quantile")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--if_pred_iter", type=bool, default=True, help="预测时是否进行迭代预测")
    parser.add_argument("--pred_iter", type=int, default=2, help="预测时迭代预测轮数")
    args = parser.parse_args()
    data_preprocess("./data/地下水埋深.xlsx", args.train_data_path,
                    args.test_data_path, "./outputs/filtered_feature_data.xlsx",
                    "./outputs/valueScale.xlsx", args.data_lookback)
    print(f"--------------{args.model}({30} days predict {30} days): {int(args.interval[1]*100)}% confidence lebel--------------")
    # train_main(args)
    eval_main(args)