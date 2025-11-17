import pandas as pd

df1=pd.read_excel('./data/2/训练集/入库流量数据.xlsx')
print(df1.head())
print(df1.shape)
print(df1.info())
print('#####################################################################')
df2=pd.read_excel('./data/2/训练集/环境观测数据.xlsx')
print(df2.head())
print(df2.shape)
print(df2.info())
print('#####################################################################')
df3=pd.read_excel('./data/2/训练集/遥测站降雨数据.xlsx')
print(df3.head())
print(df3.shape)
print(df3.info())
print('#####################################################################')
df4=pd.read_excel('./data/2/训练集/降雨预报数据.xlsx')
print(df4.head())
print(df4.shape)
print(df4.info())
print('#####################################################################')
df5=pd.read_csv('./data/2/提交模板/初赛_提交模板.csv')
print(df5.head())
print(df5.shape)
print(df1.head())
df1['TimeStample'] = pd.to_datetime(df1['TimeStample'])
df1 = df1.set_index('TimeStample')

# 生成完整时间序列（3小时频率）
full_range = pd.date_range(start='2017-04-01 00:00:00',
                           end='2021-12-31 23:59:59',
                           freq='3H')

# 重新索引 → 检查缺失
df_full = df1.reindex(full_range)

# 缺失数量
print("缺失点数:", df_full['Qi'].isna().sum())
# df1['TimeStample'] = pd.to_datetime(df1['TimeStample'])
# df1 = df1.sort_values('TimeStample').set_index('TimeStample')

# 计算相邻时间间隔
diffs = df1.index.to_series().diff()

# 不是3小时的点
bad_points = df1.index[diffs != pd.Timedelta(hours=3)]
print("不连续点数量:", len(bad_points))
print("示例:", bad_points[:10])
print(df1.shape)
print(df1.head())
print(13555/3)
print((13555+2)/3)
df6=pd.read_excel('./data/2/测试集_初赛/预测01/入库流量数据.xlsx')
print(df6.shape)
df6=pd.read_excel('./data/2/测试集_初赛/预测02/入库流量数据.xlsx')
print(df6.shape)
df6=pd.read_excel('./data/2/测试集_初赛/预测03/入库流量数据.xlsx')
print(df6.shape)
# 生成完整时间索引（只按现有数据的首尾来）
full_range = pd.date_range(start=df1.index.min(),
                           end=df1.index.max(),
                           freq='3H')

# 按完整索引重建
df_full = df1.reindex(full_range)

print("补齐后的缺失点数:", df_full['Qi'].isna().sum())
print("缺失点位置:\n", df_full[df_full['Qi'].isna()].head())

# 插值补齐
df_full['Qi'] = df_full['Qi'].interpolate(method='time')

# 检查是否补上
print("补齐后缺失点数:", df_full['Qi'].isna().sum())
# 找出所有缺失点
missing_points = df1[df1['Qi'].isna()].index

# 分段查看缺失区间
from itertools import groupby
from operator import itemgetter

# 连续缺失点分组
groups = []
for k, g in groupby(enumerate(missing_points), lambda ix: ix[0] - ix[1].value):
    group = list(map(itemgetter(1), g))
    groups.append((group[0], group[-1], len(group)))

for start, end, count in groups:
    print(f"缺失区间: {start} → {end}, 共 {count} 点")
print(df_full.shape)
print(13888/3)
# 假设你已有 df_full (3H index), df3 (hourly), df2 (daily env)
len_3h = df_full.shape[0]
len_hourly = df3.shape[0]
len_daily = df2.shape[0]

print("3-hour points:", len_3h)
print("hourly points:", len_hourly)
print("daily points:", len_daily)
print("hourly / 3 == 3-hour? ->", len_hourly / 3, len_3h)
print("3-hour / 8 == days? ->", len_3h / 8, len_daily)
import matplotlib.pyplot as plt

# 只看 2019 年的数据
subset = df_full['2019-01-01':'2019-12-31']

plt.figure(figsize=(15,5))
plt.plot(subset.index, subset['Qi'], label='2019 Inflow', color='orange')

plt.title("Reservoir Inflow in 2019")
plt.xlabel("Time")
plt.ylabel("Qi")
plt.legend()
plt.grid(True)
plt.show()
df_full['Qi_smooth'] = df_full['Qi'].rolling(window=24, min_periods=1).mean()  # 相当于3天的窗口

plt.figure(figsize=(15,5))
plt.plot(df_full.index, df_full['Qi'], alpha=0.3, label='Raw Qi')
plt.plot(df_full.index, df_full['Qi_smooth'], color='red', label='Smoothed Qi (3-day rolling mean)')

plt.title("Reservoir Inflow (Smoothed)")
plt.xlabel("Time")
plt.ylabel("Qi")
plt.legend()
plt.grid(True)
plt.show()

print(df_full.head())
print(df_full.shape)
df61=pd.read_excel('./data/2/测试集_复赛/预测01/入库流量数据.xlsx')
print(df61.shape)
df62=pd.read_excel('./data/2/测试集_复赛/预测02/入库流量数据.xlsx')
print(df62.shape)
df63=pd.read_excel('./data/2/测试集_复赛/预测03/入库流量数据.xlsx')
print(df63.shape)
print(df1.head())
from sklearn.preprocessing import StandardScaler

dfs_test = [df61, df62, df63]

# 只取 Qi 列，保证列名一致
test_qi_list = [d[['Qi']] for d in dfs_test]

# 拼接成一个大 DataFrame
concat_all = pd.concat([df_full[['Qi']]] + test_qi_list, axis=0)
print("拼接后的 shape:", concat_all.shape)

scaler = StandardScaler()
scaler.fit(concat_all)  # 计算全局均值和标准差

print("全局均值:", scaler.mean_[0])
print("全局标准差:", scaler.scale_[0])

# 对训练集标准化
df_full['Qi_norm'] = scaler.transform(df_full[['Qi']])

# 对测试集标准化
for i, d in enumerate(dfs_test):
    d['Qi_norm'] = scaler.transform(d[['Qi']])

print(df_full.head())

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class SlidingWindowDataset(Dataset):
    # 移除 mean, std, series_norm, 以及内部的标准化代码
    def __init__(self, series, hist_len=240, pred_len=56, stride=1): # 移除 mean, std 参数
        # series 已经是外部标准化后的 NumPy 数组
        self.series = series.astype(np.float32)
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.stride = stride

        # ----------------------------------------------------
        # ⚠️ 关键：移除内部标准化代码
        # self.mean = np.mean(self.series) if mean is None else mean
        # self.std = np.std(self.series) if std is None else std
        # self.series_norm = (self.series - self.mean) / self.std
        self.series_norm = self.series # 直接使用传入的标准化数据
        # ----------------------------------------------------

        self.samples = []
        for start in range(0, len(self.series_norm) - hist_len - pred_len + 1, stride):
            # hist = self.series_norm[start:start+hist_len]
            # futr = self.series_norm[start+hist_len:start+hist_len+pred_len]
            hist = self.series_norm.iloc[start:start+hist_len].values
            futr = self.series_norm.iloc[start+hist_len:start+hist_len+pred_len].values
            self.samples.append((hist, futr))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist, futr = self.samples[idx]
        # ----------------------------------------------------
        # ⚠️ 关键：移除 __getitem__ 中的 unsqueeze(-1)
        # 我们将在 DataLoader 外部处理维度 [B, L] -> [B, L, 1]
        return (
            torch.tensor(hist),  # [hist_len]
            torch.tensor(futr)   # [pred_len]
        )
        # ----------------------------------------------------

import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]

def weighted_nse(y_true, y_pred, w1=0.65, w2=0.35):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    num1 = np.sum((y_true[:16] - y_pred[:16])**2)
    denom1 = np.sum((y_true[:16] - np.mean(y_true[:16]))**2) + 1e-6
    num2 = np.sum((y_true[16:] - y_pred[16:])**2)
    denom2 = np.sum((y_true[16:] - np.mean(y_true[16:]))**2) + 1e-6
    nse = 1 - w1 * (num1/denom1) - w2 * (num2/denom2)
    return nse

def evaluate(model, loader, scaler, device="cuda"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(-1)
            #print(x.shape)
            y = y.unsqueeze(-1)
            out = model(x)
            out = scaler.inverse_transform(out.squeeze(-1).cpu().numpy())
            y = scaler.inverse_transform(y.squeeze(-1).cpu().numpy())
            preds.append(out)
            trues.append(y)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    nse_scores = [weighted_nse(trues[i], preds[i]) for i in range(len(preds))]
    nse = np.mean(nse_scores)
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    return nse, mae, rmse

# =====================
# 4. 训练函数
# =====================
def train_model(df_full, scaler, hist_len=240, pred_len=56, epochs=100, batch_size=32, lr=1e-4, device="cuda"):
    dataset = SlidingWindowDataset(df_full["Qi_norm"], hist_len, pred_len)
    train_size = int(len(dataset)*0.8)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    configs = type("cfg", (), {})()
    configs.seq_len = hist_len
    configs.pred_len = pred_len
    configs.enc_in = 1
    configs.individual = False
    model = Model(configs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()   # 🚩训练时用MSE

    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            #print(x.shape)
            x = x.unsqueeze(-1)
            #print(x.shape)
            y = y.unsqueeze(-1)
            out = model(x)
            loss = criterion(out.squeeze(-1), y.squeeze(-1))  # MSE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # ===== 验证用 NSE =====
        nse, mae, rmse = evaluate(model, val_loader, scaler, device)
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {np.mean(losses):.6f} "
              f"| Val NSE: {nse:.4f} | Val MAE: {mae:.4f} | Val RMSE: {rmse:.4f}")

    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(df_full.head())
model = train_model(df_full, scaler, hist_len=240, pred_len=56, epochs=100, device=device)

print(df61.head())
# ==========================
# 预测并逆归一化
# ==========================
model.eval()
preds = []

for df in [df61['Qi_norm'], df62['Qi_norm'], df63['Qi_norm']]:
    # 准备输入
    #x = prepare_input(df, scaler, hist_len=240)  # [hist_len]
    hist_len=240
    x = torch.tensor(df[-hist_len:], dtype=torch.float32)  # 只取最后 hist_len
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # [1, hist_len, 1]

    with torch.no_grad():
        out = model(x)  # [1, pred_len, 1]

    out = out.squeeze().cpu().numpy()              # [pred_len]
    # 逆归一化
    out = scaler.inverse_transform(out.reshape(-1, 1)).flatten()  # [pred_len]
    preds.append(out)

# ==========================
# 填入提交模板
# ==========================
df_submit = df5.copy()

for i, pred in enumerate(preds):
    # 假设 df_submit 有足够行，这里按每段预测的顺序填入前 3 行
    for j, val in enumerate(pred):
        col_name = f"Prediction{j+1}"
        df_submit.loc[i, col_name] = val

# ==========================
# 保存 CSV
# ==========================
df_submit.to_csv("提交结果_复赛_flow.csv", index=False)
print(df_submit.head())
torch.save(model.state_dict(), "last_dlinear_inflow.pth")
print("✅ 最终模型已保存为 last_dlinear_inflow.pth")
