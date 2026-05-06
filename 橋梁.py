import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.dates as mdates
import io



# ==========================================
# 1. 核心函數定義
# ==========================================

def load_and_preprocess_bridge_data(filepath, resample_freq='5min', neutral_axis_baseline=207):
    """
    載入數據、基準化處理 (offset)、重採樣並處理瞬時變化
    """
    print(f"正在載入橋梁數據: {filepath}...")
    try:
        # 使用 low_memory=False 解決 DtypeWarning
        df = pd.read_csv(filepath, low_memory=False)

        # 處理時間索引
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        df = df.dropna(subset=[df.columns[0]])
        df.set_index(df.columns[0], inplace=True)

        # 移除無用欄位
        cols_to_drop = [c for c in df.columns if '理論' in str(c) or 'Unnamed' in str(c)]
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # 統一高度欄位名稱並轉為數值
        df.rename(columns={df.columns[0]: 'Value'}, inplace=True)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df[['Value']].dropna()
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='last')]

        # 重採樣與線性插值
        df_resampled = df.resample(resample_freq).mean()
        df_resampled['Value'] = df_resampled['Value'].interpolate(method='time', limit_direction='both')

        # --- 新增中性軸偏差量 (以 207 為基準) ---
        df_resampled['offset'] = df_resampled['Value'] - neutral_axis_baseline

        # 新增基於偏移量的瞬時變化
        df_resampled['diff_1'] = df_resampled['offset'].diff(1)
        df_resampled['diff_2'] = df_resampled['offset'].diff(2)

        df_resampled.dropna(inplace=True)
        return df_resampled
    except Exception as e:
        print(f"讀取或處理失敗: {e}")
        raise

def create_advanced_features(df):
    """
    建立週期性時間特徵 (Sin/Cos) 與滯後特徵 (Lags)
    """
    df_feat = df.copy()

    # --- 時間維度 Sin/Cos 轉換 ---
    # 讓模型理解 23:55 與 00:00 的連續性
    df_feat['hour'] = df_feat.index.hour
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24.0)

    # 其他時間輔助特徵
    df_feat['dayofweek'] = df_feat.index.dayofweek

    # 建立滯後特徵 (讓模型看到過去的趨勢)
    lags = [1, 2, 3, 24] # 5min, 10min, 15min, 2hr (若採樣為5min)
    for lag in lags:
        df_feat[f'offset_lag_{lag}'] = df_feat['offset'].shift(lag)

    df_feat.dropna(inplace=True)
    df_feat.drop(columns=['hour'], inplace=True) # 移除原始 hour
    return df_feat

def create_sequences(data, seq_length, prediction_window=1, target_col_idx=0):
    """
    建立深度學習(LSTM)用的滑動視窗序列
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length - prediction_window + 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length : i + seq_length + prediction_window, target_col_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ==========================================
# 2. 主流程：執行資料整合
# ==========================================

# 1. 執行基礎預處理
BRIDGE_FILE_PATH = 'bridge.csv'
df_base = load_and_preprocess_bridge_data(BRIDGE_FILE_PATH)

# 2. 執行進階特徵工程 (Sin/Cos 轉換)
df_final = create_advanced_features(df_base)

# 3. 準備模型輸入
# 我們選擇 offset 相關的特徵，排除原始 Value
features = [col for col in df_final.columns if col != 'Value']
target_col = 'offset'
offset_idx = features.index(target_col)

# 4. 劃分訓練/測試集
train_size = int(len(df_final) * 0.8)
train_df = df_final[features].iloc[:train_size]
test_df = df_final[features].iloc[train_size:]

# 5. 資料標準化
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

# 6. 建立 LSTM 序列 (使用 24 個時間步長預測下一個點)
SEQ_LENGTH = 24
X_train_seq, y_train_seq = create_sequences(train_scaled, SEQ_LENGTH, target_col_idx=offset_idx)
X_test_seq, y_test_seq = create_sequences(test_scaled, SEQ_LENGTH, target_col_idx=offset_idx)

# ==========================================
# 3. 結果確認
# ==========================================
print("\n" + "="*30)
print("SHM 數據準備完成報告")
print("="*30)
print(f"使用的特徵欄位: {features}")
print(f"訓練集形狀: {X_train_seq.shape}")
print(f"測試集形狀: {X_test_seq.shape}")
print(f"目標偏移量(Offset)索引位置: {offset_idx}")
print("="*30)

"""LSTM Autoencoder"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 修正後的 LSTM Autoencoder 架構
# ==========================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        _, (hidden_n, _) = self.encoder(x)
        hidden_n = hidden_n.permute(1, 0, 2)
        decoder_input = hidden_n.repeat(1, self.seq_len, 1)
        decoder_output, _ = self.decoder(decoder_input)
        reconstruction = self.output_layer(decoder_output)
        return reconstruction

# ==========================================
# 2. 修正後的訓練與評估函數 (重點修改解包邏輯)
# ==========================================
def train_model(model, train_loader, epochs, lr, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    history = []

    print("開始訓練橋梁正常狀態行為模型 (LSTM-AE)...")
    for epoch in range(epochs):
        epoch_loss = 0
        # 【修改點 1】：使用 batch_data 接收 tuple，然後取第一個元素
        for batch_data in train_loader:
            batch_x = batch_data[0].to(device) # 取出 Tensor 並推上 GPU/CPU

            optimizer.zero_grad()
            reconstruction = model(batch_x)
            loss = criterion(reconstruction, batch_x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], 平均重建誤差 (MSE): {avg_loss:.6f}')

    return history

def get_reconstruction_errors(model, data_loader, device):
    model.eval()
    errors = []
    criterion = nn.L1Loss(reduction='none') # MAE 誤差

    with torch.no_grad():
        # 【修改點 2】：同理，解決測試資料集 DataLoader 的 tuple 解包問題
        for batch_data in data_loader:
            batch_x = batch_data[0].to(device)
            reconstruction = model(batch_x)

            loss = criterion(reconstruction, batch_x)
            sample_errors = loss.mean(dim=[1, 2]).cpu().numpy()
            errors.extend(sample_errors)

    return np.array(errors)

# ==========================================
# 3. 主流程：執行訓練與計算誤差
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用運算裝置: {device}")

# 確保上一階段的資料轉換為 Tensor 並建立 DataLoader
X_train_tensor = torch.FloatTensor(X_train_seq)
X_test_tensor = torch.FloatTensor(X_test_seq)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor)

BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型 (根據輸入資料的特徵維度)
n_features = X_train_seq.shape[2]
model = LSTMAutoencoder(seq_len=24, n_features=n_features, embedding_dim=32).to(device)

# 訓練模型
EPOCHS = 30
LEARNING_RATE = 0.001
loss_history = train_model(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=device)

# 計算訓練集與測試集的「重建誤差」(Damage Index)
train_errors = get_reconstruction_errors(model, train_loader, device)
test_errors = get_reconstruction_errors(model, test_loader, device)

print(f"\n✅ 模型訓練與誤差計算完成！")
print(f"訓練集誤差平均值: {np.mean(train_errors):.6f}, 標準差: {np.std(train_errors):.6f}")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 4. 定義 SHM 預警門檻 (L1, L2, L3)
# ==========================================
# 在 SHM 中，我們以「訓練集 (健康狀態)」的誤差分佈來建立基準
mu = np.mean(train_errors)
sigma = np.std(train_errors)

# 定義三級預警門檻 (基於常態分佈假設)
L1_threshold = mu + 2 * sigma  # 注意 (涵蓋 95.4% 的健康狀況)
L2_threshold = mu + 3 * sigma  # 警告 (涵蓋 99.7% 的健康狀況)
L3_threshold = mu + 5 * sigma  # 危險 (極端偏離，極可能是感測器故障或結構受損)

print(f"🌉 橋梁健康狀態基準 (Damage Index):")
print(f"平均誤差 (μ): {mu:.6f}, 標準差 (σ): {sigma:.6f}")
print(f"🟢 正常範圍: < {L1_threshold:.6f}")
print(f"🟡 L1 注意門檻 (μ+2σ): {L1_threshold:.6f}")
print(f"🟠 L2 警告門檻 (μ+3σ): {L2_threshold:.6f}")
print(f"🔴 L3 危險門檻 (μ+5σ): {L3_threshold:.6f}")

# ==========================================
# 5. 繪製 SHM 預警儀表板
# ==========================================
plt.figure(figsize=(18, 12))
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 避免中文亂碼 (若有需要可依作業系統更改字體)

# --- 圖 1: 模型訓練收斂曲線 --- 圖 1 (收斂圖)：如果曲線平滑下降並趨於平穩，代表你的 LSTM-AE 已經成功記住了這座橋「熱脹冷縮」的週期規律。
plt.subplot(3, 1, 1)
plt.plot(loss_history, label='Training Loss (MSE)', color='blue', linewidth=2)
plt.title('Figure-1 模型收斂曲線 (Model Converge curve)', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Reconstruction Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- 圖 2: 重建誤差 (Damage Index) 分佈與預警門檻 ---
#圖 2 (分佈圖)：
#藍色區塊是橋梁的「健康指紋」。
#如果橘色區塊 (Test) 大量向右偏移並跨過紅線 (L3)，這在物理上意味著橋梁產生了**「模型無法解釋的行為」**（例如：颱風過後支承墊產生不可逆的位移、或是重載車隊通過）。
plt.subplot(3, 1, 2)
sns.histplot(train_errors, bins=50, color='blue', alpha=0.5, label='健康狀態 (Train)', kde=True)
sns.histplot(test_errors, bins=50, color='orange', alpha=0.5, label='未知狀態 (Test)', kde=True)

# 畫出預警門檻線
plt.axvline(L1_threshold, color='gold', linestyle='--', linewidth=2, label='L1 注意 (μ+2σ)')
plt.axvline(L2_threshold, color='darkorange', linestyle='--', linewidth=2, label='L2 警告 (μ+3σ)')
plt.axvline(L3_threshold, color='red', linestyle='--', linewidth=2, label='L3 危險 (μ+5σ)')

plt.title('Figure-2 重建誤差分佈與 L1/L2/L3 預警門檻 (Damage Index Distribution)', fontsize=14)
plt.xlabel('Reconstruction Error (MAE)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# --- 圖 3: 測試集時間序列預警散佈圖 ---
#圖 3 (時間序列圖)：這是未來要放在戰情室螢幕上的圖。
#如果出現零星的紅點 (L3)，可能是感測器瞬間雜訊；但如果紅點連續出現超過 30 分鐘，系統就必須自動發送簡訊給巡檢人員
plt.subplot(3, 1, 3)
# 建立一個測試集的時間軸 (這裡用索引代替，若有真實時間戳會更好)
time_axis = np.arange(len(test_errors))

# 根據門檻將點分類
normal_idx = np.where(test_errors < L1_threshold)[0]
l1_idx = np.where((test_errors >= L1_threshold) & (test_errors < L2_threshold))[0]
l2_idx = np.where((test_errors >= L2_threshold) & (test_errors < L3_threshold))[0]
l3_idx = np.where(test_errors >= L3_threshold)[0]

# 畫出散佈圖
plt.scatter(time_axis[normal_idx], test_errors[normal_idx], color='green', s=10, label='🟢 正常', alpha=0.6)
plt.scatter(time_axis[l1_idx], test_errors[l1_idx], color='gold', s=20, label='🟡 L1 注意')
plt.scatter(time_axis[l2_idx], test_errors[l2_idx], color='darkorange', s=30, label='🟠 L2 警告')
plt.scatter(time_axis[l3_idx], test_errors[l3_idx], color='red', s=40, label='🔴 L3 危險')

# 畫出背景門檻區間以利識別
plt.axhspan(0, L1_threshold, facecolor='green', alpha=0.05)
plt.axhspan(L1_threshold, L2_threshold, facecolor='yellow', alpha=0.05)
plt.axhspan(L2_threshold, L3_threshold, facecolor='orange', alpha=0.05)
plt.axhspan(L3_threshold, max(test_errors.max(), L3_threshold*1.1), facecolor='red', alpha=0.05)

plt.title('Figure-3 時間序列預警監測 (Time-Series Anomaly Monitoring)', fontsize=14)
plt.xlabel('Time Steps (Testing Data)')
plt.ylabel('Reconstruction Error')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

"""LSTM"""

import matplotlib.dates as mdates

# ==========================================
# 1. 抓回真實的時間戳記 (Datetime)
# ==========================================
# test_df 是我們之前切出來的測試集 DataFrame
# 因為滑動視窗 (SEQ_LENGTH) 會消耗掉前面的資料，所以標籤 y 對應的時間是從 SEQ_LENGTH 開始
test_timestamps = test_df.index[SEQ_LENGTH:]

print(f"測試集時間範圍: {test_timestamps.min()} 到 {test_timestamps.max()}")

# ==========================================
# 2. 建立預測型 LSTM (翻譯自你的 Keras 架構)
# ==========================================
class ForecastingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(ForecastingLSTM, self).__init__()
        # 對應: model.add(LSTM(64, return_sequences=True))
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)

        # 對應: model.add(LSTM(64))
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)

        # 對應: model.add(Dense(1))
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # 只取最後一個時間步的輸出進行預測
        last_step_out = x[:, -1, :]
        out = self.fc(last_step_out)
        return out

# 初始化預測模型
forecast_model = ForecastingLSTM(input_size=n_features).to(device)
criterion_forecast = nn.MSELoss()
optimizer_forecast = optim.Adam(forecast_model.parameters(), lr=0.001)

# ==========================================
# 3. 訓練預測型 LSTM 模型
# ==========================================
print("\n開始訓練預測型 LSTM 模型 (Forecasting)...")
forecast_model.train()
EPOCHS_FORECAST = 30

for epoch in range(EPOCHS_FORECAST):
    epoch_loss = 0
    # 這裡我們需要 y_train_seq 作為預測目標 (偏移量 offset)
    # y_train_seq 已經在前面用 create_sequences 產生好了
    for i in range(0, len(X_train_tensor), BATCH_SIZE):
        batch_x = X_train_tensor[i:i+BATCH_SIZE].to(device)
        batch_y = torch.FloatTensor(y_train_seq[i:i+BATCH_SIZE]).to(device)

        optimizer_forecast.zero_grad()
        predictions = forecast_model(batch_x)
        loss = criterion_forecast(predictions, batch_y)
        loss.backward()
        optimizer_forecast.step()

        epoch_loss += loss.item()

    if (epoch+1) % 5 == 0:
         print(f'Epoch [{epoch+1}/{EPOCHS_FORECAST}], 預測誤差 (MSE): {epoch_loss/len(X_train_tensor):.6f}')

# ==========================================
# 4. 雙模型誤差計算與交叉驗證邏輯 (✅ 解決 OOM 的批次預測版)
# ==========================================
forecast_model.eval()
test_preds_list = []

print("正在進行批次預測，避免 GPU 記憶體超載...")
with torch.no_grad():
    # 【修改點】：使用 test_loader 分批次 (Batch) 拿取資料，一次只算一小批
    for batch_data in test_loader:
        batch_x = batch_data[0].to(device)
        preds = forecast_model(batch_x)
        test_preds_list.extend(preds.cpu().numpy()) # 將算完的結果存進列表

# 將批次收集完的列表，轉回我們需要的 Numpy 陣列
test_preds = np.array(test_preds_list)

# 1. 取得 AE 重建誤差 (來自上一個步驟的 test_errors)
ae_errors = test_errors

# 2. 取得預測模型的絕對誤差 (Residual)
y_test_actual = y_test_seq.reshape(-1, 1)
forecast_errors = np.abs(test_preds - y_test_actual).flatten()

# 定義預測模型的 L2 門檻 (以前 80% 資料的分位數作為粗略基準)
forecast_threshold = np.percentile(forecast_errors, 95)






# ==========================================
# 5. 繪製高階預警儀表板 (含時間戳與交叉驗證)
# ==========================================
plt.figure(figsize=(18, 14))
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

             # --- 圖 3: 加入真實時間戳的 AE 預警圖 ---
plt.subplot(2, 1, 1)

# 根據先前的 L1/L2/L3 門檻分類
normal_idx = np.where(ae_errors < L1_threshold)[0]
l3_idx = np.where(ae_errors >= L3_threshold)[0]
warning_idx = np.where((ae_errors >= L1_threshold) & (ae_errors < L3_threshold))[0]

# 使用真實時間戳 (test_timestamps) 繪圖
plt.scatter(test_timestamps[normal_idx], ae_errors[normal_idx], color='green', s=10, label='Normal(Green)', alpha=0.5)
plt.scatter(test_timestamps[warning_idx], ae_errors[warning_idx], color='darkorange', s=20, label='L1/L2 Warning(Orange)')
plt.scatter(test_timestamps[l3_idx], ae_errors[l3_idx], color='red', s=40, label='L3 Dangerous(Red)')

plt.axhline(L3_threshold, color='red', linestyle='--', linewidth=2, label='L3 Dangerous bound')

# 設定 X 軸時間格式 (顯示 月-日 時:分)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5)) # 每2天顯示一個刻度
plt.xticks(rotation=45)

plt.title('3. Time-series forecast monitor (Including Ture Datetime)', fontsize=14)
plt.xlabel('Date Time')
plt.ylabel('AE Reconstruction Error')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.title('',fontsize=14)

            # --- 圖 4: 雙模型交叉驗證散佈圖 (Cross-Validation) ---
plt.subplot(2, 1, 2)

# X軸: AE 誤差, Y軸: 預測誤差
plt.scatter(ae_errors, forecast_errors, alpha=0.6, color='royalblue', s=15)

# 畫出象限分隔線
plt.axvline(L3_threshold, color='red', linestyle='--', label='AE L3 門檻 (整體行為異常)')
plt.axhline(forecast_threshold, color='purple', linestyle='--', label='預測門檻 (瞬時數值異常)')

# 標示「雙重確認」的超級危險區
plt.fill_between([L3_threshold, max(ae_errors)*1.1],
                 forecast_threshold, max(forecast_errors)*1.1,
                 color='red', alpha=0.1, label='雙模型皆報警 (極高置信度)')

plt.title('4. 雙模型交叉驗證 (LSTM-AE vs Forecasting LSTM)', fontsize=14)
plt.xlabel('Autoencoder 重建誤差 (結構行為異常度)')
plt.ylabel('Forecasting 預測誤差 (瞬時數值異常度)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
