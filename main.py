import os
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ThreadPoolExecutor

# =============================
# 業種データの定義（簡易版：日経225主要セクター）
# =============================
# 本来はCSVから読み込むのが理想ですが、コード内で主要な流行セクターを定義します
SECTOR_MAP = {
    "半導体": ["6723.T", "8035.T", "6146.T", "6857.T", "6920.T"],
    "自動車": ["7203.T", "7267.T", "7269.T"],
    "銀行・金融": ["8306.T", "8316.T", "8411.T", "8604.T"],
    "商社": ["8001.T", "8031.T", "8053.T", "8058.T"]
}

def analyze_stock(code, data, sector_performance):
    try:
        df = data.xs(code, axis=1, level=1).copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
        df.dropna(subset=['Close'], inplace=True)
        if len(df) < 60: return None

        # --- 特徴量エンジニアリング ---
        df['Return'] = df['Close'].pct_change()
        df['MA25_Slope'] = df['Close'].rolling(25).mean().diff(3)
        df['RSI'] = calc_rsi(df['Close'])
        
        # 【新】世相（セクター）要因の追加
        # その銘柄のセクターがリストにあれば、その日のセクター平均騰落率を紐付ける
        target_sector_return = 0
        current_sector_name = "その他"
        for s_name, s_codes in SECTOR_MAP.items():
            if code in s_codes:
                target_sector_return = sector_performance.get(s_name, 0)
                current_sector_name = s_name
                break
        
        df['Sector_Score'] = target_sector_return
        
        # AI学習
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df_train = df.dropna()
        
        features = ['Return', 'RSI', 'MA25_Slope', 'Sector_Score']
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(df_train[features], df_train['Target'])
        
        prob = model.predict_proba(df_train[features].iloc[-1:])[0][1]
        
        # --- 判定ロジック ---
        last = df.iloc[-1]
        is_hot_sector = target_sector_return > 0.01 # セクター全体が1%以上上昇＝流行中
        
        level = "対象外"
        if prob > 0.7 and is_hot_sector: level = f"★★★（流行の{current_sector_name}×AI）"
        elif prob > 0.65 and last['RSI'] < 40: level = "★★（反発期待）"
        elif prob > 0.60: level = "次点（惜しい！）"

        if level == "対象外": return None

        return {
            "code": code, "level": level, "prob": prob, 
            "sector": current_sector_name, "price": last['Close']
        }
    except: return None

def main():
    codes = load_codes()
    all_data = yf.download(codes, period="1y", progress=False)

    # 1. 各セクターの「今日の勢い」を計算
    sector_performance = {}
    for s_name, s_codes in SECTOR_MAP.items():
        try:
            # セクター内銘柄の平均騰落率
            returns = [all_data.xs(c, axis=1, level=1)['Close'].pct_change().iloc[-1] for c in s_codes if c in all_data.columns.levels[1]]
            sector_performance[s_name] = np.mean(returns)
        except: sector_performance[s_name] = 0

    # 2. 解析実行
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(analyze_stock, code, all_data, sector_performance) for code in codes]
        results = [f.result() for f in futures if f.result() is not None]

    # 3. メール作成（「流行セクター」を強調）
    # ...（前述のメール送信ロジックに、r['sector'] の情報を加える）
