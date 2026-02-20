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
# 設定
# =============================
MAIL_ADDRESS = os.getenv("MAIL_ADDRESS")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_TO = os.getenv("MAIL_TO", MAIL_ADDRESS)

# 主要セクター（世相判定用）
SECTOR_MAP = {
    "半導体": ["6723.T", "8035.T", "6146.T", "6857.T", "6920.T"],
    "銀行・金融": ["8306.T", "8316.T", "8411.T", "8604.T"],
    "商社": ["8001.T", "8031.T", "8053.T", "8058.T"],
    "自動車": ["7203.T", "7267.T", "7269.T"]
}

def load_codes():
    if not os.path.exists("nikkei225.csv"):
        print("Error: nikkei225.csv が見つかりません。")
        return []
    df = pd.read_csv("nikkei225.csv", header=None)
    return [str(c).zfill(4) + ".T" for c in df.iloc[:, 0]]

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# =============================
# 解析ロジック
# =============================

def analyze_stock(code, data, sector_performance):
    try:
        # yfinanceのMultiIndexから特定銘柄のデータを抜く
        df = data.xs(code, axis=1, level=1).copy()
        df.dropna(subset=['Close'], inplace=True)
        
        if len(df) < 50:
            return None

        # 特徴量作成
        df['Return'] = df['Close'].pct_change()
        df['MA25'] = df['Close'].rolling(25).mean()
        df['MA25_Slope'] = df['MA25'].diff(3)
        df['RSI'] = calc_rsi(df['Close'])
        
        # 世相スコア
        current_sector = "その他"
        sector_ret = 0
        for s_name, s_codes in SECTOR_MAP.items():
            if code in s_codes:
                current_sector = s_name
                sector_ret = sector_performance.get(s_name, 0)
                break
        
        df['Sector_Score'] = sector_ret
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        df_train = df.dropna()
        if len(df_train) < 30: return None

        # AI学習
        features = ['Return', 'RSI', 'MA25_Slope', 'Sector_Score']
        X = df_train[features]
        y = df_train['Target']
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        
        prob = model.predict_proba(X.iloc[-1:])[0][1]
        last = df.iloc[-1]
        
        # 判定
        level = "対象外"
        if prob > 0.7 and sector_ret > 0.01:
            level = f"★★★（流行の{current_sector}×AI強気）"
        elif last['RSI'] < 40 and prob > 0.6:
            level = "★★（反発期待）"
        elif prob > 0.65:
            level = "★（AI注目）"
        elif prob > 0.55:
            level = "次点（惜しい！）"

        if level == "対象外": return None

        return {
            "code": code, "price": last['Close'], "rsi": last['RSI'],
            "prob": prob, "level": level, "sector": current_sector
        }
    except Exception as e:
        return None

# =============================
# メイン
# =============================

def main():
    print("--- スキャン開始 ---")
    codes = load_codes()
    if not codes: return
    print(f"1. 銘柄読み込み完了: {len(codes)}件")

    print("2. 市場データ取得中...")
    all_data = yf.download(codes, period="1y", interval="1d", progress=False)
    
    # 取得できた銘柄数を確認
    downloaded_codes = all_data.columns.get_level_values(1).unique()
    print(f"   データ取得完了: {len(downloaded_codes)}銘柄分")

    # セクター勢い計算
    sector_perf = {}
    for s_name, s_codes in SECTOR_MAP.items():
        rets = []
        for c in s_codes:
            if c in downloaded_codes:
                try:
                    ret = all_data.xs(c, axis=1, level=1)['Close'].pct_change().iloc[-1]
                    rets.append(ret)
                except: pass
        sector_perf[s_name] = np.mean(rets) if rets else 0
    print(f"3. セクター勢い計算完了: {sector_perf}")

    print("4. AI解析実行中（並列処理）...")
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(analyze_stock, code, all_data, sector_perf) for code in codes]
        for f in futures:
            res = f.result()
            if res: results.append(res)
    
    print(f"5. 解析完了。ヒット件数: {len(results)}件")

    # メール本文作成
    date_str = datetime.now().strftime("%Y/%m/%d")
    body = f"【{date_str} 株式スキャン結果レポート】\n\n"
    
    if results:
        results.sort(key=lambda x: x['prob'], reverse=True)
        for r in results:
            body += f"■ {r['code']} ({r['sector']})\n"
            body += f"判定: {r['level']}\n"
            body += f"株価: {r['price']:.0f}円 / AI確率: {r['prob']:.1%} / RSI: {r['rsi']:.1f}\n\n"
    else:
        body += "本日の条件に合致する銘柄はありませんでした。\n"
    
    body += "--- 解析システムより ---"

    # 送信
    print("6. メール送信試行中...")
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"] = MAIL_ADDRESS, MAIL_TO
        msg["Subject"] = f"【AI予測】{date_str} スキャン結果"
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(MAIL_ADDRESS, MAIL_PASSWORD)
            server.send_message(msg)
        print("7. メール送信成功！")
    except Exception as e:
        print(f"7. メール送信失敗: {e}")

if __name__ == "__main__":
    main()
