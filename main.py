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

SECTOR_MAP = {
    "半導体": ["6723.T", "8035.T", "6146.T", "6857.T", "6920.T"],
    "銀行・金融": ["8306.T", "8316.T", "8411.T", "8604.T"],
    "商社": ["8001.T", "8031.T", "8053.T", "8058.T"],
    "自動車": ["7203.T", "7267.T", "7269.T"]
}

def load_codes():
    if not os.path.exists("nikkei225.csv"): return []
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
        df = data.xs(code, axis=1, level=1).copy()
        df.dropna(subset=['Close'], inplace=True)
        if len(df) < 50: return None

        # 特徴量作成
        df['Return'] = df['Close'].pct_change()
        df['MA25_Slope'] = df['Close'].rolling(25).mean().diff(3)
        df['RSI'] = calc_rsi(df['Close'])
        
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
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(df_train[features], df_train['Target'])
        
        prob = model.predict_proba(df_train[features].iloc[-1:])[0][1]
        last = df.iloc[-1]
        last_rsi = last['RSI']

        # --- 絞り込みロジックの強化 ---
        level = "対象外"
        # 1. 激アツ：AI高確率 かつ 流行セクター
        if prob > 0.75 and sector_ret > 0.01:
            level = f"★★★（流行の{current_sector}×AI強気）"
        # 2. 買い：AI高確率 かつ RSIが低め
        elif prob > 0.70 and last_rsi < 45:
            level = "★★（反発期待）"
        # 3. 注目：AIがかなり自信あり（65%以上）
        elif prob > 0.68:
            level = "★（AI注目）"
        # 4. 次点：条件が重なっているものだけを残す（60%以上 かつ RSIが過熱していない）
        elif prob > 0.62 and last_rsi < 60:
            level = "次点（惜しい！）"

        if level == "対象外": return None

        # 銘柄名の取得（1回ずつTickerを叩く）
        ticker = yf.Ticker(code)
        name = ticker.info.get('shortName', code)

        return {
            "code": code, "name": name, "price": last['Close'], "rsi": last_rsi,
            "prob": prob, "level": level, "sector": current_sector
        }
    except: return None

# =============================
# メイン
# =============================

def main():
    codes = load_codes()
    all_data = yf.download(codes, period="1y", progress=False)
    
    downloaded_codes = all_data.columns.get_level_values(1).unique()
    sector_perf = {}
    for s_name, s_codes in SECTOR_MAP.items():
        rets = [all_data.xs(c, axis=1, level=1)['Close'].pct_change().iloc[-1] for c in s_codes if c in downloaded_codes]
        sector_perf[s_name] = np.mean(rets) if rets else 0

    results = []
    # 銘柄名取得を含むため、少しスレッド数を減らして安定させる
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_stock, code, all_data, sector_perf) for code in codes]
        for f in futures:
            res = f.result()
            if res: results.append(res)
    
    # メール本文作成
    date_str = datetime.now().strftime("%Y/%m/%d")
    body = f"【{date_str} 厳選スキャンレポート】\n\n"
    
    if results:
        # AI確率順に並び替え
        results.sort(key=lambda x: x['prob'], reverse=True)
        # 上位10件に絞る
        top_results = results[:10]
        
        for r in top_results:
            body += f"■ {r['name']} ({r['code']})\n"
            body += f"判定: {r['level']}\n"
            body += f"価格: {r['price']:.0f}円 / AI確率: {r['prob']:.1%} / RSI: {r['rsi']:.1f}\n\n"
        
        if len(results) > 10:
            body += f"※他 {len(results)-10} 銘柄が候補にありましたが、上位のみ表示しています。\n"
    else:
        body += "本日、厳しい条件をクリアした銘柄はありませんでした。\n"

    # 送信
    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = MAIL_ADDRESS, MAIL_TO, f"【厳選AI予測】{date_str}"
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(MAIL_ADDRESS, MAIL_PASSWORD)
        server.send_message(msg)

if __name__ == "__main__":
    main()
