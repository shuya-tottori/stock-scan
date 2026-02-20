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

# 予算設定
BUDGET_LIMIT = 2000 

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

        last_price = df['Close'].iloc[-1]
        
        # 【重要】予算フィルター：1株2000円を超える銘柄は即除外
        if last_price > BUDGET_LIMIT:
            return None

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

        features = ['Return', 'RSI', 'MA25_Slope', 'Sector_Score']
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(df_train[features], df_train['Target'])
        
        prob = model.predict_proba(df_train[features].iloc[-1:])[0][1]
        last_rsi = df['RSI'].iloc[-1]

        # 判定
        level = "対象外"
        if prob > 0.70 and last_rsi < 45:
            level = "★★（S株チャンス！）"
        elif prob > 0.65:
            level = "★（お小遣い枠）"
        elif prob > 0.60 and last_rsi < 55:
            level = "次点（低価格・注目）"

        if level == "対象外": return None

        # 銘柄名取得
        ticker = yf.Ticker(code)
        name = ticker.info.get('shortName', code)

        return {
            "code": code, "name": name, "price": last_price, "rsi": last_rsi,
            "prob": prob, "level": level, "sector": current_sector
        }
    except: return None

# =============================
# メイン
# =============================

def main():
    codes = load_codes()
    # 2000円以下の株を確実に拾うために期間は1年分取得
    all_data = yf.download(codes, period="1y", progress=False)
    
    downloaded_codes = all_data.columns.get_level_values(1).unique()
    sector_perf = {}
    for s_name, s_codes in SECTOR_MAP.items():
        rets = [all_data.xs(c, axis=1, level=1)['Close'].pct_change().iloc[-1] for c in s_codes if c in downloaded_codes]
        sector_perf[s_name] = np.mean(rets) if rets else 0

    results = []
    # 銘柄名取得の負荷を考慮し、並列数を調整
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_stock, code, all_data, sector_perf) for code in codes]
        for f in futures:
            res = f.result()
            if res: results.append(res)
    
    date_str = datetime.now().strftime("%Y/%m/%d")
    body = f"【{date_str} S株/お小遣い枠厳選レポート】\n"
    body += f"（1株 {BUDGET_LIMIT}円以下の銘柄のみ抽出）\n\n"
    
    if results:
        results.sort(key=lambda x: x['prob'], reverse=True)
        # 厳選して最大8件表示
        for r in results[:8]:
            body += f"■ {r['name']} ({r['code']})\n"
            body += f"判定: {r['level']}\n"
            body += f"価格: {r['price']:.0f}円 / AI確率: {r['prob']:.1%} / RSI: {r['rsi']:.1f}\n\n"
    else:
        body += f"本日、{BUDGET_LIMIT}円以下で条件に合う銘柄はありませんでした。\n"

    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = MAIL_ADDRESS, MAIL_TO, f"【S株AI予測】{date_str}"
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(MAIL_ADDRESS, MAIL_PASSWORD)
        server.send_message(msg)

if __name__ == "__main__":
    main()
