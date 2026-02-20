import os
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ThreadPoolExecutor

# =============================
# 設定
# =============================
MAIL_ADDRESS = os.getenv("MAIL_ADDRESS")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_TO = os.getenv("MAIL_TO") if os.getenv("MAIL_TO") else MAIL_ADDRESS

BUDGET_LIMIT = 2000 
SAVE_FILE = "recommended.csv"

# 【保有銘柄】
MY_PORTFOLIO = ["9432.T", "8001.T", "8031.T", "8316.T", "1605.T", "4503.T", "8697.T", "8766.T"]

# =============================
# 粘り強いデータ取得関数
# =============================
def fetch_data_robust(tickers, period="1y"):
    """失敗しても3回までリトライするデータ取得"""
    for i in range(3):
        try:
            data = yf.download(tickers, period=period, progress=False, group_by='ticker')
            if not data.empty:
                return data
        except Exception as e:
            print(f"データ取得失敗 (試行 {i+1}): {e}")
        time.sleep(2) # 2秒待ってリトライ
    return pd.DataFrame()

# =============================
# 解析ロジック
# =============================
def analyze_stock(code, all_data, ext_factors):
    try:
        # group_by='ticker' の場合のデータ抽出
        if code not in all_data.columns.levels[0]: return None
        df = all_data[code].copy().dropna(subset=['Close'])
        
        if len(df) < 40: return None
        last_price = df['Close'].iloc[-1]
        
        # 特徴量
        df['Return'] = df['Close'].pct_change()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
        
        df['US_Stock_Effect'] = ext_factors[0]
        df['USD_JPY_Effect'] = ext_factors[2]
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df_train = df.dropna()
        if len(df_train) < 20: return None

        features = ['Return', 'RSI', 'US_Stock_Effect', 'USD_JPY_Effect']
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(df_train[features], df_train['Target'])
        prob = model.predict_proba(df_train[features].iloc[-1:])[0][1]
        
        # 判定
        level = "対象外"
        if prob > 0.62: level = "🔥 超お宝株(激アツ)"
        elif prob > 0.55: level = "★★★ 厳選お宝株"
        elif prob > 0.48: level = "★ お宝候補"

        if level == "対象外" and code not in MY_PORTFOLIO: return None

        return {"code": code, "price": last_price, "prob": prob, "level": level, "rsi": df['RSI'].iloc[-1]}
    except: return None

# =============================
# メイン
# =============================
def main():
    print("--- グローバルAI解析開始 ---")
    
    # 世界情勢取得 (ここもリトライ)
    ext_data = fetch_data_robust(["JPY=X", "^GSPC"], period="5d")
    try:
        us_stock_change = (ext_data["^GSPC"]["Close"].iloc[-1] / ext_data["^GSPC"]["Close"].iloc[-2]) - 1
        usd_jpy_rate = ext_data["JPY=X"]["Close"].iloc[-1]
        usd_jpy_change = (ext_data["JPY=X"]["Close"].iloc[-1] / ext_data["JPY=X"]["Close"].iloc[-2]) - 1
    except:
        us_stock_change, usd_jpy_rate, usd_jpy_change = 0.0, 0.0, 0.0

    ext_factors = (us_stock_change, usd_jpy_rate, usd_jpy_change)
    
    # 銘柄リスト
    df_codes = pd.read_csv("nikkei225.csv", header=None)
    base_codes = [str(c).zfill(4) + ".T" for c in df_codes.iloc[:, 0]]
    all_target_codes = list(set(base_codes + MY_PORTFOLIO))
    
    # 全データ一括取得
    all_data = fetch_data_robust(all_target_codes, period="1y")
    if all_data.empty:
        print("データが取得できませんでした")
        return

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_stock, code, all_data, ext_factors) for code in all_target_codes]
        results = [f.result() for f in futures if f.result() is not None]

    # メールの組み立て
    market_status = "強気" if us_stock_change > 0.003 else ("弱気" if us_stock_change < -0.003 else "慎重")
    market_comment = f"【AI自信度ランク：{market_status}】\n米国株影響：{us_stock_change:.2%}\nドル円：{usd_jpy_rate:.2f}円\n"

    # 1. 保有銘柄診断
    portfolio_report = "＜保有銘柄 健康診断＞\n"
    for code in MY_PORTFOLIO:
        res = next((r for r in results if r['code'] == code), None)
        if res:
            status = "✨ 買い増し狙い目！" if res['rsi'] < 45 else ("🚀 絶好調" if res['rsi'] > 65 else "☕ 安定")
            portfolio_report += f"・{code}: {res['price']:.0f}円 ({status})\n"
        else:
            portfolio_report += f"・{code}: 取得失敗\n"

    # 2. 厳選銘柄
    recommendations = [r for r in results if r['code'] not in MY_PORTFOLIO and r['level'] != "対象外" and r['price'] <= BUDGET_LIMIT]
    recommendations.sort(key=lambda x: x['prob'], reverse=True)
    top_hits = recommendations[:8]
    if top_hits: pd.DataFrame(top_hits).to_csv(SAVE_FILE, index=False)

    # 3. 送信
    now = datetime.now() + timedelta(hours=9)
    body = f"【AIグローバルレポート - {now.strftime('%Y/%m/%d %H:%M')}】\n\n{market_comment}\n{portfolio_report}\n"
    body += "─"*20 + "\n\n＜本日の厳選お宝銘柄（2000円以下）＞\n"
    if top_hits:
        for r in top_hits:
            body += f"■ {r['code']}\n判定: {r['level']} (AI確率:{r['prob']:.1%})\n価格: {r['price']:.0f}円\n\n"
    else:
        body += "該当なし（慎重相場です）☕\n"

    msg = MIMEMultipart()
    msg["Subject"] = f"【AI解析】自信度:{market_status} {now.strftime('%H:%M')}"
    msg["From"], msg["To"] = MAIL_ADDRESS, MAIL_TO
    msg.attach(MIMEText(body, "plain"))
    
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(MAIL_ADDRESS, MAIL_PASSWORD)
        server.send_message(msg)

if __name__ == "__main__":
    main()
