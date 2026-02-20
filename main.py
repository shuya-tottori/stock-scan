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
# 1銘柄ずつ取得して解析する関数
# =============================
def get_and_analyze(code, ext_factors, is_portfolio=False):
    try:
        # 1銘柄だけダウンロード（これが一番確実）
        df = yf.download(code, period="1y", progress=False)
        if df.empty or len(df) < 40: return None
        
        last_price = float(df['Close'].iloc[-1])
        
        # 予算チェック（保有銘柄以外）
        if not is_portfolio and last_price > BUDGET_LIMIT: return None

        # 特徴量
        df['Return'] = df['Close'].pct_change()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
        
        # 世界情勢（米国株・ドル円）を結合
        df['US_Stock'] = ext_factors[0]
        df['USD_JPY'] = ext_factors[2]
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df_train = df.dropna()
        
        if len(df_train) < 20: return None

        features = ['Return', 'RSI', 'US_Stock', 'USD_JPY']
        model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
        model.fit(df_train[features], df_train['Target'])
        prob = float(model.predict_proba(df_train[features].iloc[-1:])[0][1])
        
        level = "対象外"
        if prob > 0.60: level = "🔥 超お宝株"
        elif prob > 0.53: level = "★★★ 厳選お宝株"
        elif prob > 0.48: level = "★ お宝候補"

        return {
            "code": code, "price": last_price, "prob": prob, 
            "level": level, "rsi": float(df['RSI'].iloc[-1])
        }
    except:
        return None

# =============================
# メイン
# =============================
def main():
    print("--- 安定版AI解析開始 ---")
    
    # 世界情勢の取得
    ext_data = yf.download(["^GSPC", "JPY=X"], period="5d", progress=False)['Close']
    try:
        us_change = (ext_data["^GSPC"].iloc[-1] / ext_data["^GSPC"].iloc[-2]) - 1
        usd_jpy = ext_data["JPY=X"].iloc[-1]
        usd_change = (ext_data["JPY=X"].iloc[-1] / ext_data["JPY=X"].iloc[-2]) - 1
    except:
        us_change, usd_jpy, usd_change = 0.0, 150.0, 0.0

    ext_factors = (us_change, usd_jpy, usd_change)

    # 銘柄リスト
    df_codes = pd.read_csv("nikkei225.csv", header=None)
    codes = [str(c).zfill(4) + ".T" for c in df_codes.iloc[:, 0]]

    # 1. 保有銘柄の解析
    portfolio_results = []
    print("保有銘柄チェック中...")
    for code in MY_PORTFOLIO:
        res = get_and_analyze(code, ext_factors, is_portfolio=True)
        if res: portfolio_results.append(res)
        time.sleep(0.5) # サーバーに優しく

    # 2. 全銘柄からお宝探し（時間がかかるので上位のみメール）
    print("全銘柄スキャン中...")
    all_hits = []
    for code in codes:
        if code in MY_PORTFOLIO: continue
        res = get_and_analyze(code, ext_factors)
        if res and res['level'] != "対象外":
            all_hits.append(res)
        # 1銘柄ごとにわずかに待機
        time.sleep(0.2)

    all_hits.sort(key=lambda x: x['prob'], reverse=True)
    top_hits = all_hits[:8]
    if top_hits: pd.DataFrame(top_hits).to_csv(SAVE_FILE, index=False)

    # 3. メールの組み立て
    status = "強気" if us_change > 0.003 else ("弱気" if us_change < -0.003 else "慎重")
    now = datetime.now() + timedelta(hours=9)
    
    body = f"【AI解析レポート - {now.strftime('%Y/%m/%d %H:%M')}】\n\n"
    body += f"自信度：{status}\n米国株：{us_change:.2%}\nドル円：{usd_jpy:.2f}円\n\n"
    
    body += "＜保有銘柄 健康診断＞\n"
    for r in portfolio_results:
        diag = "✨ 買い増し狙い目！" if r['rsi'] < 45 else ("🚀 絶好調" if r['rsi'] > 65 else "☕ 安定")
        body += f"・{r['code']}: {r['price']:.0f}円 ({diag})\n"

    body += "\n" + "─"*20 + "\n\n＜本日の厳選お宝銘柄＞\n"
    if top_hits:
        for r in top_hits:
            body += f"■ {r['code']}\n判定: {r['level']} (確率:{r['prob']:.1%})\n価格: {r['price']:.0f}円\n\n"
    else:
        body += "該当なし（今は待ちの姿勢です）☕\n"

    # 送信
    msg = MIMEMultipart()
    msg["Subject"] = f"【AI解析】自信度:{status} {now.strftime('%H:%M')}"
    msg["From"], msg["To"] = MAIL_ADDRESS, MAIL_TO
    msg.attach(MIMEText(body, "plain"))
    
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(MAIL_ADDRESS, MAIL_PASSWORD)
        server.send_message(msg)
    print("完了！")

if __name__ == "__main__":
    main()
