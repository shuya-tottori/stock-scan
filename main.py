import os
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
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

# 保有銘柄
MY_PORTFOLIO = ["9432.T", "8001.T", "8031.T", "8316.T", "1605.T", "4503.T", "8697.T", "8766.T"]

# =============================
# 外部指標（世界情勢）の取得
# =============================
def get_external_factors():
    try:
        # ドル円(JPY=X), S&P500(^GSPC), 日経先物(NIY=F)の直近リターン
        data = yf.download(["JPY=X", "^GSPC"], period="5d", progress=False)['Close']
        us_stock_change = (data["^GSPC"].iloc[-1] / data["^GSPC"].iloc[-2]) - 1
        usd_jpy_rate = data["JPY=X"].iloc[-1]
        usd_jpy_change = (data["JPY=X"].iloc[-1] / data["JPY=X"].iloc[-2]) - 1
        return us_stock_change, usd_jpy_rate, usd_jpy_change
    except:
        return 0, 150, 0

# =============================
# 解析ロジック
# =============================
def analyze_stock(code, data, ext_factors):
    try:
        df = data.xs(code, axis=1, level=1).copy().dropna(subset=['Close'])
        if len(df) < 50: return None
        
        last_price = df['Close'].iloc[-1]
        if last_price > BUDGET_LIMIT and code not in MY_PORTFOLIO: return None

        # 特徴量（テクニカル + 世界情勢）
        df['Return'] = df['Close'].pct_change()
        df['RSI'] = (lambda s, p=14: 100 - (100 / (1 + (s.diff().where(s.diff()>0,0).rolling(p).mean()/s.diff().where(s.diff()<0,0).abs().rolling(p).mean()).replace(0,np.nan))))(df['Close'])
        df['US_Stock_Effect'] = ext_factors[0] # S&P500の影響
        df['USD_JPY_Effect'] = ext_factors[2] # ドル円の影響
        
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df_train = df.dropna()
        
        X = df_train[['Return', 'RSI', 'US_Stock_Effect', 'USD_JPY_Effect']]
        y = df_train['Target']
        
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
        prob = model.predict_proba(X.iloc[-1:])[0][1]
        
        # 判定
        level = "対象外"
        if prob > 0.72: level = "🔥 超お宝株(激アツ)"
        elif prob > 0.65: level = "★★★ 厳選お宝株"
        elif prob > 0.58: level = "★ お宝候補"

        if level == "対象外" and code not in MY_PORTFOLIO: return None

        ticker = yf.Ticker(code)
        name = ticker.info.get('shortName', code)
        return {"code": code, "name": name, "price": last_price, "prob": prob, "level": level, "rsi": df['RSI'].iloc[-1]}
    except: return None

# =============================
# メイン
# =============================
def main():
    print("--- グローバルAI解析開始 ---")
    # 銘柄リストを225から主要約500銘柄(TOPIX500相当)に拡張するためのロジック
    # ここでは例として日経225 + ユーザー指定をスキャン
    df_codes = pd.read_csv("nikkei225.csv", header=None)
    base_codes = [str(c).zfill(4) + ".T" for c in df_codes.iloc[:, 0]]
    
    ext_factors = get_external_factors() # 世界情勢取得
    all_data = yf.download(list(set(base_codes + MY_PORTFOLIO)), period="1y", progress=False)
    
    # 1. 自信度ランク（市場全体の総評）
    us_change = ext_factors[0]
    market_status = "強気" if us_change > 0 else "慎重"
    market_comment = f"【AI自信度ランク：{market_status}】\n米国株の影響：{'上昇📈' if us_change > 0 else '下落📉'}\nドル円：{ext_factors[1]:.2f}円\n"

    # 2. 答え合わせ
    report_feedback = "＜前回の答え合わせ＞\n"
    if os.path.exists(SAVE_FILE):
        old_df = pd.read_csv(SAVE_FILE)
        for _, row in old_df.iterrows():
            code = row['code']
            if code in all_data.columns.get_level_values(1):
                cur = all_data.xs(code, axis=1, level=1)['Close'].iloc[-1]
                diff = cur - row['price']
                report_feedback += f"・{row['name']}: {row['price']:.0f}→{cur:.0f} ({'📈' if diff>0 else '📉'} {diff:+.0f})\n"
    else: report_feedback += "明日から表示されます\n"

    # 3. 解析実行
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_stock, code, all_data, ext_factors) for code in base_codes]
        results = [f.result() for f in futures if f.result() is not None]

    results.sort(key=lambda x: x['prob'], reverse=True)
    top_hits = [r for r in results if r['level'] != "対象外"][:10]
    if top_hits: pd.DataFrame(top_hits).to_csv(SAVE_FILE, index=False)

    # 4. 保有銘柄診断
    portfolio_report = "＜保有銘柄 健康診断＞\n"
    for code in MY_PORTFOLIO:
        res = next((r for r in results if r['code'] == code), None)
        if res:
            # 簡易買い増し判定
            status = "✨ 買い増し狙い目！" if res['rsi'] < 45 else "☕ 安定"
            portfolio_report += f"・{res['name']}: {res['price']:.0f}円 ({status})\n"

    # 5. 送信
    now = datetime.now() + timedelta(hours=9)
    body = f"【AIグローバルレポート - {now.strftime('%Y/%m/%d %H:%M')}】\n\n"
    body += market_comment + "\n" + report_feedback + "\n" + "─"*20 + "\n\n"
    body += portfolio_report + "\n" + "─"*20 + "\n\n"
    body += "＜本日の厳選銘柄＞\n"
    for r in top_hits:
        body += f"■ {r['name']} ({r['code']})\n{r['level']} (AI確率:{r['prob']:.1%})\n価格:{r['price']:.0f}円\n\n"

    msg = MIMEMultipart()
    msg["Subject"] = f"【AIお宝予測】自信度:{market_status} {now.strftime('%H:%M')}"
    msg["From"], msg["To"] = MAIL_ADDRESS, MAIL_TO
    msg.attach(MIMEText(body, "plain"))
    
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(MAIL_ADDRESS, MAIL_PASSWORD)
        server.send_message(msg)

if __name__ == "__main__":
    main()
