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

# 【保有銘柄リスト】
MY_PORTFOLIO = [
    "9432.T", # NTT
    "8001.T", # 伊藤忠商事
    "8031.T", # 三井物産
    "8316.T", # 三井住友FG
    "1605.T", # INPEX
    "4503.T", # アステラス製薬
    "8697.T", # 日本取引所G
    "8766.T"  # 東京海上HD
]

# =============================
# 補助関数
# =============================

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def get_stock_status(code, data):
    try:
        df = data.xs(code, axis=1, level=1).copy().dropna(subset=['Close'])
        if len(df) < 30: return "データ不足"
        
        last_price = df['Close'].iloc[-1]
        rsi = calc_rsi(df['Close']).iloc[-1]
        ma25 = df['Close'].rolling(25).mean().iloc[-1]
        
        # 買い増し狙い目の判定：移動平均より安く、売られすぎ(RSI 45以下)のとき
        if last_price < ma25 and rsi < 45: 
            return "✨ 絶好の買い増し狙い目！"
        if last_price > ma25 and rsi > 60: 
            return "🚀 絶好調 (イケイケ状態)"
        if rsi < 30:
            return "⚠️ かなり割安 (反発待ち)"
        
        return "☕ 安定稼働中 (静観)"
    except: return "解析不能"

def analyze_stock(code, data):
    try:
        df = data.xs(code, axis=1, level=1).copy().dropna(subset=['Close'])
        if len(df) < 50: return None
        last_price = df['Close'].iloc[-1]
        if last_price > BUDGET_LIMIT: return None

        df['Return'] = df['Close'].pct_change()
        df['MA25_Slope'] = df['Close'].rolling(25).mean().diff(3)
        df['RSI'] = calc_rsi(df['Close'])
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        df_train = df.dropna()
        if len(df_train) < 30: return None

        X, y = df_train[['Return', 'RSI', 'MA25_Slope']], df_train['Target']
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        prob = model.predict_proba(X.iloc[-1:])[0][1]
        
        level = "対象外"
        if prob > 0.68: level = "★★★ 厳選お宝株"
        elif prob > 0.62: level = "★ お宝候補（要チェック）"
        if level == "対象外": return None

        ticker = yf.Ticker(code)
        name = ticker.info.get('shortName', code)
        return {"code": code, "name": name, "price": last_price, "prob": prob, "level": level, "rsi": df['RSI'].iloc[-1]}
    except: return None

# =============================
# メイン
# =============================

def main():
    if not os.path.exists("nikkei225.csv"): return
    df_codes = pd.read_csv("nikkei225.csv", header=None)
    codes = [str(c).zfill(4) + ".T" for c in df_codes.iloc[:, 0]]
    
    target_codes = list(set(codes + MY_PORTFOLIO))
    all_data = yf.download(target_codes, period="1y", progress=False)
    
    # 1. 前回の答え合わせ
    report_feedback = "＜前回のオススメのその後＞\n"
    if os.path.exists(SAVE_FILE):
        old_df = pd.read_csv(SAVE_FILE)
        for _, row in old_df.iterrows():
            code = row['code']
            if code in all_data.columns.get_level_values(1):
                current_p = all_data.xs(code, axis=1, level=1)['Close'].iloc[-1]
                diff = current_p - row['price']
                mark = "📈" if diff > 0 else "📉"
                report_feedback += f"・{row['name']}: {row['price']:.0f}円 → {current_p:.0f}円 ({mark} {diff:+.0f}円)\n"
    else: report_feedback += "前回データなし（明日から表示）\n"

    # 2. 今回の解析
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_stock, code, all_data) for code in codes]
        results = [f.result() for f in futures if f.result() is not None]

    results.sort(key=lambda x: x['prob'], reverse=True)
    top_hits = results[:8]
    if top_hits: pd.DataFrame(top_hits).to_csv(SAVE_FILE, index=False)

    # 3. 保有銘柄の健康診断（買い増し狙い目あり）
    portfolio_report = "＜現在の保有銘柄 健康診断＞\n"
    for code in MY_PORTFOLIO:
        status = get_stock_status(code, all_data)
        ticker = yf.Ticker(code)
        name = ticker.info.get('shortName', code)
        price = all_data.xs(code, axis=1, level=1)['Close'].iloc[-1]
        portfolio_report += f"・{name}({code}): {price:.1f}円\n  判定: {status}\n"

    # 4. メール送信
    now_jst = datetime.now() + timedelta(hours=9)
    time_str = now_jst.strftime("%Y/%m/%d %H:%M")
    body = f"【AIスキャンレポート - {time_str}】\n\n"
    body += report_feedback + "\n" + "─" * 20 + "\n\n"
    body += portfolio_report + "\n" + "─" * 20 + "\n\n"
    body += f"＜本日の厳選お宝銘柄（{BUDGET_LIMIT}円以下）＞\n"
    
    for r in top_hits:
        body += f"■ {r['name']} ({r['code']})\n判定: {r['level']} / AI確率: {r['prob']:.1%}\n価格: {r['price']:.0f}円 / RSI: {r['rsi']:.1f}\n\n"

    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = MAIL_ADDRESS, MAIL_TO, f"【S株AI予測】{time_str} 解析完了"
    msg.attach(MIMEText(body, "plain"))
    
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(MAIL_ADDRESS, MAIL_PASSWORD)
        server.send_message(msg)

if __name__ == "__main__":
    main()
