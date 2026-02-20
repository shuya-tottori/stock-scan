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
# 宛先が空の場合は自分のアドレスを使う
MAIL_TO = os.getenv("MAIL_TO") if os.getenv("MAIL_TO") else MAIL_ADDRESS

BUDGET_LIMIT = 2000 
SAVE_FILE = "recommended.csv"

# =============================
# 補助関数
# =============================

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def analyze_stock(code, data):
    try:
        df = data.xs(code, axis=1, level=1).copy()
        df.dropna(subset=['Close'], inplace=True)
        if len(df) < 50: return None

        last_price = df['Close'].iloc[-1]
        # 予算フィルター
        if last_price > BUDGET_LIMIT: return None

        # 特徴量作成
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
        if prob > 0.68: level = "★注目株"
        elif prob > 0.62: level = "次点"

        if level == "対象外": return None

        ticker = yf.Ticker(code)
        name = ticker.info.get('shortName', code)

        return {"code": code, "name": name, "price": last_price, "prob": prob, "level": level, "rsi": df['RSI'].iloc[-1]}
    except: return None

# =============================
# メイン
# =============================

def main():
    print("--- 解析開始 ---")
    if not os.path.exists("nikkei225.csv"):
        print("Error: nikkei225.csv missing")
        return

    df_codes = pd.read_csv("nikkei225.csv", header=None)
    codes = [str(c).zfill(4) + ".T" for c in df_codes.iloc[:, 0]]
    
    print(f"データ取得中... ({len(codes)}銘柄)")
    all_data = yf.download(codes, period="1y", progress=False)
    
    # --- 1. 前回の答え合わせ ---
    report_feedback = "＜前回のオススメのその後＞\n"
    if os.path.exists(SAVE_FILE):
        try:
            old_df = pd.read_csv(SAVE_FILE)
            for _, row in old_df.iterrows():
                code = row['code']
                if code in all_data.columns.get_level_values(1):
                    current_p = all_data.xs(code, axis=1, level=1)['Close'].iloc[-1]
                    old_p = row['price']
                    diff = current_p - old_p
                    mark = "📈" if diff > 0 else "📉"
                    report_feedback += f"・{row['name']}: {old_p:.0f}円 → {current_p:.0f}円 ({mark} {diff:+.0f}円)\n"
        except Exception as e:
            report_feedback += f"答え合わせ失敗: {e}\n"
    else:
        report_feedback += "初回または前回データなし（明日から表示されます）\n"

    # --- 2. 今回の解析 ---
    print("AI解析中...")
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_stock, code, all_data) for code in codes]
        results = [f.result() for f in futures if f.result() is not None]

    results.sort(key=lambda x: x['prob'], reverse=True)
    top_hits = results[:8]

    # --- 3. 今回の結果を保存 ---
    if top_hits:
        pd.DataFrame(top_hits).to_csv(SAVE_FILE, index=False)

    # --- 4. メール作成・送信 ---
    now_jst = datetime.now() + timedelta(hours=9)
    time_str = now_jst.strftime("%Y/%m/%d %H:%M")
    
    # 宛先チェック
    if not MAIL_TO or "@" not in str(MAIL_TO):
        print(f"送信中止: 宛先が無効です (MAIL_TO: {MAIL_TO})")
        return

    body = f"【AIスキャン結果レポート - {time_str}】\n\n"
    body += report_feedback + "\n" + "─" * 20 + "\n\n"
    body += f"＜本日の厳選銘柄（{BUDGET_LIMIT}円以下）＞\n"
    
    if top_hits:
        for r in top_hits:
            body += f"■ {r['name']} ({r['code']})\n判定: {r['level']} / AI確率: {r['prob']:.1%}\n価格: {r['price']:.0f}円 / RSI: {r['rsi']:.1f}\n\n"
    else:
        body += "該当なし\n"

    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"] = MAIL_ADDRESS, MAIL_TO
        msg["Subject"] = f"【S株AI予測】{time_str} 解析結果"
        msg.attach(MIMEText(body, "plain"))
        
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(MAIL_ADDRESS, MAIL_PASSWORD)
            server.send_message(msg)
        print(f"メール送信完了！宛先: {MAIL_TO}")
    except Exception as e:
        print(f"メール送信エラー: {e}")

if __name__ == "__main__":
    main()
