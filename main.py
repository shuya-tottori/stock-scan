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
MAIL_TO = os.getenv("MAIL_TO", MAIL_ADDRESS)
BUDGET_LIMIT = 2000 
SAVE_FILE = "recommended.csv"

# =============================
# 解析ロジック
# =============================

def analyze_stock(code, data):
    try:
        df = data.xs(code, axis=1, level=1).copy()
        df.dropna(subset=['Close'], inplace=True)
        if len(df) < 50: return None

        last_price = df['Close'].iloc[-1]
        if last_price > BUDGET_LIMIT: return None

        # 特徴量作成
        df['Return'] = df['Close'].pct_change()
        df['MA25_Slope'] = df['Close'].rolling(25).mean().diff(3)
        df['RSI'] = (lambda s, p=14: 100 - (100 / (1 + (s.diff().where(s.diff() > 0, 0).rolling(p).mean() / s.diff().where(s.diff() < 0, 0).abs().rolling(p).mean()).replace(0, np.nan))))(df['Close'])
        
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

        return {"code": code, "name": name, "price": last_price, "prob": prob, "level": level}
    except: return None

# =============================
# メイン処理
# =============================

def main():
    df_codes = pd.read_csv("nikkei225.csv", header=None)
    codes = [str(c).zfill(4) + ".T" for c in df_codes.iloc[:, 0]]
    
    print("データ取得中...")
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
            report_feedback += f"データ復元失敗: {e}\n"
    else:
        report_feedback += "初回実行のため前回のデータはありません。\n"

    # --- 2. 今回の解析 ---
    print("解析中...")
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_stock, code, all_data) for code in codes]
        results = [f.result() for f in futures if f.result() is not None]

    results.sort(key=lambda x: x['prob'], reverse=True)
    top_hits = results[:8]

    # --- 3. 今回の結果を保存（次回用） ---
    if top_hits:
        pd.DataFrame(top_hits).to_csv(SAVE_FILE, index=False)

    # --- 4. メール送信 ---
    now_jst = datetime.now() + timedelta(hours=9)
    time_str = now_jst.strftime("%Y/%m/%d %H:%M")
    
    body = f"【AIスキャン結果レポート - {time_str}】\n\n"
    body += report_feedback + "\n" + "─" * 20 + "\n\n"
    body += "＜本日の厳選お小遣い銘柄（2000円以下）＞\n"
    
    if top_hits:
        for r in top_hits:
            body += f"■ {r['name']} ({r['code']})\n判定: {r['level']} / AI確率: {r['prob']:.1%}\n価格: {r['price']:.0f}円\n\n"
    else:
        body += "該当なし\n"

    msg = MIMEMultipart()
    msg["From"], msg["To"] = MAIL_ADDRESS, MAIL_TO
    msg["Subject"] = f"【S株AI予測】答え合わせ付き {time_str}"
    msg.attach(MIMEText(body, "plain"))
    
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(MAIL_ADDRESS, MAIL_PASSWORD)
        server.send_message(msg)
    print("完了！")

if __name__ == "__main__":
    main()
