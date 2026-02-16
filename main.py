import os
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from sklearn.ensemble import RandomForestClassifier


# =============================
# 環境変数（GitHub Secrets用）
# =============================

MAIL_ADDRESS = os.getenv("MAIL_ADDRESS")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_TO = os.getenv("MAIL_TO", MAIL_ADDRESS)


# =============================
# 銘柄リスト
# =============================

CODES = [
    "7203.T",
    "6758.T",
    "9432.T",
    "9984.T",
    "8306.T",
    "4502.T",
    "4503.T",
    "9501.T",
]


# =============================
# 銘柄名取得
# =============================

def get_stock_name(code):

    try:
        info = yf.Ticker(code).info
        return info.get("shortName", "不明")

    except:
        return "不明"


# =============================
# RSI計算
# =============================

def calc_rsi(series, period=14):

    delta = series.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


# =============================
# AIモデル作成
# =============================

def train_ai(df):

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA25"] = df["Close"].rolling(25).mean()
    df["RSI"] = calc_rsi(df["Close"])

    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    df = df.dropna()

    X = df[["Close", "MA5", "MA25", "RSI", "Volume"]]
    y = df["Target"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

    return model, X.iloc[-1:]


# =============================
# 買いレベル判定
# =============================

def judge_level(rsi, ma5, ma25, ai_prob):

    if rsi < 30 and ma5 > ma25 and ai_prob > 0.7:
        return "★★★（強い買い）"

    elif rsi < 35 and ai_prob > 0.6:
        return "★★（買い）"

    elif rsi < 40 and ai_prob > 0.55:
        return "★（弱い買い）"

    else:
        return "対象外"


# =============================
# メール送信
# =============================

def send_mail(body):

    msg = MIMEMultipart()
    msg["From"] = MAIL_ADDRESS
    msg["To"] = MAIL_TO
    from datetime import datetime

today = datetime.now().strftime("%-m/%-d")  # 例: 2/16（Linux用）
msg["Subject"] = f"[{today}]_リサーチ結果通知" 

    msg.attach(MIMEText(body, "plain", "utf-8"))

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(MAIL_ADDRESS, MAIL_PASSWORD)

    server.send_message(msg)

    server.quit()


# =============================
# メイン処理
# =============================

def main():

    print(f"銘柄数: {len(CODES)}")

    body = "【本日の注目銘柄（AI分析）】\n\n"

    results = []

    for code in CODES:

        print("分析中:", code)

        try:

            df = yf.download(
                code,
                period="2y",
                interval="1d",
                progress=False
            )

            if len(df) < 200:
                continue

            close = df["Close"]
            volume = df["Volume"]

            rsi = calc_rsi(close)

            price = close.iloc[-1].item()
            rsi_val = rsi.iloc[-1].item()

            ma5 = close.rolling(5).mean().iloc[-1]
            ma25 = close.rolling(25).mean().iloc[-1]

            vol_now = volume.iloc[-1]
            vol_avg = volume.rolling(20).mean().iloc[-1]

            volume_ok = vol_now > vol_avg
            trend_up = ma5 > ma25

            # ===== AI予測 =====
            model, latest = train_ai(df)

            prob = model.predict_proba(latest)[0][1]

            # ===== レベル =====
            level = judge_level(rsi_val, ma5, ma25, prob)

            if rsi_val < 45 and trend_up and volume_ok:

                name = get_stock_name(code)

                body += f"""
■ {name} ({code})
株価：{round(price,1)}円
RSI：{round(rsi_val,1)}
出来高：{'↑' if volume_ok else '-'}
AI上昇確率：{round(prob*100,1)}%
レベル：{level}

"""

                results.append([
                    code,
                    name,
                    price,
                    rsi_val,
                    prob,
                    level
                ])

        except Exception as e:

            print("Error:", code, e)


    # CSV保存
    if results:

        df_out = pd.DataFrame(
            results,
            columns=[
                "Code",
                "Name",
                "Price",
                "RSI",
                "AI_Prob",
                "Level"
            ]
        )

        df_out.to_csv("result_today.csv", index=False)

        print("保存: result_today.csv")


    if len(results) == 0:
        body += "本日は該当銘柄なし\n"


    send_mail(body)


# =============================
# 実行
# =============================

if __name__ == "__main__":

    main()
