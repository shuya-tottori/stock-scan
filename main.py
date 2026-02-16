import yfinance as yf
import pandas as pd
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


# ===== 環境変数（Secretsから取得）=====
GMAIL = os.getenv("GMAIL_USER")
PASS = os.getenv("GMAIL_PASS")
TO = os.getenv("MAIL_TO", GMAIL)


# ===== 銘柄リスト =====
CODES = [
    "4502.T", "7203.T", "9432.T", "9984.T", "8306.T",
    "6758.T", "5401.T", "2502.T", "9501.T"
]


# ===== RSI計算 =====
def calc_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ===== 株分析 =====
def analyze():

    results = []

    print("銘柄数:", len(CODES))

    for code in CODES:

        try:
            print("分析中:", code)

            df = yf.download(code, period="3mo", progress=False)

            if df.empty:
                continue

            close = df["Close"]

            rsi = calc_rsi(close)

            latest_price = round(close.iloc[-1], 1)
            latest_rsi = round(rsi.iloc[-1], 1)

            # 期待度判定（例）
            if latest_rsi < 30:
                level = "★★★★★（買い候補）"
            elif latest_rsi < 40:
                level = "★★★★☆"
            elif latest_rsi < 60:
                level = "★★★☆☆"
            elif latest_rsi < 70:
                level = "★★☆☆☆"
            else:
                level = "★☆☆☆☆（過熱）"

            results.append([
                code,
                latest_price,
                latest_rsi,
                level
            ])

        except Exception as e:
            print("Error:", code, e)

    df = pd.DataFrame(
        results,
        columns=["code", "price", "rsi", "expect"]
    )

    return df


# ===== メール送信 =====
def send_mail(text):

    msg = MIMEMultipart()

    msg["From"] = GMAIL
    msg["To"] = TO
    msg["Subject"] = "株スキャン結果 " + datetime.now().strftime("%Y-%m-%d")

    msg.attach(MIMEText(text, "plain", "utf-8"))

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()

    server.login(GMAIL, PASS)

    server.send_message(msg)

    server.quit()


# ===== メイン処理 =====
def main():

    df = analyze()

    if df.empty:
        body = "該当銘柄なし"
    else:
        body = df.to_string(index=False)

    # CSV保存
    df.to_csv("result_today.csv", index=False, encoding="utf-8-sig")

    print("保存: result_today.csv")

    send_mail(body)


if __name__ == "__main__":
    main()
