import yfinance as yf
import pandas as pd
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


GMAIL = os.getenv("GMAIL_USER")
PASS = os.getenv("GMAIL_PASS")
TO = os.getenv("MAIL_TO", GMAIL)


CODES = [
    "4502.T", "7203.T", "9432.T", "9984.T", "8306.T",
    "6758.T", "5401.T", "2502.T", "9501.T"
]


def calc_rsi(close, period=14):

    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi.astype(float)


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

            price = float(close.iloc[-1])
            rsi_val = float(rsi.iloc[-1])

            price = round(price, 1)
            rsi_val = round(rsi_val, 1)

            if rsi_val < 30:
                level = "★★★★★"
            elif rsi_val < 40:
                level = "★★★★☆"
            elif rsi_val < 60:
                level = "★★★☆☆"
            elif rsi_val < 70:
                level = "★★☆☆☆"
            else:
                level = "★☆☆☆☆"

            results.append([
                code,
                price,
                rsi_val,
                level
            ])

        except Exception as e:
            print("Error:", code, e)

    return pd.DataFrame(
        results,
        columns=["code", "price", "rsi", "expect"]
    )

def send_mail(msg):

    if not GMAIL or not PASS:
        raise ValueError("GMAIL_USER / GMAIL_PASS が未設定です")

    body = MIMEText(msg, "plain", "utf-8")
    body["Subject"] = "株スキャン結果"
    body["From"] = GMAIL
    body["To"] = TO

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL, PASS)
        server.send_message(body)




def main():

    df = analyze()

    if df.empty:
        body = "該当銘柄なし"
    else:
        body = df.to_string(index=False)

    df.to_csv("result_today.csv", index=False, encoding="utf-8-sig")

    print("保存: result_today.csv")

    send_mail(body)


if __name__ == "__main__":
    main()
