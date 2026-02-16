import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

# Drive上のCSV
CSV_PATH = "nikkei225.csv"

df = pd.read_csv(CSV_PATH)

codes = df.iloc[:, 0].astype(str)
codes = codes[codes.str.isdigit()]
codes = (codes + ".T").tolist()

print("銘柄数:", len(codes))


# =============================
# スキャン設定
# =============================

budget = 2000
results = []


for code in codes:
    try:
        print(f"分析中: {code}")

        data = yf.download(code, period="6mo", progress=False)

        if data.empty:
            continue

        close = data["Close"].squeeze()

        ma25 = close.rolling(25).mean()
        rsi = RSIIndicator(close).rsi()

        price = float(close.iloc[-1])
        last_rsi = float(rsi.iloc[-1])
        last_ma25 = float(ma25.iloc[-1])

        if (
            price <= budget and
            price > last_ma25 and
            40 <= last_rsi <= 55
        ):

            results.append({
                "code": code,
                "price": round(price, 1),
                "rsi": round(last_rsi, 1)
            })

  except Exception as e:
        print("=== エラー発生 ===")
        print(code)
        print(type(e))
        print(e)
        continue


# =============================
# 保存
# =============================

result_df = pd.DataFrame(results)

SAVE_PATH = "/content/drive/MyDrive/result_today.csv"
result_df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")

print("\n=== 今日の買い候補 ===")

if result_df.empty:
    print("該当なし")
else:
    display(result_df)

print("\n保存先:", SAVE_PATH)

import smtplib
from email.mime.text import MIMEText

# ===== 設定 =====
GMAIL = "shuya.tottori@gmail.com"
PASS = "aybhsvrafilebveo"

# ===== 送信関数 =====
def send_mail(msg):

    body = MIMEText(msg, "plain", "utf-8")
    body["Subject"] = "株スキャン結果"
    body["From"] = GMAIL
    body["To"] = GMAIL

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL, PASS)
        server.send_message(body)


# ===== 結果作成 =====
if not results:
    text = "今日の買い候補：なし"
else:
    text = "今日の買い候補\n\n"
    for r in results:
        text += f"{r['code']}  ¥{r['price']}  RSI:{r['rsi']}\n"

# ===== 送信 =====
send_mail(text)
