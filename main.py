import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
import smtplib
from email.mime.text import MIMEText


# =============================
# CSV読み込み
# =============================

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

        data = yf.download(
            code,
            period="6mo",
            progress=False
        )

        if data.empty:
            print("  → データなし")
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
        print("=== エラー ===", code, e)
        continue


# =============================
# 保存
# =============================

result_df = pd.DataFrame(results)

SAVE_PATH = "result_today.csv"
result_df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")

print("保存:", SAVE_PATH)
print(result_df)


# =============================
# メール送信
# =============================

GMAIL = "shuya.tottori@gmail.com"
PASS = "アプリパスワード"


def send_mail(msg):

    body = MIMEText(msg, "plain", "utf-8")
    body["Subject"] = "株スキャン結果"
    body["From"] = GMAIL
    body["To"] = GMAIL

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL, PASS)
        server.send_message(body)


# =============================
# 本文作成
# =============================

if not results:
    text = "今日の買い候補：なし"
else:
    text = "今日の買い候補\n\n"
    for r in results:
        text += f"{r['code']}  ¥{r['price']}  RSI:{r['rsi']}\n"


send_mail(text)
