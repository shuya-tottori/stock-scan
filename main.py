import os
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
import smtplib
from email.mime.text import MIMEText


# =============================
# メール設定（GitHub Secretsから取得）
# =============================

MAIL_ADDRESS = os.getenv("MAIL_ADDRESS")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_TO = os.getenv("MAIL_TO", MAIL_ADDRESS)


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

BUDGET = 2000   # 2000円以下
results = []


# =============================
# メイン処理
# =============================

def scan():

    for code in codes:

        try:
            print(f"分析中: {code}")

            data = yf.download(
                code,
                period="6mo",
                progress=False
            )

            if data.empty:
                continue

            close = data["Close"].squeeze()

            ma25 = close.rolling(25).mean()
            rsi = RSIIndicator(close).rsi()

            price = close.iloc[-1].item()
            rsi_val = rsi.iloc[-1].item()
            ma25_val = ma25.iloc[-1].item()

            # 買い条件
            if (
                price <= BUDGET and
                price > ma25_val and
                40 <= rsi_val <= 55
            ):

                score = calc_score(price, rsi_val, ma25_val)

                results.append({
                    "code": code,
                    "price": round(price, 1),
                    "rsi": round(rsi_val, 1),
                    "score": score
                })

        except Exception as e:

            print(f"Error: {code} → {e}")
            continue


# =============================
# 期待度計算
# =============================

def calc_score(price, rsi, ma25):

    score = 0

    if rsi < 45:
        score += 2
    elif rsi < 50:
        score += 1

    if price > ma25:
        score += 2

    if price < 1000:
        score += 1

    if score >= 4:
        return "★★★ 高"
    elif score >= 2:
        return "★★☆ 中"
    else:
        return "★☆☆ 低"


# =============================
# CSV保存
# =============================

def save_csv():

    df = pd.DataFrame(results)

    path = "result_today.csv"

    df.to_csv(path, index=False, encoding="utf-8-sig")

    print("保存:", path)


# =============================
# メール送信
# =============================

def send_mail(body):

    if not MAIL_ADDRESS or not MAIL_PASSWORD:
        raise ValueError("メール認証情報が設定されていません")

    msg = MIMEText(body, "plain", "utf-8")

    msg["Subject"] = "株スキャン結果（自動）"
    msg["From"] = MAIL_ADDRESS
    msg["To"] = MAIL_TO

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:

        server.login(MAIL_ADDRESS, MAIL_PASSWORD)
        server.send_message(msg)


# =============================
# 本体
# =============================

def main():

    scan()

    save_csv()

    # ===== メール本文作成 =====

    if not results:

        body = "今日の買い候補はありません。"

    else:

        body = "【今日の買い候補】\n\n"

        for r in results:

            body += (
                f"{r['code']}  "
                f"¥{r['price']}  "
                f"RSI:{r['rsi']}  "
                f"期待度:{r['score']}\n"
            )

    send_mail(body)

    print("メール送信完了")


# =============================
# 実行
# =============================

if __name__ == "__main__":

    main()
