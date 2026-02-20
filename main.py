import os
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier


# =============================
# 環境変数（GitHub Secrets）
# =============================

MAIL_ADDRESS = os.getenv("MAIL_ADDRESS")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_TO = os.getenv("MAIL_TO", MAIL_ADDRESS)


# =============================
# 日経225コード読み込み（列名なし）
# =============================

def load_codes_from_csv():
    df = pd.read_csv("nikkei225.csv", header=None)

    codes = (
        df.iloc[:, 0]
        .astype(str)
        .str.zfill(4)
        + ".T"
    )

    return codes.tolist()


# =============================
# RSI計算
# =============================

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# =============================
# yfinance列正規化（MultiIndex対策）
# =============================

def normalize_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# =============================
# 銘柄名取得
# =============================

def get_stock_name(code):
    try:
        info = yf.Ticker(code).info
        return info.get("shortName", "不明")
    except Exception:
        return "不明"


# =============================
# AIモデル
# =============================

def train_ai(df):

    df = normalize_columns(df)

    if "Close" not in df.columns or "Volume" not in df.columns:
        return None, None, None

    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].iloc[:, 0]

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA25"] = df["Close"].rolling(25).mean()
    df["RSI"] = calc_rsi(df["Close"])

    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    df = df.dropna()

    if len(df) < 50:
        return None, None, None

    X = df[["Close", "MA5", "MA25", "RSI", "Volume"]]
    y = df["Target"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

    latest_row = X.iloc[-1:]
    prob = model.predict_proba(latest_row)[0][1]

    return model, latest_row, prob


# =============================
# 判定ロジック
# =============================

def judge_level(rsi, ma5, ma25, ai_prob):

    if rsi < 30 and ma5 > ma25 and ai_prob > 0.7:
        return "★★★（強い買い）"

    elif rsi < 35 and ai_prob > 0.6:
        return "★★（買い）"

    elif rsi < 40 and ai_prob > 0.55:
        return "★（弱い買い）"

    elif rsi < 50 and ai_prob > 0.5:
        return "△（様子見候補）"

    else:
        return "対象外"


# =============================
# メイン処理
# =============================

def main():

    strong_list = []
    watch_list = []

    CODES = load_codes_from_csv()
    print(f"監視銘柄数: {len(CODES)}")

    for code in CODES:

        try:
            df = yf.download(
                code,
                period="6mo",
                progress=False,
                auto_adjust=True
            )
        except Exception:
            continue

        if df is None or df.empty:
            continue

        model, latest_row, ai_prob = train_ai(df)

        if model is None:
            continue

        df = normalize_columns(df)

        rsi = calc_rsi(df["Close"]).iloc[-1]
        ma5 = df["Close"].rolling(5).mean().iloc[-1]
        ma25 = df["Close"].rolling(25).mean().iloc[-1]

        level = judge_level(rsi, ma5, ma25, ai_prob)

        if level == "対象外":
            continue

        name = get_stock_name(code)

        stock_text = (
            f"■ {name} ({code})\n"
            f"現在値: {df['Close'].iloc[-1]:.0f}円\n"
            f"RSI: {rsi:.1f}\n"
            f"AI上昇確率: {ai_prob:.2%}\n"
            f"判定: {level}\n\n"
        )

        if "★" in level:
            strong_list.append(stock_text)
        elif "△" in level:
            watch_list.append(stock_text)

    # =============================
    # メール作成
    # =============================

    today = datetime.now()
    date_str = today.strftime("%Y年%m月%d日")

    body = f"【{date_str} 銘柄レポート】\n\n"

    if today.day == 15:
        body += "【お知らせ】\n"
        body += "月に一度はGitHub Actionsを手動実行してください。\n"
        body += "（自動停止防止のため）\n\n"

    if strong_list:
        body += "■ 本日の注目銘柄\n\n"
        body += "".join(strong_list)

    if watch_list:
        body += "■ 様子見候補（参考）\n\n"
        body += "".join(watch_list)

    if not strong_list and not watch_list:
        body += "本日は該当なし\n"

    msg = MIMEMultipart()
    msg["From"] = MAIL_ADDRESS
    msg["To"] = MAIL_TO
    msg["Subject"] = f"{date_str} 株式スキャン結果"

    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(MAIL_ADDRESS, MAIL_PASSWORD)
        server.send_message(msg)


if __name__ == "__main__":
    main()
