import os
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ThreadPoolExecutor

# =============================
# 設定
# =============================
MAIL_ADDRESS = os.getenv("MAIL_ADDRESS")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_TO = os.getenv("MAIL_TO", MAIL_ADDRESS)

# =============================
# 補助関数
# =============================

def load_codes():
    df = pd.read_csv("nikkei225.csv", header=None)
    return [str(c).zfill(4) + ".T" for c in df.iloc[:, 0]]

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# =============================
# AI学習 & 判定ロジック
# =============================

def analyze_stock(code, data):
    """
    1銘柄ずつの学習と判定を行う関数（並列実行用）
    """
    try:
        # MultiIndexからのデータ抽出
        df = data.xs(code, axis=1, level=1).copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
        df.dropna(subset=['Close'], inplace=True)
        
        if len(df) < 50: return None

        # --- 特徴量エンジニアリング ---
        df['Return'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA25'] = df['Close'].rolling(25).mean()
        df['Diff_MA5'] = (df['Close'] - df['MA5']) / df['MA5'] # 移動平均乖離率
        df['RSI'] = calc_rsi(df['Close'])
        df['Vol_Change'] = df['Volume'].pct_change()
        
        # 目的変数：翌日の終値がプラスか
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        df_train = df.dropna()
        if len(df_train) < 30: return None

        # 特徴量選択
        features = ['Return', 'Diff_MA5', 'RSI', 'Vol_Change']
        X = df_train[features]
        y = df_train['Target']

        # 学習
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)

        # 最新予測
        latest_row = X.iloc[-1:]
        prob = model.predict_proba(latest_row)[0][1]
        
        # 判定
        last_rsi = df['RSI'].iloc[-1]
        last_close = df['Close'].iloc[-1]
        level = "対象外"
        
        if last_rsi < 30 and prob > 0.7: level = "★★★（激アツ）"
        elif last_rsi < 35 and prob > 0.6: level = "★★（買い）"
        elif last_rsi < 45 and prob > 0.55: level = "★（弱気買い）"
        elif prob > 0.65: level = "△（AI注目）"

        if level == "対象外": return None

        return {
            "code": code,
            "price": last_close,
            "rsi": last_rsi,
            "prob": prob,
            "level": level
        }
    except Exception:
        return None

# =============================
# メイン処理
# =============================

def main():
    codes = load_codes()
    print(f"データ取得中... ({len(codes)}銘柄)")
    
    # 1. 一括ダウンロード（これが一番速い）
    all_data = yf.download(codes, period="1y", interval="1d", progress=False)

    print("AI解析実行中...")
    results = []
    
    # 2. 並列処理で全銘柄を一気に解析
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(analyze_stock, code, all_data) for code in codes]
        for f in futures:
            res = f.result()
            if res: results.append(res)

    # 3. メール本文作成
    date_str = datetime.now().strftime("%Y/%m/%d")
    body = f"【{date_str} 株式解析レポート】\n\n"
    
    # レベル順にソート
    results.sort(key=lambda x: x['prob'], reverse=True)
    
    if not results:
        body += "本日の注目銘柄はありません。"
    else:
        for r in results:
            body += f"■ {r['code']}\n"
            body += f"判定: {r['level']}\n"
            body += f"株価: {r['price']:.0f}円 / RSI: {r['rsi']:.1f} / 上昇確率: {r['prob']:.1%}\n\n"

    # 4. 送信
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = MAIL_ADDRESS, MAIL_TO, f"【AI予測】{date_str} スキャン結果"
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(MAIL_ADDRESS, MAIL_PASSWORD)
            server.send_message(msg)
        print("メール送信完了！")
    except Exception as e:
        print(f"メール送信失敗: {e}")

if __name__ == "__main__":
    main()
