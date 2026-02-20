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
MAIL_TO = os.getenv("MAIL_TO") if os.getenv("MAIL_TO") else MAIL_ADDRESS

BUDGET_LIMIT = 2000 
SAVE_FILE = "recommended.csv"

# 【保有銘柄】
MY_PORTFOLIO = ["9432.T", "8001.T", "8031.T", "8316.T", "1605.T", "4503.T", "8697.T", "8766.T"]

# =============================
# 外部指標取得
# =============================
def get_external_factors():
    try:
        # S&P500とドル円
        data = yf.download(["JPY=X", "^GSPC"], period="5d", progress=False)['Close']
        us_stock_change = (data["^GSPC"].iloc[-1] / data["^GSPC"].iloc[-2]) - 1
        usd_jpy_rate = data["JPY=X"].iloc[-1]
        usd_jpy_change = (data["JPY=X"].iloc[-1] / data["JPY=X"].iloc[-2]) - 1
        return us_stock_change, usd_jpy_rate, usd_jpy_change
    except:
        return 0.0, 150.0, 0.0

# =============================
# 解析ロジック
# =============================
def analyze_stock(code, data, ext_factors):
    try:
        # データの切り出しとクレンジング
        df = data.xs(code, axis=1, level=1).copy().dropna(subset=['Close'])
        if len(df) < 40: return None
        
        last_price = df['Close'].iloc[-1]
        
        # 特徴量作成
        df['Return'] = df['Close'].pct_change()
        # RSI自作
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['US_Stock_Effect'] = ext_factors[0]
        df['USD_JPY_Effect'] = ext_factors[2]
        
        # 学習（直近30日の動きから明日を予測）
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df_train = df.dropna()
        if len(df_train) < 20: return None

        features = ['Return', 'RSI', 'US_Stock_Effect', 'USD_JPY_Effect']
        X = df_train[features]
        y = df_train['Target']
        
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
        prob = model.predict_proba(X.iloc[-1:])[0][1]
        
        # 判定しきい値を少し下げて「該当なし」を防ぐ
        level = "対象外"
        if prob > 0.65: level = "🔥 超お宝株(激アツ)"
        elif prob > 0.58: level = "★★★ 厳選お宝株"
        elif prob > 0.52: level = "★ お宝候補"

        # 保有銘柄は「対象外」でも結果に残す
        if level == "対象外" and code not in MY_PORTFOLIO:
            return None

        ticker_name = code # フォールバック用
        return {
            "code": code, 
            "name": ticker_name, 
            "price": last_price, 
            "prob": prob, 
            "level": level, 
            "rsi": df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
        }
    except Exception as e:
        return None

# =============================
# メイン
# =============================
def main():
    print("--- グローバルAI解析開始 ---")
    
    # 銘柄リスト読み込み
    if not os.path.exists("nikkei225.csv"):
        print("CSV missing")
        return
    df_codes = pd.read_csv("nikkei225.csv", header=None)
    base_codes = [str(c).zfill(4) + ".T" for c in df_codes.iloc[:, 0]]
    
    ext_factors = get_external_factors()
    all_target_codes = list(set(base_codes + MY_PORTFOLIO))
    
    print(f"データ取得中... {len(all_target_codes)}銘柄")
    all_data = yf.download(all_target_codes, period="1y", progress=False)
    
    # 銘柄名の取得（一括だと重いので解析時に個別取得かコード表示）
    # 解析実行
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_stock, code, all_data, ext_factors) for code in all_target_codes]
        results = [f.result() for f in futures if f.result() is not None]

    # 自信度ランク
    us_change = ext_factors[0]
    market_status = "強気" if us_change > 0.005 else ("弱気" if us_change < -0.005 else "慎重")
    market_comment = f"【AI自信度ランク：{market_status}】\n米国株の影響：{'上昇📈' if us_change > 0 else '下落📉'} ({us_change:.2%})\nドル円：{ext_factors[1]:.2f}円\n"

    # 1. 前回の答え合わせ
    report_feedback = "＜前回の答え合わせ＞\n"
    if os.path.exists(SAVE_FILE):
        try:
            old_df = pd.read_csv(SAVE_FILE)
            for _, row in old_df.iterrows():
                code = row['code']
                if code in all_data.columns.get_level_values(1):
                    cur = all_data.xs(code, axis=1, level=1)['Close'].iloc[-1]
                    diff = cur - row['price']
                    report_feedback += f"・{code}: {row['price']:.0f}→{cur:.0f} ({'📈' if diff>0 else '📉'} {diff:+.0f})\n"
        except: report_feedback += "データ読み込みエラー\n"
    else: report_feedback += "初回実行のため明日から表示されます\n"

    # 2. 保有銘柄診断
    portfolio_report = "＜保有銘柄 健康診断＞\n"
    for code in MY_PORTFOLIO:
        res = next((r for r in results if r['code'] == code), None)
        if res:
            status = "✨ 買い増し狙い目！" if res['rsi'] < 45 else ("🚀 絶好調" if res['rsi'] > 65 else "☕ 安定")
            portfolio_report += f"・{code}: {res['price']:.0f}円 ({status})\n"
        else:
            portfolio_report += f"・{code}: データ取得待ち\n"

    # 3. 厳選銘柄の抽出（予算内のみ）
    recommendations = [r for r in results if r['code'] not in MY_PORTFOLIO and r['level'] != "対象外" and r['price'] <= BUDGET_LIMIT]
    recommendations.sort(key=lambda x: x['prob'], reverse=True)
    top_hits = recommendations[:10]
    
    # 次回のために保存
    if top_hits:
        pd.DataFrame(top_hits).to_csv(SAVE_FILE, index=False)

    # 4. メール送信
    now = datetime.now() + timedelta(hours=9)
    body = f"【AIグローバルレポート - {now.strftime('%Y/%m/%d %H:%M')}】\n\n"
    body += market_comment + "\n" + report_feedback + "\n" + "─"*20 + "\n\n"
    body += portfolio_report + "\n" + "─"*20 + "\n\n"
    body += "＜本日の厳選お宝銘柄（2000円以下）＞\n"
    
    if top_hits:
        for r in top_hits:
            body += f"■ {r['code']}\n判定: {r['level']} (AI確率:{r['prob']:.1%})\n価格: {r['price']:.0f}円 / RSI: {r['rsi']:.1f}\n\n"
    else:
        body += "現在、AIの基準を満たす銘柄はありません。慎重な相場です。☕\n"

    msg = MIMEMultipart()
    msg["Subject"] = f"【AI予測】自信度:{market_status} {now.strftime('%H:%M')}"
    msg["From"], msg["To"] = MAIL_ADDRESS, MAIL_TO
    msg.attach(MIMEText(body, "plain"))
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(MAIL_ADDRESS, MAIL_PASSWORD)
            server.send_message(msg)
        print("Mail sent successfully")
    except Exception as e:
        print(f"Mail failed: {e}")

if __name__ == "__main__":
    main()
