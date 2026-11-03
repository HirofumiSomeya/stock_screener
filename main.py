from google.colab import files
import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
from tqdm import tqdm
import time
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# 1. ダウンロードしたJPX銘柄一覧ファイルのアップロード
file_name = 'data_j.xlsx'

# 2. ファイルの読み込み
if file_name.endswith('.csv'):
    df_jpx = pd.read_csv(file_name)
else:
    df_jpx = pd.read_excel(file_name)

# 3. 普通株式（企業）のみを抽出
df_stocks = df_jpx[df_jpx['市場・商品区分'].str.contains('プライム|スタンダード|グロース', na=False)]

# 4. 証券コードをリスト化
valid_tickers = [str(code).strip() + ".T" for code in df_stocks['コード']]

print(f"抽出完了: 全上場企業 {len(valid_tickers)} 銘柄をスクリーニング対象に設定しました。")

print(f"\n第1段階: 株価3000円未満、かつ上昇トレンドの銘柄をスクリーニング中...")
uptrend_data = []
latest_features = []
passed_tickers = []

chunk_size = 50
for i in tqdm(range(0, len(valid_tickers), chunk_size)):
    chunk = valid_tickers[i:i + chunk_size]
    
    try:
        bulk_data = yf.download(chunk, period="2y", interval="1d", progress=False, group_by='ticker')
        
        for ticker in chunk:
            if len(chunk) == 1:
                df = bulk_data.copy()
            else:
                if ticker not in bulk_data.columns.levels[0]: continue
                df = bulk_data[ticker].copy()
                
            df = df.dropna(subset=['Close'])
            if df.empty or len(df) < 100: continue
            
            # --- 3,000円未満 フィルター ---
            if df['Close'].iloc[-1] >= 3000:
                continue
                
            # --- 上昇トレンド フィルター ---
            df['SMA25'] = df['Close'].rolling(window=25).mean()
            df['SMA75'] = df['Close'].rolling(window=75).mean()

            latest = df.iloc[-1]
            if pd.isna(latest['SMA75']):
                continue
            
            is_uptrend = (latest['Close'] > latest['SMA25']) and (latest['Close'] > latest['SMA75'])
                         
            if not is_uptrend:
                continue
                
            # 特徴量の計算
            df['Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Return'].rolling(window=10).std()
            df['Volume_Change'] = df['Volume'].pct_change()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # RSIの計算
            df['RSI'] = 100 - (100 / (1 + gain / loss))
            
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
            if df.empty: continue
                
            df['Ticker'] = ticker
            uptrend_data.append(df[:-1])
            latest_features.append(df.iloc[-1:])
            passed_tickers.append(ticker)
            
    except Exception as e:
        time.sleep(2)
        continue

print(f"\nスクリーニング通過: {len(passed_tickers)} 社")

if not uptrend_data:
    print("条件を満たす銘柄がありませんでした。日経平均などの地合いを確認してください。")
else:
    print("\n第2段階: 通過した有望銘柄のファンダメンタル情報を取得しています...")
    fund_data = []
    for ticker in tqdm(passed_tickers):
        try:
            info = yf.Ticker(ticker).info
            fund_data.append({
                'Ticker': ticker,
                'PER': info.get('trailingPE', np.nan),
                'PBR': info.get('priceToBook', np.nan)
            })
        except:
            pass
            
    fund_df = pd.DataFrame(fund_data)
    
    combined_train = pd.concat(uptrend_data).merge(fund_df, on='Ticker', how='left')
    combined_latest = pd.concat(latest_features).merge(fund_df, on='Ticker', how='left')
    
    combined_train = combined_train.fillna(combined_train.mean(numeric_only=True))
    combined_latest = combined_latest.fillna(combined_train.mean(numeric_only=True))

    features = ['Return', 'Volatility', 'Volume_Change', 'RSI', 'PER', 'PBR']
    
    X_train, y_train = combined_train[features], combined_train['Target']
    
    print("\n第3段階: AI学習と明日の勝率ランキング生成中...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.01, max_depth=5, 
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    X_latest = combined_latest[features]
    combined_latest['AI_Win_Probability'] = lgb_model.predict_proba(X_latest)[:, 1]
    
    ranking = combined_latest[['Ticker', 'Close', 'AI_Win_Probability', 'PER', 'PBR']].copy()
    ranking = ranking.sort_values(by='AI_Win_Probability', ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    
    # データの算出（価格は整数に丸めてスッキリさせる）
    ranking['Target_Buy'] = ranking['Close'].round(0).astype(int)
    ranking['Take_Profit'] = (ranking['Close'] * 1.05).round(0).astype(int)
    ranking['Stop_Loss'] = (ranking['Close'] * 0.98).round(0).astype(int)
    
    # 表示用の文字列加工
    ranking['AI_Win_Probability_Str'] = (ranking['AI_Win_Probability'] * 100).round(1).astype(str) + '%'
    ranking['Close_Str'] = ranking['Close'].round(0).astype(int).apply(lambda x: f"{x:,}円")
    ranking['Buy_Str'] = ranking['Target_Buy'].apply(lambda x: f"{x:,}円")
    ranking['TP_Str'] = ranking['Take_Profit'].apply(lambda x: f"{x:,}円 (+5%)")
    ranking['SL_Str'] = ranking['Stop_Loss'].apply(lambda x: f"{x:,}円 (-2%)")
    ranking['PER_Str'] = ranking['PER'].round(1)
    ranking['PBR_Str'] = ranking['PBR'].round(2)

    print("\n" + "="*85)
    print("🔥 【AI厳選】3,000円未満・上昇有望銘柄トレードプラン (上位15選)")
    print("="*85)
    print(f"{'順位':<4} {'銘柄コード':<10} {'現在値':<10} {'AI勝率':<8} {'PER':<6} {'PBR':<6} {'利確目標(+5%)':<14} {'損切り(-2%)':<14}")
    print("-"*85)

    for idx, row in ranking.head(15).iterrows():
        print(f"{idx:<4} {row['Ticker']:<10} {row['Close_Str']:<10} {row['AI_Win_Probability_Str']:<8} "
              f"{row['PER_Str']:<6} {row['PBR_Str']:<6} {row['TP_Str']:<14} {row['SL_Str']:<14}")
    print("="*85)

# --- 【機能追加】予測結果の蓄積と検証システム ---
history_file = 'ai_prediction_history.csv'
today_str = datetime.now().strftime('%Y-%m-%d')

# 1. 本日のランキング上位15銘柄のデータを抽出してログ用データフレームを作成
today_log = ranking.head(15).copy()
today_log['Date'] = today_str
today_log['Actual_Close_Next'] = np.nan # 翌日終値（後から埋まる）
today_log['Is_Hit'] = np.nan            # 的中判定（後から埋まる）

# 2. 過去の履歴ファイルが存在する場合、過去データの「答え合わせ」を試みる
if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)
    
    # まだ答え合わせができていない（ActualがNaNの）過去レコードをチェック
    mask = history_df['Actual_Close_Next'].isna()
    if mask.sum() > 0:
        print("\n過去の予測結果の検証（答え合わせ）を実行中...")
        for idx, row in history_df[mask].iterrows():
            ticker = row['Ticker']
            pred_date = row['Date']
            base_close = row['Close'] # 予測日の終値
            
            try:
                # 予測日の翌日以降のデータをYahoo Financeから取得して実際の株価を確認
                df_check = yf.download(ticker, start=pred_date, progress=False)
                if isinstance(df_check.columns, pd.MultiIndex):
                    df_check.columns = df_check.columns.droplevel(1)
                
                # 予測日より後のデータが存在する場合
                if len(df_check) > 1:
                    next_close = df_check.iloc[1]['Close'] # 翌営業日の終値
                    history_df.loc[idx, 'Actual_Close_Next'] = next_close
                    
                    # 実際に株価が上がっていれば的中(1)、下がっていれば外れ(0)
                    is_hit = 1 if next_close > base_close else 0
                    history_df.loc[idx, 'Is_Hit'] = is_hit
            except:
                pass
                
        # 検証結果を上書き保存
        history_df.to_csv(history_file, index=False)
        
        # 過去の累計的中率（勝率）を計算して表示
        hit_evaluated = history_df.dropna(subset=['Is_Hit'])
        if len(hit_evaluated) > 0:
            total_accuracy = (hit_evaluated['Is_Hit'].sum() / len(hit_evaluated)) * 100
            print(f"📊 【AI運用実績】これまでの予測総数: {len(hit_evaluated)}件 | 累計的中率: {total_accuracy:.1f}%")

    # 本日の新しい予測データを履歴ファイルに追記
    updated_history = pd.concat([history_df, today_log], ignore_index=True)
    updated_history.to_csv(history_file, index=False)
else:
    # 初回実行時はそのままファイルを作成
    today_log.to_csv(history_file, index=False)
    print(f"\n📁 新規の予測履歴ファイル '{history_file}' を作成し、本日の結果を保存しました。")

print(f"本日の予測データが {history_file} に正常に蓄積されました。")