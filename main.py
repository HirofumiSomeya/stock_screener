import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
from tqdm import tqdm
import time
import warnings
import os
import random
from datetime import datetime
warnings.filterwarnings('ignore')

# 1. 銘柄一覧ファイルの指定
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

# --- 事前処理: 過去の予測履歴の検証（AI学習にフィードバックするため最初に実行） ---
history_file = 'ai_prediction_history.csv'
report_file = 'prediction_report.md'
today_str = datetime.now().strftime('%Y-%m-%d')
newly_verified_indices = []
history_df = pd.DataFrame()

if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)
    if 'Return_Pct' not in history_df.columns:
        history_df['Return_Pct'] = np.nan
        
    mask = history_df['Actual_Close_Next'].isna() & (history_df['Date'] < today_str)
    if mask.sum() > 0:
        print("\n🔍 過去の予測結果の検証（答え合わせ）を実行中...")
        for idx, row in history_df[mask].iterrows():
            ticker = row['Ticker']
            pred_date = row['Date']
            base_close = float(row['Close'])
            
            try:
                df_check = yf.download(ticker, start=pred_date, progress=False)
                if isinstance(df_check.columns, pd.MultiIndex):
                    df_check.columns = df_check.columns.droplevel(1)
                
                if len(df_check) > 1:
                    next_close = float(df_check.iloc[1]['Close'])
                    ret_pct = ((next_close - base_close) / base_close) * 100
                    is_hit = 1 if next_close > base_close else 0
                    
                    history_df.loc[idx, 'Actual_Close_Next'] = round(next_close, 1)
                    history_df.loc[idx, 'Return_Pct'] = round(ret_pct, 2)
                    history_df.loc[idx, 'Is_Hit'] = is_hit
                    newly_verified_indices.append(idx)
            except Exception:
                pass
            
            time.sleep(1.0)
                
        history_df.to_csv(history_file, index=False)

# --- 第1段階: スクリーニング ---
print(f"\n第1段階: 株価3000円未満、かつ上昇トレンドの銘柄をスクリーニング中...")
uptrend_data = []
latest_features = []
passed_tickers = []

chunk_size = 25
for i in tqdm(range(0, len(valid_tickers), chunk_size)):
    chunk = valid_tickers[i:i + chunk_size]
    
    success = False
    for attempt in range(3):
        try:
            bulk_data = yf.download(chunk, period="2y", interval="1d", progress=False, group_by='ticker')
            success = True
            break
        except Exception:
            time.sleep(3 * (attempt + 1))
            
    if not success:
        time.sleep(5)
        continue
        
    try:
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
            df['RSI'] = 100 - (100 / (1 + gain / loss))
            
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
            if df.empty: continue
            
            # 過去履歴との照合用に日付文字列カラムを保持
            df['Date_Str'] = pd.to_datetime(df.index).strftime('%Y-%m-%d')
            df['Ticker'] = ticker
            
            uptrend_data.append(df[:-1])
            latest_features.append(df.iloc[-1:])
            passed_tickers.append(ticker)
            
    except Exception:
        continue
        
    time.sleep(random.uniform(2.0, 4.0))

print(f"\nスクリーニング通過: {len(passed_tickers)} 社")

if not uptrend_data:
    print("条件を満たす銘柄がありませんでした。日経平均などの地合いを確認してください。")
else:
    print(f"\n第2段階: 通過した有望銘柄のファンダメンタル情報を取得しています...")
    fund_data = []
    
    for ticker in tqdm(passed_tickers):
        success = False
        info = {}
        
        for attempt in range(3):
            try:
                stock_obj = yf.Ticker(ticker)
                info = stock_obj.info
                if info and ('trailingPE' in info or 'priceToBook' in info):
                    success = True
                    break
            except Exception:
                time.sleep(3 * (attempt + 1))
        
        if success and info:
            fund_data.append({
                'Ticker': ticker,
                'PER': info.get('trailingPE', np.nan),
                'PBR': info.get('priceToBook', np.nan)
            })
        else:
            fund_data.append({
                'Ticker': ticker,
                'PER': np.nan,
                'PBR': np.nan
            })
            
        time.sleep(1.5)
            
    fund_df = pd.DataFrame(fund_data)
    
    combined_train = pd.concat(uptrend_data).merge(fund_df, on='Ticker', how='left')
    combined_latest = pd.concat(latest_features).merge(fund_df, on='Ticker', how='left')
    
    train_means = combined_train.mean(numeric_only=True)
    combined_train = combined_train.fillna(train_means)
    combined_latest = combined_latest.fillna(train_means)

    # --- フィードバック学習（過去の予測・検証結果をAIに反映） ---
    sample_weights = np.ones(len(combined_train))
    combined_train['Past_Hit_Rate'] = 0.5
    combined_latest['Past_Hit_Rate'] = 0.5

    if not history_df.empty:
        verified_df = history_df.dropna(subset=['Is_Hit']).copy()
        if len(verified_df) > 0:
            print(f"\n🧠 過去の検証データ（{len(verified_df)}件）をAIの学習へフィードバック中...")
            
            # 1. 銘柄ごとのAI予測相性スコア（平滑化勝率）を算出して特徴量に追加
            ticker_stats = verified_df.groupby('Ticker')['Is_Hit'].agg(['count', 'sum'])
            ticker_stats['smoothed_rate'] = (ticker_stats['sum'] + 1.0) / (ticker_stats['count'] + 2.0)
            rate_map = ticker_stats['smoothed_rate'].to_dict()
            
            combined_train['Past_Hit_Rate'] = combined_train['Ticker'].map(rate_map).fillna(0.5)
            combined_latest['Past_Hit_Rate'] = combined_latest['Ticker'].map(rate_map).fillna(0.5)
            
            # 2. 過去に予測を外したパターン（Is_Hit == 0）に5倍のペナルティ重みを設定
            verified_ outcome_map = verified_df.set_index(['Ticker', 'Date'])['Is_Hit'].to_dict()
            train_keys = list(zip(combined_train['Ticker'], combined_train['Date_Str']))
            
            matched_misses = 0
            for idx_w, key in enumerate(train_keys):
                if key in verified_outcome_map:
                    if verified_outcome_map[key] == 0:
                        sample_weights[idx_w] = 5.0  # 外した予測日は5倍の重みで反省学習
                        matched_misses += 1
                    else:
                        sample_weights[idx_w] = 2.0  # 的中した予測日も2倍で強化
            print(f"   ➔ 過去の失敗パターン {matched_misses} 件に重点ペナルティ重み（x5.0）を適用しました。")

    features = ['Return', 'Volatility', 'Volume_Change', 'RSI', 'PER', 'PBR', 'Past_Hit_Rate']
    X_train, y_train = combined_train[features], combined_train['Target']
    
    print("\n第3段階: AI学習と明日の勝率ランキング生成中...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.01, max_depth=5, 
        random_state=42, verbose=-1
    )
    # sample_weight を渡してフィードバック学習を実行
    lgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    X_latest = combined_latest[features]
    combined_latest['AI_Win_Probability'] = lgb_model.predict_proba(X_latest)[:, 1]
    
    ranking = combined_latest[['Ticker', 'Close', 'AI_Win_Probability', 'PER', 'PBR']].copy()
    ranking = ranking.sort_values(by='AI_Win_Probability', ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    
    # データの算出
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

    # 0. 最新ランキングのCSV保存
    save_cols = ['Ticker', 'Close', 'AI_Win_Probability', 'PER', 'PBR', 'Target_Buy', 'Take_Profit', 'Stop_Loss']
    ranking[save_cols].to_csv('ai_ranking_with_prices.csv', index=False)
    print("\n📁 最新ランキングを 'ai_ranking_with_prices.csv' に保存しました。")

    # --- 検証レポートの出力と本日の履歴保存 ---
    today_log = ranking[save_cols].head(15).copy()
    today_log['Date'] = today_str
    today_log['Actual_Close_Next'] = np.nan 
    today_log['Return_Pct'] = np.nan
    today_log['Is_Hit'] = np.nan            

    hit_evaluated = history_df.dropna(subset=['Is_Hit']).copy() if not history_df.empty else pd.DataFrame()
    
    if len(hit_evaluated) > 0:
        if newly_verified_indices:
            target_ver_df = history_df.loc[newly_verified_indices]
            ver_title = "今回確定した予測の答え合わせ"
        else:
            latest_ver_date = hit_evaluated['Date'].max()
            target_ver_df = hit_evaluated[hit_evaluated['Date'] == latest_ver_date]
            ver_title = f"直近検証日 ({latest_ver_date}) の答え合わせ"

        print("\n" + "="*75)
        print(f"🎯 【{ver_title}】個別銘柄の勝敗結果")
        print("="*75)
        print(f"{'予測日':<12} {'銘柄':<8} {'AI勝率':<8} {'予測時終値':<10} {'翌日終値':<10} {'騰落率':<9} {'判定'}")
        print("-"*75)
        for _, r in target_ver_df.iterrows():
            prob_s = f"{r['AI_Win_Probability']*100:.1f}%"
            base_s = f"{int(round(r['Close'])):,}円"
            next_s = f"{int(round