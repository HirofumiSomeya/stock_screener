import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
from tqdm import tqdm
import time
import warnings
import os
import random
import requests
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
warnings.filterwarnings('ignore')

# 1. 銘柄一覧ファイルの指定
file_name = 'data_j.xlsx'

# 2. ファイルの読み込み
if file_name.endswith('.csv'):
    df_jpx = pd.read_csv(file_name)
else:
    df_jpx = pd.read_excel(file_name)

# 3. 普通株式（企業）のみを抽出
df_stocks = df_jpx[df_jpx['市場・商品区分'].str.contains('プライム|スタンダード|グロース', na=False)].copy()

# 4. 証券コード・銘柄名・業種マッピングの作成
df_stocks['Ticker'] = df_stocks['コード'].astype(str).str.strip() + ".T"
valid_tickers = df_stocks['Ticker'].tolist()

name_map = dict(zip(df_stocks['Ticker'], df_stocks['銘柄名'].astype(str).str.strip()))
if '33業種区分' in df_stocks.columns:
    sector_map = dict(zip(df_stocks['Ticker'], df_stocks['33業種区分'].astype(str).str.strip()))
else:
    sector_map = {t: '-' for t in valid_tickers}

unique_sectors = sorted(list(set(sector_map.values())))
sector_code_map = {sec: i for i, sec in enumerate(unique_sectors)}

print(f"抽出完了: 全上場企業 {len(valid_tickers)} 銘柄をスクリーニング対象に設定しました。")

# --- 日本時間（JST）での日付取得 ---
JST = timezone(timedelta(hours=9))
today_str = datetime.now(JST).strftime('%Y-%m-%d')
print(f"📅 本日の実行日 (JST): {today_str}")

# --- 📰 経済・海外情勢ニュース センチメント分析ヘルパー ---
def fetch_market_news_sentiment():
    """
    主要経済・世界情勢ニュースの見出しから市場センチメント（-1.0〜+1.0）を算出
    """
    query = "日経平均 OR 米国株 OR 世界経済 OR 為替 OR 地政学"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=ja&gl=JP&ceid=JP:ja"
    
    pos_words = [
        '上昇', '高値', '反発', '好調', '買い', '最高', '改善', '緩和', 
        '堅調', '急伸', '強含み', '続伸', '利下げ', '黒字', '上方修正', '回復'
    ]
    neg_words = [
        '下落', '急落', '安値', '反落', '警戒', '売り', '悪化', '懸念', 
        '暴落', '緊縮', 'リスク', '戦争', '低迷', '利上げ', '赤字', '下方修正', '制裁'
    ]
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        titles = [item.find('title').text for item in root.findall('.//item') if item.find('title') is not None]
        titles = titles[:40] # 直近40件の見出しを解析
        
        pos_count = 0
        neg_count = 0
        for t in titles:
            for w in pos_words:
                if w in t: pos_count += 1
            for w in neg_words:
                if w in t: neg_count += 1
                
        total = pos_count + neg_count
        if total == 0:
            score = 0.0
        else:
            score = (pos_count - neg_count) / total
            
        print(f"📰 本日の経済・世界情勢ニュース感情スコア: {score:+.2f} (ポジティブ:{pos_count}件, ネガティブ:{neg_count}件)")
        return float(round(score, 3))
    except Exception as e:
        print(f"⚠️ ニュース感情分析スキップ (デフォルト0.0): {e}")
        return 0.0

# --- J-Quants API V2 ヘルパー関数 ---
def get_jquants_api_key():
    api_key = os.environ.get('JQUANTS_API_KEY', '').strip()
    return api_key if api_key else None

def fetch_jquants_fundamentals(tickers, api_key):
    """通過銘柄の最新財務データをJ-Quants API V2から取得"""
    fund_dict = {}
    headers = {'x-api-key': api_key}
    print("\n🏛️ J-Quants公式API (V2) から通過銘柄の財務データ（EPS・BPS）を取得中...")
    for t in tqdm(tickers):
        code4 = t.replace('.T', '')
        try:
            r = requests.get(
                "https://api.jquants.com/v2/fins/summary",
                params={"code": code4},
                headers=headers,
                timeout=10
            )
            if r.status_code == 200:
                data_list = r.json().get('data', []) or r.json().get('fins_summary', []) or r.json().get('statements', [])
                if data_list:
                    latest_s = data_list[-1]
                    eps = (
                        latest_s.get('FEPS') or latest_s.get('NxFEPS') or 
                        latest_s.get('EPS') or latest_s.get('ForecastEarningsPerShare') or 
                        latest_s.get('EarningsPerShare')
                    )
                    bps = latest_s.get('BPS') or latest_s.get('BookValuePerShare')
                    eps_val = float(eps) if eps not in (None, "", "-") else np.nan
                    bps_val = float(bps) if bps not in (None, "", "-") else np.nan
                    fund_dict[t] = {'EPS': eps_val, 'BPS': bps_val}
        except Exception:
            pass
        time.sleep(0.12)
    return fund_dict

# --- 事前処理1: 過去の予測履歴の検証（答え合わせ） ---
history_file = 'ai_prediction_history.csv'
report_file = 'prediction_report.md'
newly_verified_indices = []
history_df = pd.DataFrame()

if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)
    if 'Return_Pct' not in history_df.columns:
        history_df['Return_Pct'] = np.nan
    if 'Name' not in history_df.columns:
        history_df['Name'] = history_df['Ticker'].map(name_map).fillna('-')
    if 'Sector' not in history_df.columns:
        history_df['Sector'] = history_df['Ticker'].map(sector_map).fillna('-')
        
    mask = history_df['Actual_Close_Next'].isna() & (history_df['Date'] < today_str)
    print(f"🔍 検証対象となる過去の予測データ数: {mask.sum()} 件")
    
    if mask.sum() > 0:
        print("\n🔍 過去の予測結果の検証（答え合わせ）を実行中...")
        for idx, row in history_df[mask].iterrows():
            ticker = row['Ticker']
            pred_date = str(row['Date']).strip()
            base_close = float(row['Close'])
            
            try:
                df_check = yf.download(ticker, start=pred_date, progress=False)
                if isinstance(df_check.columns, pd.MultiIndex):
                    df_check.columns = df_check.columns.droplevel(1)
                
                df_check = df_check.dropna(subset=['Close'])
                df_check.index = pd.to_datetime(df_check.index).strftime('%Y-%m-%d')
                
                df_future = df_check[df_check.index > pred_date]
                
                if not df_future.empty:
                    next_close = float(df_future.iloc[0]['Close'])
                    ret_pct = ((next_close - base_close) / base_close) * 100
                    is_hit = 1 if next_close > base_close else 0
                    
                    history_df.loc[idx, 'Actual_Close_Next'] = round(next_close, 1)
                    history_df.loc[idx, 'Return_Pct'] = round(ret_pct, 2)
                    history_df.loc[idx, 'Is_Hit'] = is_hit
                    newly_verified_indices.append(idx)
            except Exception:
                pass
            
            time.sleep(0.5)
                
        history_df.to_csv(history_file, index=False)
        print(f"   ➔ 新たに {len(newly_verified_indices)} 件の答え合わせが完了しました。")

# --- 事前処理2: 市場全体（マクロ指標＋ニュース感情指標）の取得 ---
print("\n🌐 市場全体指標（日経平均・グロース指数・S&P500・ドル円・VIX・米金利）を取得中...")
macro_tickers = ['^N225', '2516.T', '^GSPC', 'JPY=X', '^VIX', '^TNX']
macro_df = pd.DataFrame()

# ニュース感情スコアの算出
current_news_sentiment = fetch_market_news_sentiment()

for attempt in range(3):
    try:
        m_data = yf.download(macro_tickers, period="2y", interval="1d", progress=False, group_by='ticker')
        
        def extract_close(ticker_sym):
            if ticker_sym in m_data.columns.levels[0]:
                return m_data[ticker_sym]['Close']
            return pd.Series(dtype=float)

        macro_raw = pd.DataFrame({
            'N225': extract_close('^N225'),
            'GROWTH': extract_close('2516.T'),
            'SP500': extract_close('^GSPC'),
            'USDJPY': extract_close('JPY=X'),
            'VIX': extract_close('^VIX'),
            'US10Y': extract_close('^TNX')
        }).sort_index().ffill()

        macro_df['N225_Return'] = macro_raw['N225'].pct_change()
        macro_df['N225_Return_5d'] = macro_raw['N225'].pct_change(5)
        macro_df['N225_Trend'] = (macro_raw['N225'] / macro_raw['N225'].rolling(25).mean()) - 1
        macro_df['Growth_Return'] = macro_raw['GROWTH'].pct_change()
        macro_df['Growth_Trend'] = (macro_raw['GROWTH'] / macro_raw['GROWTH'].rolling(25).mean()) - 1
        macro_df['SP500_Return'] = macro_raw['SP500'].pct_change()
        macro_df['USDJPY_Return'] = macro_raw['USDJPY'].pct_change()
        macro_df['VIX_Level'] = macro_raw['VIX']
        macro_df['US10Y_Return'] = macro_raw['US10Y'].pct_change()
        macro_df['Date_Str'] = pd.to_datetime(macro_df.index).strftime('%Y-%m-%d')
        macro_df = macro_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        print("   ➔ マクロ指標の取得完了。")
        break
    except Exception:
        time.sleep(3 * (attempt + 1))

time.sleep(2.0)

# --- 第1段階: スクリーニング & マルチタイムフレーム特徴量の一括生成 ---
print(f"\n第1段階: 株価3000円未満、かつ上昇トレンドの銘柄をスクリーニング中...")
uptrend_data = []
latest_features = []
passed_tickers = []

chunk_size = 30
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
                
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
            if df.empty or len(df) < 100: continue
            
            # 3,000円未満フィルター
            if df['Close'].iloc[-1] >= 3000:
                continue
                
            # 移動平均線（短期5日・中期25日・長期75日）
            df['SMA5'] = df['Close'].rolling(window=5).mean()
            df['SMA25'] = df['Close'].rolling(window=25).mean()
            df['SMA75'] = df['Close'].rolling(window=75).mean()

            latest = df.iloc[-1]
            if pd.isna(latest['SMA75']):
                continue
            
            is_uptrend = (latest['Close'] > latest['SMA25']) and (latest['Close'] > latest['SMA75'])
            if not is_uptrend:
                continue
                
            # マルチタイムフレーム特徴量
            df['Return'] = df['Close'].pct_change()
            df['Return_3d'] = df['Close'].pct_change(3)
            df['Return_5d'] = df['Close'].pct_change(5)
            df['Return_20d'] = df['Close'].pct_change(20)
            df['Return_60d'] = df['Close'].pct_change(60)
            
            vol_5d = df['Return'].rolling(window=5).std()
            df['Volatility'] = df['Return'].rolling(window=10).std()
            vol_20d = df['Return'].rolling(window=20).std()
            df['Vol_Regime'] = vol_5d / (vol_20d + 1e-8)
            
            df['Volume_Change'] = df['Volume'].pct_change()
            vol_ma20 = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / (vol_ma20 + 1e-8)
            
            df['SMA5_Dev'] = (df['Close'] / df['SMA5']) - 1
            df['SMA25_Dev'] = (df['Close'] / df['SMA25']) - 1
            df['SMA75_Dev'] = (df['Close'] / df['SMA75']) - 1
            
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_sig = macd.ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = (macd - macd_sig) / (df['Close'] + 1e-8)
            
            high_60 = df['High'].rolling(window=60).max()
            low_60 = df['Low'].rolling(window=60).min()
            df['Range_60d_Pos'] = (df['Close'] - low_60) / (high_60 - low_60 + 1e-8)
            
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            df['BB_Pos'] = (df['Close'] - sma20) / (2 * std20 + 1e-8)
            
            delta = df['Close'].diff()
            gain14 = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss14 = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + gain14 / (loss14 + 1e-8)))
            
            gain5 = (delta.where(delta > 0, 0)).rolling(window=5).mean()
            loss5 = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
            df['RSI_5'] = 100 - (100 / (1 + gain5 / (loss5 + 1e-8)))
            
            # 正解ラベル（翌営業日に上がったか）
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
            if df.empty: continue
            
            df['Date_Str'] = pd.to_datetime(df.index).strftime('%Y-%m-%d')
            df['Ticker'] = ticker
            
            uptrend_data.append(df[:-1])
            latest_features.append(df.iloc[-1:])
            passed_tickers.append(ticker)
            
    except Exception:
        continue
        
    time.sleep(random.uniform(1.0, 2.0))

print(f"\nスクリーニング通過: {len(passed_tickers)} 社")

if not uptrend_data:
    print("条件を満たす銘柄がありませんでした。日経平均などの地合いを確認してください。")
else:
    combined_train = pd.concat(uptrend_data)
    combined_latest = pd.concat(latest_features)
    
    # --- J-Quants API V2 財務データの統合 ---
    jq_api_key = get_jquants_api_key()
    if jq_api_key:
        jq_fund = fetch_jquants_fundamentals(passed_tickers, jq_api_key)
        eps_map = {k: v['EPS'] for k, v in jq_fund.items()}
        bps_map = {k: v['BPS'] for k, v in jq_fund.items()}
        
        for df_part in [combined_train, combined_latest]:
            eps_s = df_part['Ticker'].map(eps_map)
            bps_s = df_part['Ticker'].map(bps_map)
            df_part['PER'] = np.where(eps_s > 0, df_part['Close'] / eps_s, np.nan)
            df_part['PBR'] = np.where(bps_s > 0, df_part['Close'] / bps_s, np.nan)
            df_part['ROE'] = np.where(bps_s > 0, (eps_s / bps_s) * 100, np.nan)
        print(f"   ➔ J-Quants財務指標の取得完了（有効取得: {len(jq_fund)} 社）。")
    else:
        print("\n💡 JQUANTS_API_KEY が未設定のため、マルチタイムフレーム＆マクロ指標モードで学習します。")
        for df_part in [combined_train, combined_latest]:
            df_part['PER'] = 15.0
            df_part['PBR'] = 1.2
            df_part['ROE'] = 8.0

    # 業種コードの付与
    combined_train['Sector_Code'] = combined_train['Ticker'].map(sector_map).map(sector_code_map).fillna(-1)
    combined_latest['Sector_Code'] = combined_latest['Ticker'].map(sector_map).map(sector_code_map).fillna(-1)

    # 業種別モメンタム
    sector_daily_ret = combined_train.groupby(['Date_Str', 'Sector_Code'])['Return'].transform('mean')
    combined_train['Sector_Return'] = sector_daily_ret.fillna(0)
    
    latest_sec_ret = combined_latest.groupby('Sector_Code')['Return'].transform('mean')
    combined_latest['Sector_Return'] = latest_sec_ret.fillna(0)

    # マクロ指標の結合
    macro_cols = [
        'N225_Return', 'N225_Return_5d', 'N225_Trend', 'Growth_Return', 'Growth_Trend',
        'SP500_Return', 'USDJPY_Return', 'VIX_Level', 'US10Y_Return'
    ]
    if not macro_df.empty:
        combined_train = combined_train.merge(macro_df[['Date_Str'] + macro_cols], on='Date_Str', how='left')
        combined_latest = combined_latest.merge(macro_df[['Date_Str'] + macro_cols], on='Date_Str', how='left')
    else:
        for col in macro_cols:
            combined_train[col] = 0.0
            combined_latest[col] = 0.0

    # 📰 ニュース感情スコア（最新の市場センチメントを特徴量として結合）
    combined_train['News_Sentiment'] = 0.0 # 過去の学習データ平均
    combined_latest['News_Sentiment'] = current_news_sentiment # 最新の予測データ

    # 対市場の相対的強さ
    combined_train['Relative_Return'] = combined_train['Return'] - combined_train['N225_Return'].fillna(0)
    combined_latest['Relative_Return'] = combined_latest['Return'] - combined_latest['N225_Return'].fillna(0)
    combined_train['Relative_Return_5d'] = combined_train['Return_5d'] - combined_train['N225_Return_5d'].fillna(0)
    combined_latest['Relative_Return_5d'] = combined_latest['Return_5d'] - combined_latest['N225_Return_5d'].fillna(0)
    combined_train['Relative_Growth'] = combined_train['Return'] - combined_train['Growth_Return'].fillna(0)
    combined_latest['Relative_Growth'] = combined_latest['Return'] - combined_latest['Growth_Return'].fillna(0)

    train_means = combined_train.mean(numeric_only=True)
    combined_train = combined_train.replace([np.inf, -np.inf], np.nan).fillna(train_means).fillna(0)
    combined_latest = combined_latest.replace([np.inf, -np.inf], np.nan).fillna(train_means).fillna(0)

    # --- フィードバック学習（過去の予測・検証結果をAIに反映） ---
    sample_weights = np.ones(len(combined_train))
    combined_train['Past_Hit_Rate'] = 0.5
    combined_latest['Past_Hit_Rate'] = 0.5

    if not history_df.empty:
        verified_df = history_df.dropna(subset=['Is_Hit']).copy()
        if len(verified_df) > 0:
            print(f"\n🧠 過去の検証データ（{len(verified_df)}件）をAIの学習へフィードバック中...")
            
            ticker_stats = verified_df.groupby('Ticker')['Is_Hit'].agg(['count', 'sum'])
            ticker_stats['smoothed_rate'] = (ticker_stats['sum'] + 1.0) / (ticker_stats['count'] + 2.0)
            rate_map = ticker_stats['smoothed_rate'].to_dict()
            
            combined_train['Past_Hit_Rate'] = combined_train['Ticker'].map(rate_map).fillna(0.5)
            combined_latest['Past_Hit_Rate'] = combined_latest['Ticker'].map(rate_map).fillna(0.5)
            
            verified_outcome_map = verified_df.set_index(['Ticker', 'Date'])['Is_Hit'].to_dict()
            train_keys = list(zip(combined_train['Ticker'], combined_train['Date_Str']))
            
            matched_misses = 0
            for idx_w, key in enumerate(train_keys):
                if key in verified_outcome_map:
                    if verified_outcome_map[key] == 0:
                        sample_weights[idx_w] = 5.0
                        matched_misses += 1
                    else:
                        sample_weights[idx_w] = 2.0
            print(f"   ➔ 過去の失敗パターン {matched_misses} 件に重点ペナルティ重み（x5.0）を適用しました。")

    # 全33種類の特徴量（マルチタイムフレーム＋テクニカル＋財務＋マクロ＋ニュース感情）
    features = [
        'Return', 'Return_3d', 'Return_5d', 'Return_20d', 'Return_60d',
        'Volatility', 'Vol_Regime', 'Volume_Change', 'Volume_Ratio',
        'SMA5_Dev', 'SMA25_Dev', 'SMA75_Dev', 'MACD_Hist', 'Range_60d_Pos', 'BB_Pos',
        'RSI', 'RSI_5', 'PER', 'PBR', 'ROE',
        'Sector_Code', 'Sector_Return', 'Past_Hit_Rate',
        'N225_Return', 'N225_Return_5d', 'N225_Trend', 'Growth_Return', 'Growth_Trend',
        'Relative_Return', 'Relative_Return_5d', 'Relative_Growth',
        'SP500_Return', 'USDJPY_Return', 'VIX_Level', 'US10Y_Return',
        'News_Sentiment' # 📰 経済・世界情勢感情指標
    ]
    X_train, y_train = combined_train[features], combined_train['Target']
    
    print("\n第2段階: AI学習（ニュース感情・需給・マルチタイムフレーム・マクロ地合い統合）と勝率ランキング生成中...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.01, max_depth=5, 
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    X_latest = combined_latest[features]
    combined_latest['AI_Win_Probability'] = lgb_model.predict_proba(X_latest)[:, 1]
    
    ranking = combined_latest[['Ticker', 'Close', 'AI_Win_Probability', 'Volume_Ratio', 'RSI', 'Return_5d', 'Return_20d']].copy()
    ranking['Name'] = ranking['Ticker'].map(name_map).fillna('-')
    ranking['Sector'] = ranking['Ticker'].map(sector_map).fillna('-')
    
    ranking = ranking.sort_values(by='AI_Win_Probability', ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    
    # データの算出
    ranking['Target_Buy'] = ranking['Close'].round(0).astype(int)
    ranking['Take_Profit'] = (ranking['Close'] * 1.05).round(0).astype(int)
    ranking['Stop_Loss'] = (ranking['Close'] * 0.98).round(0).astype(int)
    
    # 表示用の文字列加工
    ranking['AI_Win_Probability_Str'] = (ranking['AI_Win_Probability'] * 100).round(1).astype(str) + '%'
    ranking['Close_Str'] = ranking['Close'].round(0).astype(int).apply(lambda x: f"{x:,}円")
    ranking['Vol_Ratio_Str'] = ranking['Volume_Ratio'].round(1).astype(str) + '倍'
    ranking['Ret5d_Str'] = (ranking['Return_5d'] * 100).round(1).apply(lambda x: f"{x:+.1f}%")
    ranking['RSI_Str'] = ranking['RSI'].round(0).astype(int).astype(str)
    ranking['TP_Str'] = ranking['Take_Profit'].apply(lambda x: f"{x:,}円 (+5%)")
    ranking['SL_Str'] = ranking['Stop_Loss'].apply(lambda x: f"{x:,}円 (-2%)")

    print("\n" + "="*110)
    print("🔥 【AI厳選】3,000円未満・上昇有望銘柄トレードプラン (上位15選)")
    print("="*110)
    print(f"{'順位':<4} {'コード':<8} {'銘柄名':<16} {'業種':<12} {'現在値':<9} {'AI勝率':<7} {'5日騰落':<8} {'出来高倍':<8} {'利確目標(+5%)':<14} {'損切り(-2%)':<12}")
    print("-"*110)

    for idx, row in ranking.head(15).iterrows():
        name_short = row['Name'][:10]
        sec_short = row['Sector'][:8]
        print(f"{idx:<4} {row['Ticker']:<8} {name_short:<16} {sec_short:<12} {row['Close_Str']:<9} {row['AI_Win_Probability_Str']:<7} "
              f"{row['Ret5d_Str']:<8} {row['Vol_Ratio_Str']:<8} {row['TP_Str']:<14} {row['SL_Str']:<12}")
    print("="*110)

    # 0. 最新ランキングのCSV保存
    save_cols = ['Ticker', 'Name', 'Sector', 'Close', 'AI_Win_Probability', 'Target_Buy', 'Take_Profit', 'Stop_Loss']
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

        print("\n" + "="*95)
        print(f"🎯 【{ver_title}】個別銘柄の勝敗結果")
        print("="*95)
        print(f"{'予測日':<11} {'コード':<8} {'銘柄名':<16} {'AI勝率':<7} {'予測時終値':<10} {'翌日終値':<10} {'騰落率':<8} {'判定'}")
        print("-"*95)
        for _, r in target_ver_df.iterrows():
            prob_s = f"{r['AI_Win_Probability']*100:.1f}%"
            base_s = f"{int(round(r['Close'])):,}円"
            next_s = f"{int(round(r['Actual_Close_Next'])):,}円"
            ret_val = r['Return_Pct'] if pd.notna(r['Return_Pct']) else ((r['Actual_Close_Next'] - r['Close']) / r['Close'] * 100)
            ret_s = f"{ret_val:+.2f}%"
            mark = "⭕ 的中" if r['Is_Hit'] == 1 else "❌ 外れ"
            r_name = str(r.get('Name', name_map.get(r['Ticker'], '-')))[:10]
            print(f"{r['Date']:<11} {r['Ticker']:<8} {r_name:<16} {prob_s:<7} {base_s:<10} {next_s:<10} {ret_s:<8} {mark}")
        print("-"*95)

        total_hits = int(hit_evaluated['Is_Hit'].sum())
        total_count = len(hit_evaluated)
        total_acc = (total_hits / total_count) * 100
        avg_ret = ((hit_evaluated['Actual_Close_Next'] - hit_evaluated['Close']) / hit_evaluated['Close'] * 100).mean()
        print(f"📊 【累計AI運用実績】 通算: {total_hits}勝 {total_count - total_hits}敗 (計{total_count}件) | 累計的中率: {total_acc:.1f}% | 平均騰落率: {avg_ret:+.2f}%")
        print("="*95)

    # Markdownレポートの生成
    sentiment_desc = "強気 (リスクオン)" if current_news_sentiment > 0.1 else ("弱気 (リスクオフ・警戒)" if current_news_sentiment < -0.1 else "中立")
    md_lines = [f"# 📊 AI株価予測 検証レポート (最終更新: {today_str})\n"]
    
    if not macro_df.empty:
        latest_m = macro_df.iloc[-1]
        md_lines.append("## 🌐 直近の市場環境（マクロ・ニュース・中小型株指標）")
        md_lines.append(f"- **ニュース・市場センチメント**: **{sentiment_desc}**（スコア: {current_news_sentiment:+.2f}）")
        md_lines.append(f"- **日経平均 前日比**: {latest_m['N225_Return']*100:+.2f}% （5日騰落: {latest_m['N225_Return_5d']*100:+.2f}% / 25日線乖離: {latest_m['N225_Trend']*100:+.2f}%）")
        md_lines.append(f"- **東証グロース250 前日比**: {latest_m['Growth_Return']*100:+.2f}% （25日線乖離: {latest_m['Growth_Trend']*100:+.2f}%）")
        md_lines.append(f"- **米国S&P500 前日比**: {latest_m['SP500_Return']*100:+.2f}% | **米10年債利回り変化**: {latest_m['US10Y_Return']*100:+.2f}%")
        md_lines.append(f"- **ドル円変動率**: {latest_m['USDJPY_Return']*100:+.2f}% | **VIX指数**: {latest_m['VIX_Level']:.1f}\n")

    if len(hit_evaluated) > 0:
        md_lines.append("## 1. 累計パフォーマンス（マルチタイムフレーム＆ニュース統合＆自己改善適用済）")
        md_lines.append(f"- **累計検証数**: {total_count} 銘柄（{total_hits}勝 {total_count - total_hits}敗）")
        md_lines.append(f"- **累計的中率（勝率）**: **{total_acc:.1f}%**")
        md_lines.append(f"- **1銘柄あたり平均騰落率**: **{avg_ret:+.2f}%**\n")
        
        md_lines.append(f"## 2. {ver_title}")
        md_lines.append("| 予測日 | コード | 銘柄名 | 業種 | AI勝率 | 予測時終値 | 翌営業日終値 | 騰落率 | 判定 |")
        md_lines.append("| :--- | :--- | :--- | :--- | :---: | ---: | ---: | ---: | :---: |")
        for _, r in target_ver_df.iterrows():
            ret_val = r['Return_Pct'] if pd.notna(r['Return_Pct']) else ((r['Actual_Close_Next'] - r['Close']) / r['Close'] * 100)
            mark = "⭕ 的中" if r['Is_Hit'] == 1 else "❌ 外れ"
            r_name = r.get('Name', name_map.get(r['Ticker'], '-'))
            r_sec = r.get('Sector', sector_map.get(r['Ticker'], '-'))
            md_lines.append(f"| {r['Date']} | {r['Ticker']} | {r_name} | {r_sec} | {r['AI_Win_Probability']*100:.1f}% | {int(round(r['Close'])):,}円 | {int(round(r['Actual_Close_Next'])):,}円 | {ret_val:+.2f}% | {mark} |")
        md_lines.append("\n")
    else:
        md_lines.append("> ※まだ翌営業日の終値が確定した検証済みデータがありません。翌営業日以降の実行時に答え合わせ結果とフィードバック学習が反映されます。\n")

    md_lines.append(f"## 3. 本日 ({today_str}) のAI厳選ランキング (上位15銘柄)")
    md_lines.append("| 順位 | コード | 銘柄名 | 業種 | 現在値 | AI勝率 | 5日騰落 | 出来高倍率 | RSI | 利確目標 (+5%) | 損切り (-2%) |")
    md_lines.append("| :---: | :--- | :--- | :--- | ---: | :---: | :---: | :---: | :---: | ---: | ---: |")
    for idx, r in ranking.head(15).iterrows():
        md_lines.append(f"| {idx} | {r['Ticker']} | {r['Name']} | {r['Sector']} | {r['Close_Str']} | {r['AI_Win_Probability_Str']} | {r['Ret5d_Str']} | {r['Vol_Ratio_Str']} | {r['RSI_Str']} | {r['TP_Str']} | {r['SL_Str']} |")

    report_content = "\n".join(md_lines)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"📝 詳細検証レポートを '{report_file}' に出力しました。")

    if 'GITHUB_STEP_SUMMARY' in os.environ:
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write(report_content + "\n")

    # 履歴ファイルの更新・保存
    if not history_df.empty:
        history_df = history_df[history_df['Date'] != today_str]
        updated_history = pd.concat([history_df, today_log], ignore_index=True)
        updated_history.to_csv(history_file, index=False)
    else:
        today_log.to_csv(history_file, index=False)
        print(f"\n📁 新規の予測履歴ファイル '{history_file}' を作成し、本日の結果を保存しました。")

    print(f"本日の予測データが {history_file} に正常に蓄積されました。")