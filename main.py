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

# --- 日本時間（JST）での日付・時刻取得 ---
JST = timezone(timedelta(hours=9))
now_jst = datetime.now(JST)
today_str = now_jst.strftime('%Y-%m-%d')
print(f"📅 本日の実行日 (JST): {today_str} ({now_jst.strftime('%A')} {now_jst.strftime('%H:%M')})")

# =====================================================================
# 🛡️ 安全ガード1: 16:00前の実行時は「Dry-run（テスト実行モード）」に設定
# =====================================================================
is_dry_run = now_jst.hour < 16
if is_dry_run:
    print("\n" + "!"*80)
    print("⚠️ 【Dry-run（テスト実行モード）起動】")
    print(f"   現在の実行時刻は {now_jst.strftime('%H:%M')} です（16:00前のザラ場中・途中経過）。")
    print("   データの未確定による誤判定や本番履歴の汚染を防ぐため、")
    print("   『ai_prediction_history.csv』への本番上書き保存は行われません。")
    print("!"*80 + "\n")
else:
    print("✅ 16:00以降の実行を確認しました（本番更新モードで動作します）。\n")

# =====================================================================
# 🛡️ 安全ガード2: 東証営業日・祝日判定（土日および市場休業日スキップ）
# =====================================================================
# 土日判定
if now_jst.weekday() >= 5:
    print(f"🛑 本日は週末（土曜/日曜）のため東証は休場です。スクリーニング処理をスキップします。")
    exit(0)

# 祝日・休場日判定（日経平均の直近取引日から東証が開いていたか確認）
try:
    check_n225 = yf.download('^N225', period='5d', progress=False)
    if isinstance(check_n225.columns, pd.MultiIndex):
        check_n225.columns = check_n225.columns.droplevel(1)
    check_n225.index = pd.to_datetime(check_n225.index).tz_localize(None).strftime('%Y-%m-%d')
    latest_market_date = check_n225.index[-1]
    
    if now_jst.hour >= 16 and latest_market_date != today_str:
        print(f"🛑 本日 ({today_str}) は東証の休場日・祝日です（直近取引日: {latest_market_date}）。")
        print(f"   ➔ 不正な重複予測の生成を防ぐため、本日のスクリーニング処理を安全に終了します。")
        exit(0)
except Exception as e:
    print(f"⚠️ 市場営業日チェック警告 (処理を継続します): {e}")

# --- 📰 経済・海外情勢ニュース センチメント分析ヘルパー ---
def fetch_market_news_sentiment():
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
        titles = titles[:40]
        
        pos_count = 0
        neg_count = 0
        for t in titles:
            for w in pos_words:
                if w in t: pos_count += 1
            for w in neg_words:
                if w in t: neg_count += 1
                
        total = pos_count + neg_count
        score = (pos_count - neg_count) / total if total > 0 else 0.0
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

# --- 🎯 精密トリプルバリア判定関数 ---
def evaluate_triple_barrier(base_close, next_open, next_high, next_low, next_close):
    tp_price = base_close * 1.02   # 利確ライン (+2.0%)
    sl_price = base_close * 0.985  # 損切りライン (-1.5%)
    
    if next_open >= tp_price:
        return 1
    if next_open <= sl_price:
        return 0

    reached_tp = (next_high >= tp_price)
    reached_sl = (next_low <= sl_price)

    if reached_tp and not reached_sl:
        return 1
    if reached_sl and not reached_tp:
        return 0
    if not reached_tp and not reached_sl:
        return 0

    if next_close >= base_close:
        return 1
    else:
        return 0

# --- 事前処理1: 過去の予測履歴の検証 ---
history_file = 'ai_prediction_history.csv'
report_file = 'prediction_report_dry_run.md' if is_dry_run else 'prediction_report.md'
newly_verified_indices = []
history_df = pd.DataFrame()

if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)
    for col in ['Return_Pct', 'Max_Up_Pct', 'Max_Down_Pct']:
        if col not in history_df.columns:
            history_df[col] = np.nan
    if 'Name' not in history_df.columns:
        history_df['Name'] = history_df['Ticker'].map(name_map).fillna('-')
    if 'Sector' not in history_df.columns:
        history_df['Sector'] = history_df['Ticker'].map(sector_map).fillna('-')
        
    mask = (history_df['Date'] < today_str) & (
        history_df['Actual_Close_Next'].isna() | 
        history_df['Max_Up_Pct'].isna()
    )
    print(f"🔍 検証対象となる過去の予測データ数: {mask.sum()} 件")
    
    if mask.sum() > 0:
        print("\n🔍 過去の予測結果の検証（精密トリプルバリア法：利確+2.0%/損切-1.5%）を実行中...")
        for idx, row in history_df[mask].iterrows():
            ticker = row['Ticker']
            pred_date = str(row['Date']).strip()
            base_close = float(row['Close'])
            
            try:
                start_dt = (datetime.strptime(pred_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
                df_check = yf.download(ticker, start=start_dt, progress=False)
                
                if isinstance(df_check.columns, pd.MultiIndex):
                    df_check.columns = df_check.columns.droplevel(1)
                
                df_check = df_check.dropna(subset=['Open', 'Close', 'High', 'Low'])
                if df_check.empty:
                    continue

                df_check.index = pd.to_datetime(df_check.index).tz_localize(None).strftime('%Y-%m-%d')
                df_future = df_check[df_check.index > pred_date]
                
                if not df_future.empty:
                    next_day = df_future.iloc[0]
                    next_open = float(next_day['Open'])
                    next_close = float(next_day['Close'])
                    next_high = float(next_day['High'])
                    next_low = float(next_day['Low'])
                    
                    if np.isnan(next_close) or np.isnan(next_high) or np.isnan(next_low) or next_close <= 0:
                        continue

                    ret_pct = ((next_close - base_close) / base_close) * 100
                    max_up_pct = ((next_high - base_close) / base_close) * 100
                    max_down_pct = ((next_low - base_close) / base_close) * 100
                    
                    is_hit = evaluate_triple_barrier(base_close, next_open, next_high, next_low, next_close)
                    
                    history_df.loc[idx, 'Actual_Close_Next'] = round(next_close, 1)
                    history_df.loc[idx, 'Return_Pct'] = round(ret_pct, 2)
                    history_df.loc[idx, 'Max_Up_Pct'] = round(max_up_pct, 2)
                    history_df.loc[idx, 'Max_Down_Pct'] = round(max_down_pct, 2)
                    history_df.loc[idx, 'Is_Hit'] = is_hit
                    newly_verified_indices.append(idx)
            except Exception as e:
                print(f"⚠️ {ticker} の検証スキップ: {e}")
            
            time.sleep(0.3)
                
        if not is_dry_run:
            history_df.to_csv(history_file, index=False)
            print(f"   ➔ 新たに {len(newly_verified_indices)} 件の確定した答え合わせを履歴CSVへ保存しました。")
        else:
            print(f"   ➔ [Dry-run] 試算として {len(newly_verified_indices)} 件の答え合わせを確認しました（CSVへの書き込みはスキップ）。")

# --- 事前処理2: 市場全体（マクロ指標＋ニュース感情指標）の取得 ---
print("\n🌐 市場全体指標（日経平均・グロース指数・S&P500・ドル円・VIX・米金利）を取得中...")
macro_tickers = ['^N225', '2516.T', '^GSPC', 'JPY=X', '^VIX', '^TNX']
macro_df = pd.DataFrame()

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

# --- 第1段階: スクリーニング（緩和版：押し目許容・出来高0.6倍基準） ---
print(f"\n第1段階: 株価4000円未満・順張り＆健全な押し目銘柄をスクリーニング中...")
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
                
            df = df.dropna(subset=['Open', 'Close', 'High', 'Low', 'Volume'])
            if df.empty or len(df) < 100: continue
            
            if df['Close'].iloc[-1] >= 4000:
                continue
                
            df['SMA5'] = df['Close'].rolling(window=5).mean()
            df['SMA25'] = df['Close'].rolling(window=25).mean()
            df['SMA75'] = df['Close'].rolling(window=75).mean()

            latest = df.iloc[-1]
            if pd.isna(latest['SMA25']):
                continue
            
            vol_ma20 = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / (vol_ma20 + 1e-8)
            df['Return_5d'] = df['Close'].pct_change(5)

            latest_ret5d = latest['Return_5d'] if pd.notna(latest['Return_5d']) else 0.0
            latest_vol_ratio = latest['Volume_Ratio'] if pd.notna(latest['Volume_Ratio']) else 1.0

            sma75_ok = (latest['Close'] > latest['SMA75']) if pd.notna(latest['SMA75']) else True
            
            is_valid_trend = (
                (latest['Close'] > latest['SMA25']) and
                sma75_ok and
                (latest_ret5d >= -0.05) and
                (latest_vol_ratio >= 0.6)
            )
            if not is_valid_trend:
                continue
                
            df['Return'] = df['Close'].pct_change()
            df['Return_3d'] = df['Close'].pct_change(3)
            df['Return_20d'] = df['Close'].pct_change(20)
            df['Return_60d'] = df['Close'].pct_change(60)
            
            vol_5d = df['Return'].rolling(window=5).std()
            df['Volatility'] = df['Return'].rolling(window=10).std()
            vol_20d = df['Return'].rolling(window=20).std()
            df['Vol_Regime'] = vol_5d / (vol_20d + 1e-8)
            
            df['Volume_Change'] = df['Volume'].pct_change()
            
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
            
            next_open = df['Open'].shift(-1)
            next_high = df['High'].shift(-1)
            next_low = df['Low'].shift(-1)
            next_close = df['Close'].shift(-1)
            curr_close = df['Close']
            
            tp_price = curr_close * 1.02
            sl_price = curr_close * 0.985
            
            target_series = pd.Series(0, index=df.index)
            target_series[next_open >= tp_price] = 1
            target_series[(next_high >= tp_price) & (next_low > sl_price)] = 1
            target_series[(next_high >= tp_price) & (next_low <= sl_price) & (next_close >= curr_close)] = 1
            
            df['Target'] = target_series
            
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

# =====================================================================
# 1. スクリーニング通過銘柄がある場合の「AI学習・本日の予測」処理
# =====================================================================
ranking = pd.DataFrame()
today_log = pd.DataFrame()

if not uptrend_data:
    print("\n⚠️ 緩和条件でもスクリーニング通過銘柄は 0 社でした。")
    print("   ➔ 市場全体の急落警戒のため、本日は【見送り（ノーポジション）】と判定します。")
else:
    combined_train = pd.concat(uptrend_data)
    combined_latest = pd.concat(latest_features)
    
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

    combined_train['Sector_Code'] = combined_train['Ticker'].map(sector_map).map(sector_code_map).fillna(-1)
    combined_latest['Sector_Code'] = combined_latest['Ticker'].map(sector_map).map(sector_code_map).fillna(-1)

    sector_daily_ret = combined_train.groupby(['Date_Str', 'Sector_Code'])['Return'].transform('mean')
    combined_train['Sector_Return'] = sector_daily_ret.fillna(0)
    
    latest_sec_ret = combined_latest.groupby('Sector_Code')['Return'].transform('mean')
    combined_latest['Sector_Return'] = latest_sec_ret.fillna(0)

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

    combined_train['News_Sentiment'] = 0.0
    combined_latest['News_Sentiment'] = current_news_sentiment

    combined_train['Relative_Return'] = combined_train['Return'] - combined_train['N225_Return'].fillna(0)
    combined_latest['Relative_Return'] = combined_latest['Return'] - combined_latest['N225_Return'].fillna(0)
    combined_train['Relative_Return_5d'] = combined_train['Return_5d'] - combined_train['N225_Return_5d'].fillna(0)
    combined_latest['Relative_Return_5d'] = combined_latest['Return_5d'] - combined_latest['N225_Return_5d'].fillna(0)
    combined_train['Relative_Growth'] = combined_train['Return'] - combined_train['Growth_Return'].fillna(0)
    combined_latest['Relative_Growth'] = combined_latest['Return'] - combined_latest['Growth_Return'].fillna(0)

    train_means = combined_train.mean(numeric_only=True)
    combined_train = combined_train.replace([np.inf, -np.inf], np.nan).fillna(train_means).fillna(0)
    combined_latest = combined_latest.replace([np.inf, -np.inf], np.nan).fillna(train_means).fillna(0)

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
                        sample_weights[idx_w] = 3.0
                        matched_misses += 1
                    else:
                        sample_weights[idx_w] = 1.5
            print(f"   ➔ 過去の失敗パターン {matched_misses} 件にペナルティ重み（x3.0）を適用しました。")

    features = [
        'Return', 'Return_3d', 'Return_5d', 'Return_20d', 'Return_60d',
        'Volatility', 'Vol_Regime', 'Volume_Change', 'Volume_Ratio',
        'SMA5_Dev', 'SMA25_Dev', 'SMA75_Dev', 'MACD_Hist', 'Range_60d_Pos', 'BB_Pos',
        'RSI', 'RSI_5', 'PER', 'PBR', 'ROE',
        'Sector_Code', 'Sector_Return', 'Past_Hit_Rate',
        'N225_Return', 'N225_Return_5d', 'N225_Trend', 'Growth_Return', 'Growth_Trend',
        'Relative_Return', 'Relative_Return_5d', 'Relative_Growth',
        'SP500_Return', 'USDJPY_Return', 'VIX_Level', 'US10Y_Return',
        'News_Sentiment'
    ]
    X_train, y_train = combined_train[features], combined_train['Target']
    
    print("\n第2段階: AI学習（過学習抑制・精密トリプルバリア法・押し目対応）実行中...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=250, learning_rate=0.015, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
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
    
    ranking['Target_Buy'] = ranking['Close'].round(0).astype(int)
    ranking['Take_Profit'] = (ranking['Close'] * 1.02).round(0).astype(int)
    ranking['Stop_Loss'] = (ranking['Close'] * 0.985).round(0).astype(int)
    
    ranking['AI_Win_Probability_Str'] = (ranking['AI_Win_Probability'] * 100).round(1).astype(str) + '%'
    ranking['Close_Str'] = ranking['Close'].round(0).astype(int).apply(lambda x: f"{x:,}円")
    ranking['Vol_Ratio_Str'] = ranking['Volume_Ratio'].round(1).astype(str) + '倍'
    ranking['Ret5d_Str'] = (ranking['Return_5d'] * 100).round(1).apply(lambda x: f"{x:+.1f}%")
    ranking['RSI_Str'] = ranking['RSI'].round(0).astype(int).astype(str)
    ranking['TP_Str'] = ranking['Take_Profit'].apply(lambda x: f"{x:,}円 (+2.0%)")
    ranking['SL_Str'] = ranking['Stop_Loss'].apply(lambda x: f"{x:,}円 (-1.5%)")

    tag = "【Dry-run 参考値】" if is_dry_run else "【AI厳選】"
    print("\n" + "="*110)
    print(f"🔥 {tag} 3,000円未満・精密トリプルバリア推奨トレードプラン (上位15選)")
    print("="*110)
    print(f"{'順位':<4} {'コード':<8} {'銘柄名':<16} {'業種':<12} {'現在値':<9} {'AI勝率':<7} {'5日騰落':<8} {'出来高倍':<8} {'利確目標(+2.0%)':<16} {'損切り(-1.5%)':<14}")
    print("-"*110)

    for idx, row in ranking.head(15).iterrows():
        name_short = row['Name'][:10]
        sec_short = row['Sector'][:8]
        print(f"{idx:<4} {row['Ticker']:<8} {name_short:<16} {sec_short:<12} {row['Close_Str']:<9} {row['AI_Win_Probability_Str']:<7} "
              f"{row['Ret5d_Str']:<8} {row['Vol_Ratio_Str']:<8} {row['TP_Str']:<16} {row['SL_Str']:<14}")
    print("="*110)

    save_cols = ['Ticker', 'Name', 'Sector', 'Close', 'AI_Win_Probability', 'Target_Buy', 'Take_Profit', 'Stop_Loss']
    
    if is_dry_run:
        ranking[save_cols].to_csv('ai_ranking_dry_run.csv', index=False)
        print("\n📁 [Dry-run] テスト結果を 'ai_ranking_dry_run.csv' に保存しました（本番ファイルは保護）。")
    else:
        ranking[save_cols].to_csv('ai_ranking_with_prices.csv', index=False)
        print("\n📁 本番最新ランキングを 'ai_ranking_with_prices.csv' に保存しました。")

    today_log = ranking[save_cols].head(15).copy()
    today_log['Date'] = today_str
    today_log['Actual_Close_Next'] = np.nan 
    today_log['Return_Pct'] = np.nan
    today_log['Max_Up_Pct'] = np.nan
    today_log['Max_Down_Pct'] = np.nan
    today_log['Is_Hit'] = np.nan

# =====================================================================
# 2. 過去の答え合わせ結果の表示（スクリーニング通過数に関係なく必ず実行）
# =====================================================================
hit_evaluated = history_df.dropna(subset=['Is_Hit']).copy() if not history_df.empty else pd.DataFrame()
target_ver_df = pd.DataFrame()
ver_title = ""

if len(hit_evaluated) > 0:
    if newly_verified_indices:
        target_ver_df = history_df.loc[newly_verified_indices]
        ver_title = "今回確定した予測の答え合わせ (精密トリプルバリア判定)"
    else:
        latest_ver_date = hit_evaluated['Date'].max()
        target_ver_df = hit_evaluated[hit_evaluated['Date'] == latest_ver_date]
        ver_title = f"直近検証日 ({latest_ver_date}) の答え合わせ (精密トリプルバリア判定)"

    print("\n" + "="*105)
    print(f"🎯 【{ver_title}】個別銘柄の勝敗結果")
    print("="*105)
    print(f"{'予測日':<11} {'コード':<8} {'銘柄名':<16} {'AI勝率':<7} {'基準終値':<9} {'最高到達':<9} {'最安到達':<9} {'判定'}")
    print("-"*105)
    for _, r in target_ver_df.iterrows():
        prob_s = f"{r['AI_Win_Probability']*100:.1f}%"
        base_s = f"{int(round(r['Close'])):,}円"
        up_s = f"{r.get('Max_Up_Pct', 0):+.2f}%" if pd.notna(r.get('Max_Up_Pct')) else "-"
        down_s = f"{r.get('Max_Down_Pct', 0):+.2f}%" if pd.notna(r.get('Max_Down_Pct')) else "-"
        mark = "⭕ 利確達成" if r['Is_Hit'] == 1 else "❌ 未達/損切"
        r_name = str(r.get('Name', name_map.get(r['Ticker'], '-')))[:10]
        print(f"{r['Date']:<11} {r['Ticker']:<8} {r_name:<16} {prob_s:<7} {base_s:<9} {up_s:<9} {down_s:<9} {mark}")
    print("-"*105)

    total_hits = int(hit_evaluated['Is_Hit'].sum())
    total_count = len(hit_evaluated)
    total_acc = (total_hits / total_count) * 100
    avg_ret = hit_evaluated['Return_Pct'].mean()
    print(f"📊 【累計AI運用実績】 通算: {total_hits}勝 {total_count - total_hits}敗 (計{total_count}件) | トリプルバリア達成率: {total_acc:.1f}% | 翌日終値平均騰落率: {avg_ret:+.2f}%")
    print("="*105)

# =====================================================================
# 3. Markdownレポート（GitHub Actionsサマリー）の生成
# =====================================================================
sentiment_desc = "強気 (リスクオン)" if current_news_sentiment > 0.1 else ("弱気 (リスクオフ・警戒)" if current_news_sentiment < -0.1 else "中立")
header_note = " ⚠️ [Dry-run テスト出力]" if is_dry_run else ""
md_lines = [f"# 📊 AI株価予測 検証レポート (最終更新: {today_str}){header_note}\n"]

if is_dry_run:
    md_lines.append("> ⚠️ **Dry-runモードで実行されました（16:00前の途中経過データ）。**  \n> 本番履歴ファイル（`ai_prediction_history.csv`）の更新はスキップされています。\n")

if not macro_df.empty:
    latest_m = macro_df.iloc[-1]
    md_lines.append("## 🌐 直近の市場環境（マクロ・ニュース・中小型株指標）")
    md_lines.append(f"- **ニュース・市場センチメント**: **{sentiment_desc}**（スコア: {current_news_sentiment:+.2f}）")
    md_lines.append(f"- **日経平均 前日比**: {latest_m['N225_Return']*100:+.2f}% （5日騰落: {latest_m['N225_Return_5d']*100:+.2f}% / 25日線乖離: {latest_m['N225_Trend']*100:+.2f}%）")
    md_lines.append(f"- **東証グロース250 前日比**: {latest_m['Growth_Return']*100:+.2f}% （25日線乖離: {latest_m['Growth_Trend']*100:+.2f}%）")
    md_lines.append(f"- **米国S&P500 前日比**: {latest_m['SP500_Return']*100:+.2f}% | **米10年債利回り変化**: {latest_m['US10Y_Return']*100:+.2f}%")
    md_lines.append(f"- **ドル円変動率**: {latest_m['USDJPY_Return']*100:+.2f}% | **VIX指数**: {latest_m['VIX_Level']:.1f}\n")

if len(hit_evaluated) > 0:
    md_lines.append("## 1. 累計パフォーマンス（精密トリプルバリア法：利確+2.0%/損切-1.5%）")
    md_lines.append(f"- **累計検証数**: {total_count} 銘柄（{total_hits}勝 {total_count - total_hits}敗）")
    md_lines.append(f"- **トリプルバリア達成率**: **{total_acc:.1f}%**")
    md_lines.append(f"- **翌日終値平均騰落率**: **{avg_ret:+.2f}%**\n")
    
    if not target_ver_df.empty:
        md_lines.append(f"## 2. {ver_title}")
        md_lines.append("| 予測日 | コード | 銘柄名 | 業種 | AI勝率 | 基準終値 | ザラ場最高 | ザラ場最安 | 判定 |")
        md_lines.append("| :--- | :--- | :--- | :--- | :---: | ---: | ---: | ---: | :---: |")
        for _, r in target_ver_df.iterrows():
            mark = "⭕ 利確達成" if r['Is_Hit'] == 1 else "❌ 未達/損切"
            r_name = r.get('Name', name_map.get(r['Ticker'], '-'))
            r_sec = r.get('Sector', sector_map.get(r['Ticker'], '-'))
            up_s = f"{r.get('Max_Up_Pct', 0):+.2f}%" if pd.notna(r.get('Max_Up_Pct')) else "-"
            down_s = f"{r.get('Max_Down_Pct', 0):+.2f}%" if pd.notna(r.get('Max_Down_Pct')) else "-"
            md_lines.append(f"| {r['Date']} | {r['Ticker']} | {r_name} | {r_sec} | {r['AI_Win_Probability']*100:.1f}% | {int(round(r['Close'])):,}円 | {up_s} | {down_s} | {mark} |")
        md_lines.append("\n")

md_lines.append(f"## 3. 本日 ({today_str}) のAI厳選ランキング")
if not ranking.empty:
    md_lines.append("| 順位 | コード | 銘柄名 | 業種 | 現在値 | AI勝率 | 5日騰落 | 出来高倍率 | RSI | 利確目標 (+2.0%) | 損切り (-1.5%) |")
    md_lines.append("| :---: | :--- | :--- | :--- | ---: | :---: | :---: | :---: | :---: | ---: | ---: |")
    for idx, r in ranking.head(15).iterrows():
        md_lines.append(f"| {idx} | {r['Ticker']} | {r['Name']} | {r['Sector']} | {r['Close_Str']} | {r['AI_Win_Probability_Str']} | {r['Ret5d_Str']} | {r['Vol_Ratio_Str']} | {r['RSI_Str']} | {r['TP_Str']} | {r['SL_Str']} |")
else:
    md_lines.append("> ⚠️ **本日の買い推奨銘柄: 0 件**  \n> 市場全体の急落・地合い悪化に伴い、条件を満たす銘柄が存在しませんでした。リスク管理のため本日は**新規エントリーを見送り（キャッシュ維持）**とします。\n")

report_content = "\n".join(md_lines)
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"\n📝 詳細検証レポートを '{report_file}' に出力しました。")

# 本番実行時のみ GitHub Step Summary に書き込む
if 'GITHUB_STEP_SUMMARY' in os.environ and not is_dry_run:
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
        f.write(report_content + "\n")

# =====================================================================
# 4. 履歴ファイル（ai_prediction_history.csv）の更新（本番実行時のみ）
# =====================================================================
if is_dry_run:
    print("\n🛑 [Dry-run Guard] 16:00前の実行であるため、ai_prediction_history.csv の更新を完全にスキップしました。")
else:
    if not today_log.empty:
        if not history_df.empty:
            history_df = history_df[history_df['Date'] != today_str]
            updated_history = pd.concat([history_df, today_log], ignore_index=True)
            updated_history.to_csv(history_file, index=False)
        else:
            today_log.to_csv(history_file, index=False)
        print(f"本日の予測データが {history_file} に正常に蓄積されました。")
    else:
        if not history_df.empty:
            history_df.to_csv(history_file, index=False)
        print("本日は新規推奨銘柄がないため、予測履歴の新規追加はありません（検証結果のみ更新保存完了）。")