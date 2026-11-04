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
df_stocks = df_jpx[df_jpx['市場・商品区分'].str.contains('プライム|スタンダード|グロース', na=False)].copy()

# 4. 証券コード・銘柄名・業種マッピングの作成
df_stocks['Ticker'] = df_stocks['コード'].astype(str).str.strip() + ".T"
valid_tickers = df_stocks['Ticker'].tolist()

name_map = dict(zip(df_stocks['Ticker'], df_stocks['銘柄名'].astype(str).str.strip()))
if '33業種区分' in df_stocks.columns:
    sector_map = dict(zip(df_stocks['Ticker'], df_stocks['33業種区分'].astype(str).str.strip()))
else:
    sector_map = {t: '-' for t in valid_tickers}

# AI学習用に業種を数値コード化
unique_sectors = sorted(list(set(sector_map.values())))
sector_code_map = {sec: i for i, sec in enumerate(unique_sectors)}

print(f"抽出完了: 全上場企業 {len(valid_tickers)} 銘柄をスクリーニング対象に設定しました。")

# --- 事前処理1: 過去の予測履歴の検証（AI学習にフィードバックするため最初に実行） ---
history_file = 'ai_prediction_history.csv'
report_file = 'prediction_report.md'
today_str = datetime.now().strftime('%Y-%m-%d')
newly_verified_indices = []
history_df = pd.DataFrame()

if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)
    if 'Return_Pct' not in history_df.columns:
        history_df['Return_Pct'] = np.nan
    # 過去履歴に銘柄名や業種がない場合はマッピングで補完
    if 'Name' not in history_df.columns:
        history_df['Name'] = history_df['Ticker'].map(name_map).fillna('-')
    if 'Sector' not in history_df.columns:
        history_df['Sector'] = history_df['Ticker'].map(sector_map).fillna('-')
        
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

# --- 事前処理2: 市場全体（マクロ指標）データの取得 ---
print("\n🌐 市場全体指標（日経平均・S&P500・ドル円・VIX）を取得中...")
macro_tickers = ['^N225', '^GSPC', 'JPY=X', '^VIX']
macro_df = pd.DataFrame()

for attempt in range(3):
    try:
        m_data = yf.download(macro_tickers, period="2y", interval="1d", progress=False, group_by='ticker')
        
        def extract_close(ticker_sym):
            if ticker_sym in m_data.columns.levels[0]:
                return m_data[ticker_sym]['Close']
            return pd.Series(dtype=float)

        n225 = extract_close('^N225')
        sp500 = extract_close('^GSPC')
        usdjpy = extract_close('JPY=X')
        vix = extract_close('^VIX')

        macro_raw = pd.DataFrame({
            'N225': n225,
            'SP500': sp500,
            'USDJPY': usdjpy,
            'VIX': vix
        }).sort_index().ffill()

        macro_df['N225_Return'] = macro_raw['N225'].pct_change()
        macro_df['N225_Trend'] = (macro_raw['N225'] / macro_raw['N225'].rolling(25).mean()) - 1
        macro_df['SP500_Return'] = macro_raw['SP500'].pct_change()
        macro_df['USDJPY_Return'] = macro_raw['USDJPY'].pct_change()
        macro_df['VIX_Level'] = macro_raw['VIX']
        macro_df['Date_Str'] = pd.to_datetime(macro_df.index).strftime('%Y-%m-%d')
        macro_df = macro_df.replace([np.inf, -np.inf], np.nan).dropna()
        print("   ➔ マクロ指標の取得完了。")
        break
    except Exception:
        time.sleep(3 * (attempt + 1))

time.sleep(2.0)

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
    
    # 業種コードの付与
    combined_train['Sector_Code'] = combined_train['Ticker'].map(sector_map).map(sector_code_map).fillna(-1)
    combined_latest['Sector_Code'] = combined_latest['Ticker'].map(sector_map).map(sector_code_map).fillna(-1)

    # --- マクロ指標の結合と「対日経平均の相対的強さ」の算出 ---
    macro_cols = ['N225_Return', 'N225_Trend', 'SP500_Return', 'USDJPY_Return', 'VIX_Level']
    if not macro_df.empty:
        combined_train = combined_train.merge(macro_df[['Date_Str'] + macro_cols], on='Date_Str', how='left')
        combined_latest = combined_latest.merge(macro_df[['Date_Str'] + macro_cols], on='Date_Str', how='left')
    else:
        for col in macro_cols:
            combined_train[col] = 0.0
            combined_latest[col] = 0.0

    combined_train['Relative_Return'] = combined_train['Return'] - combined_train['N225_Return'].fillna(0)
    combined_latest['Relative_Return'] = combined_latest['Return'] - combined_latest['N225_Return'].fillna(0)

    train_means = combined_train.mean(numeric_only=True)
    combined_train = combined_train.fillna(train_means).fillna(0)
    combined_latest = combined_latest.fillna(train_means).fillna(0)

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

    features = [
        'Return', 'Volatility', 'Volume_Change', 'RSI', 'PER', 'PBR', 'Sector_Code', 'Past_Hit_Rate',
        'N225_Return', 'N225_Trend', 'Relative_Return', 'SP500_Return', 'USDJPY_Return', 'VIX_Level'
    ]
    X_train, y_train = combined_train[features], combined_train['Target']
    
    print("\n第3段階: AI学習（マクロ地合い＋業種＋過去検証フィードバック統合）と勝率ランキング生成中...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.01, max_depth=5, 
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    X_latest = combined_latest[features]
    combined_latest['AI_Win_Probability'] = lgb_model.predict_proba(X_latest)[:, 1]
    
    ranking = combined_latest[['Ticker', 'Close', 'AI_Win_Probability', 'PER', 'PBR']].copy()
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
    ranking['Buy_Str'] = ranking['Target_Buy'].apply(lambda x: f"{x:,}円")
    ranking['TP_Str'] = ranking['Take_Profit'].apply(lambda x: f"{x:,}円 (+5%)")
    ranking['SL_Str'] = ranking['Stop_Loss'].apply(lambda x: f"{x:,}円 (-2%)")
    ranking['PER_Str'] = ranking['PER'].round(1)
    ranking['PBR_Str'] = ranking['PBR'].round(2)

    print("\n" + "="*105)
    print("🔥 【AI厳選】3,000円未満・上昇有望銘柄トレードプラン (上位15選)")
    print("="*105)
    print(f"{'順位':<4} {'コード':<8} {'銘柄名':<16} {'業種':<12} {'現在値':<9} {'AI勝率':<7} {'PER':<6} {'PBR':<5} {'利確目標(+5%)':<14} {'損切り(-2%)':<12}")
    print("-"*105)

    for idx, row in ranking.head(15).iterrows():
        name_short = row['Name'][:10]
        sec_short = row['Sector'][:8]
        print(f"{idx:<4} {row['Ticker']:<8} {name_short:<16} {sec_short:<12} {row['Close_Str']:<9} {row['AI_Win_Probability_Str']:<7} "
              f"{row['PER_Str']:<6} {row['PBR_Str']:<5} {row['TP_Str']:<14} {row['SL_Str']:<12}")
    print("="*105)

    # 0. 最新ランキングのCSV保存
    save_cols = ['Ticker', 'Name', 'Sector', 'Close', 'AI_Win_Probability', 'PER', 'PBR', 'Target_Buy', 'Take_Profit', 'Stop_Loss']
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

    # Markdownレポートの生成（メール本文・GitHub表示用）
    md_lines = [f"# 📊 AI株価予測 検証レポート (最終更新: {today_str})\n"]
    
    if not macro_df.empty:
        latest_m = macro_df.iloc[-1]
        md_lines.append("## 🌐 直近の市場環境（マクロ指標）")
        md_lines.append(f"- **日経平均 前日比**: {latest_m['N225_Return']*100:+.2f}% （25日線乖離: {latest_m['N225_Trend']*100:+.2f}%）")
        md_lines.append(f"- **米国S&P500 前日比**: {latest_m['SP500_Return']*100:+.2f}%")
        md_lines.append(f"- **ドル円変動率**: {latest_m['USDJPY_Return']*100:+.2f}% | **VIX指数**: {latest_m['VIX_Level']:.1f}\n")

    if len(hit_evaluated) > 0:
        md_lines.append("## 1. 累計パフォーマンス（マクロ指標＆フィードバック学習適用済）")
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
    md_lines.append("| 順位 | コード | 銘柄名 | 業種 | 現在値 | AI勝率 | PER | PBR | 利確目標 (+5%) | 損切り (-2%) |")
    md_lines.append("| :---: | :--- | :--- | :--- | ---: | :---: | ---: | ---: | ---: | ---: |")
    for idx, r in ranking.head(15).iterrows():
        md_lines.append(f"| {idx} | {r['Ticker']} | {r['Name']} | {r['Sector']} | {r['Close_Str']} | {r['AI_Win_Probability_Str']} | {r['PER_Str']} | {r['PBR_Str']} | {r['TP_Str']} | {r['SL_Str']} |")

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