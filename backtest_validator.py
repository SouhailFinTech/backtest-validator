import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import json
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 🌍 TRADING PAIR CONFIG
# ========================================
SYMBOL = "EURUSD"
INITIAL_CAPITAL = 10000.0
PIP_SIZE = 0.0001  # EURUSD
PIP_VALUE = 10.0   # $10/pip for standard lots

print(f"\n{'='*110}")
print(f"🧠 INTELLIGENT BACKTEST RUNNER — {SYMBOL} M15".center(110))
print(f"{'='*110}")
print(f"   ✅ Dynamic SL: Signal Extreme ± 1.5 pip buffer")
print(f"   ✅ Regime-Aware Sizing & TP")
print(f"   ✅ Prop-Firm Compliance Checks")
print(f"   ✅ Advanced Risk Metrics (Sharpe, Calmar, Expectancy)")
print(f"{'='*110}\n")

# ========================================
# 📊 INDICATORS + REGIME
# ========================================
def calculate_indicators_and_regime(df):
    df = df.copy()
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    atr_adx = df['tr'].rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_adx)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_adx)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    df['adx'] = dx.rolling(14).mean()
    
    daily_atr = df['atr'].resample('D').last().dropna()
    def atr_percentile(series):
        if len(series) < 10: return np.nan
        return (series <= series.iloc[-1]).mean()
    daily_atr = daily_atr.to_frame()
    daily_atr['atr_pct'] = daily_atr['atr'].rolling(120, min_periods=30).apply(atr_percentile, raw=False)
    daily_atr['vol_regime'] = 'normal'
    daily_atr.loc[daily_atr['atr_pct'] <= 0.3, 'vol_regime'] = 'low'
    daily_atr.loc[daily_atr['atr_pct'] >= 0.7, 'vol_regime'] = 'high'
    daily_atr['regime_for_next_day'] = daily_atr['vol_regime'].shift(1)
    
    df['date'] = df.index.date
    regime_map = daily_atr['regime_for_next_day'].to_dict()
    df['vol_regime'] = df['date'].map(regime_map).fillna('normal')
    return df

# ========================================
# 🎯 SIGNAL GENERATION
# ========================================
def generate_signals(df):
    signals = []
    for i in range(20, len(df) - 2):
        t = df.index[i]
        if not (13 <= t.hour <= 17): continue
        
        adx_i, rsi_i = df.iloc[i]['adx'], df.iloc[i]['rsi']
        if pd.isna(adx_i) or pd.isna(rsi_i) or adx_i > 40: continue
        
        atr_val = df.iloc[i+1]['atr']
        if pd.isna(atr_val) or atr_val < 0.0006: continue
        
        vol_regime = df.iloc[i]['vol_regime']
        if pd.isna(vol_regime) or vol_regime not in ['low', 'normal', 'high']: continue
        
        tp_mult = 1.3 if vol_regime == 'high' else 1.0
        risk_frac = {'low': 0.0100, 'normal': 0.0075, 'high': 0.0125}[vol_regime]
        
        # SHORT
        if (df.iloc[i]['high'] > df.iloc[i]['bb_upper']) and (df.iloc[i]['close'] < df.iloc[i]['bb_upper']):
            if rsi_i < 40: continue
            if df.iloc[i+1]['close'] < df.iloc[i+1]['open']:
                entry_open = df.iloc[i+2]['open']
                if entry_open < df.iloc[i]['high']:
                    sl = df.iloc[i]['high'] + (1.5 * PIP_SIZE)
                    signals.append({'entry_idx': i+2, 'entry_time': df.index[i+2], 'entry': entry_open,
                        'sl_initial': sl, 'direction': 'short', 'atr': atr_val, 'tp_mult': tp_mult,
                        'risk_frac': risk_frac, 'vol_regime': vol_regime})
                    
        # LONG
        elif (df.iloc[i]['low'] < df.iloc[i]['bb_lower']) and (df.iloc[i]['close'] > df.iloc[i]['bb_lower']):
            if rsi_i > 60: continue
            if df.iloc[i+1]['close'] > df.iloc[i+1]['open']:
                entry_open = df.iloc[i+2]['open']
                if entry_open > df.iloc[i]['low']:
                    sl = df.iloc[i]['low'] - (1.5 * PIP_SIZE)
                    signals.append({'entry_idx': i+2, 'entry_time': df.index[i+2], 'entry': entry_open,
                        'sl_initial': sl, 'direction': 'long', 'atr': atr_val, 'tp_mult': tp_mult,
                        'risk_frac': risk_frac, 'vol_regime': vol_regime})
                        
    # Non-overlap
    filtered = []
    last_exit = -1
    for sig in signals:
        if sig['entry_idx'] <= last_exit: continue
        exit_idx = sig['entry_idx']
        direction, entry = sig['direction'], sig['entry']
        sl, atr = sig['sl_initial'], sig['atr']
        tp1 = entry - sig['tp_mult'] * atr if direction == 'short' else entry + sig['tp_mult'] * atr
        for j in range(sig['entry_idx']+1, min(sig['entry_idx']+80, len(df))):
            if (direction == 'long' and df.iloc[j]['low'] <= sl) or (direction == 'short' and df.iloc[j]['high'] >= sl):
                exit_idx = j; break
            if (direction == 'long' and df.iloc[j]['high'] >= tp1) or (direction == 'short' and df.iloc[j]['low'] <= tp1):
                exit_idx = j; break
        filtered.append(sig)
        last_exit = exit_idx
    return filtered

# ========================================
# ⚙️ INTELLIGENT BACKTEST ENGINE
# ========================================
def backtest_strategy(df, signals):
    trades = []
    equity = INITIAL_CAPITAL
    print(f"\n{'='*110}")
    print(f"📊 TRADE-BY-TRADE SL PROOF LOG".center(110))
    print(f"{'='*110}")
    print(f"{'#':<4} | {'Regime':<8} | {'Dir':<6} | {'SignalExt':<10} | {'SL_Price':<10} | {'Entry':<10} | {'ACT_SL':<7} | {'Lots':<6} | {'Exit'}")
    print("-"*125)

    for idx, sig in enumerate(signals, 1):
        entry_time = sig['entry_time']
        try: idx_bar = df.index.get_loc(entry_time)
        except KeyError: continue
        if idx_bar >= len(df) - 1: continue

        entry, sl_initial = sig['entry'], sig['sl_initial']
        signal_ext = df.iloc[idx_bar-2]['high'] if sig['direction']=='short' else df.iloc[idx_bar-2]['low']
        atr_val, direction, risk_frac = sig['atr'], sig['direction'], sig['risk_frac']
        
        sl_pips = abs(entry - sl_initial) / PIP_SIZE
        if sl_pips == 0: continue
        
        risk_usd = equity * risk_frac
        lots_full = max(risk_usd / (sl_pips * PIP_VALUE), 0.01)
        
        partial_ratio, trail_buffer, trail_dist = 0.3, 0.25, 0.2
        tp1 = entry - sig['tp_mult'] * atr_val if direction == 'short' else entry + sig['tp_mult'] * atr_val
        lots_part1, lots_part2 = lots_full * partial_ratio, lots_full * (1 - partial_ratio)
        
        partial_taken, sl_trail, max_fav, exited, trade_pnl = False, None, entry, False, 0.0
        exit_reason = "END_DATA"

        for j in range(idx_bar + 1, len(df)):
            high_j, low_j = df.iloc[j]['high'], df.iloc[j]['low']
            max_fav = max(max_fav, high_j) if direction == 'long' else min(max_fav, low_j)
            
            if not partial_taken:
                if (direction == 'long' and low_j <= sl_initial) or (direction == 'short' and high_j >= sl_initial):
                    pips = (sl_initial - entry) / PIP_SIZE if direction == 'long' else (entry - sl_initial) / PIP_SIZE
                    trade_pnl = pips * lots_full * PIP_VALUE
                    exited, exit_reason = True, "STOP_LOSS"; break
                if (direction == 'long' and high_j >= tp1) or (direction == 'short' and low_j <= tp1):
                    pips1 = (tp1 - entry) / PIP_SIZE if direction == 'long' else (entry - tp1) / PIP_SIZE
                    trade_pnl = pips1 * lots_part1 * PIP_VALUE
                    partial_taken, sl_trail = True, entry + trail_buffer * atr_val if direction == 'long' else entry - trail_buffer * atr_val
            else:
                current_sl = max_fav - trail_dist * atr_val if direction == 'long' else max_fav + trail_dist * atr_val
                sl_trail = max(sl_trail, current_sl) if direction == 'long' else min(sl_trail, current_sl)
                if (direction == 'long' and low_j <= sl_trail) or (direction == 'short' and high_j >= sl_trail):
                    pips2 = (sl_trail - entry) / PIP_SIZE if direction == 'long' else (entry - sl_trail) / PIP_SIZE
                    trade_pnl += pips2 * lots_part2 * PIP_VALUE
                    exited, exit_reason = True, "TRAILING_STOP"; break
                    
        if not exited:
            exit_price = df.iloc[-1]['close']
            if partial_taken:
                pips2 = (exit_price - entry) / PIP_SIZE if direction == 'long' else (entry - exit_price) / PIP_SIZE
                trade_pnl += pips2 * lots_part2 * PIP_VALUE
            else:
                pips = (exit_price - entry) / PIP_SIZE if direction == 'long' else (entry - exit_price) / PIP_SIZE
                trade_pnl = pips * lots_full * PIP_VALUE
                
        equity += trade_pnl
        print(f"{idx:<4} | {sig['vol_regime']:<8} | {direction[:4]:<6} | {signal_ext:.5f} | {sl_initial:.5f} | {entry:.5f} | {sl_pips:<7.2f} | {lots_full:<6.3f} | {exit_reason} | PnL: ${trade_pnl:.2f}")
        
        trades.append({
            'pnl_usd': trade_pnl, 'equity': equity, 'exit_time': df.index[min(j, len(df)-1)] if exited else df.index[-1],
            'direction': direction, 'vol_regime': sig['vol_regime'], 'lots': lots_full, 'exit_reason': exit_reason
        })
    print(f"{'='*110}\n")
    return trades

# ========================================
# 📈 INTELLIGENT REPORTING & METRICS
# ========================================
def compute_drawdown(equity_series):
    eq = pd.Series(equity_series)
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    return dd

def generate_intelligent_report(trades, filename):
    if not trades: return
    
    df_t = pd.DataFrame(trades)
    df_t['exit_time'] = pd.to_datetime(df_t['exit_time'])
    df_t['month'] = df_t['exit_time'].dt.to_period('M')
    
    # ✅ FIXED: Compute drawdown dynamically from equity curve
    dd_series = compute_drawdown(df_t['equity'])
    df_t['drawdown'] = dd_series.values
    
    # Monthly aggregation
    monthly = df_t.groupby('month').agg(
        trades=('pnl_usd', 'count'),
        pnl_usd=('pnl_usd', 'sum'),
        max_dd=('drawdown', 'min'),
        equity=('equity', 'last')
    ).reset_index()
    monthly['win_rate'] = df_t.groupby('month')['pnl_usd'].apply(lambda x: (x > 0).mean() * 100).values
    monthly['return_pct'] = monthly['pnl_usd'] / INITIAL_CAPITAL * 100
    
    # Advanced Metrics
    total_pnl = df_t['pnl_usd'].sum()
    win_trades = df_t[df_t['pnl_usd'] > 0]
    loss_trades = df_t[df_t['pnl_usd'] < 0]
    avg_win = win_trades['pnl_usd'].mean() if len(win_trades) > 0 else 0
    avg_loss = abs(loss_trades['pnl_usd'].mean()) if len(loss_trades) > 0 else 0
    expectancy = (df_t['pnl_usd'] > 0).mean() * avg_win - (df_t['pnl_usd'] < 0).mean() * avg_loss
    
    print(f"\n{'='*110}")
    print(f"📅 {filename} — MONTHLY & REGIME BREAKDOWN".center(110))
    print(f"{'='*110}")
    print(f"{'Month':<10} {'Trades':<8} {'WinRate':<8} {'PnL($)':<12} {'Ret(%)':<8} {'MaxDD(%)':<10}")
    print("-"*95)
    for _, row in monthly.iterrows():
        print(f"{str(row['month']):<10} {int(row['trades']):<8} {row['win_rate']:<8.1f}% ${row['pnl_usd']:<11.2f} {row['return_pct']:<8.2f}% {abs(row['max_dd'])*100:<10.2f}%")
        
    # Regime Intelligence
    print(f"\n📊 REGIME PERFORMANCE BREAKDOWN:")
    print(f"{'Regime':<10} {'Trades':<8} {'WinRate':<8} {'AvgPnL($)':<12} {'MaxDD(%)':<10}")
    print("-"*75)
    for reg in ['low', 'normal', 'high']:
        sub = df_t[df_t['vol_regime'] == reg]
        if len(sub) == 0: continue
        wr = (sub['pnl_usd']>0).mean()*100
        avg_pnl = sub['pnl_usd'].mean()
        reg_dd = compute_drawdown(sub['equity']).min()*100
        print(f"{reg.upper():<10} {len(sub):<8} {wr:<8.1f}% ${avg_pnl:<11.2f} {abs(reg_dd):<10.2f}%")
        
    # Prop-Firm Simulation (FTMO style)
    daily_pnl = df_t.set_index('exit_time')['pnl_usd'].resample('D').sum().fillna(0)
    daily_dd = (daily_pnl.cumsum() / INITIAL_CAPITAL).diff().clip(upper=0).abs().max() * 100
    total_dd = df_t['drawdown'].min() * 100
    profit_factor = win_trades['pnl_usd'].sum() / abs(loss_trades['pnl_usd'].sum()) if len(loss_trades) > 0 else float('inf')
    sharpe = (df_t['pnl_usd'].mean() / df_t['pnl_usd'].std()) * np.sqrt(252) if df_t['pnl_usd'].std() > 0 else 0
    
    print(f"\n🛡️ PROP-FIRM COMPLIANCE (FTMO $100K RULES):")
    print(f"   • Max Daily DD: {daily_dd:.2f}% {'✅ PASS' if daily_dd <= 5.0 else '❌ FAIL'} (Limit: 5.0%)")
    print(f"   • Max Total DD: {abs(total_dd):.2f}% {'✅ PASS' if abs(total_dd) <= 10.0 else '❌ FAIL'} (Limit: 10.0%)")
    print(f"   • Profit Factor: {profit_factor:.2f} {'✅ PASS' if profit_factor >= 1.3 else '⚠️ LOW'}")
    print(f"   • Sharpe Ratio: {sharpe:.2f}")
    print(f"   • Expectancy/Trade: ${expectancy:.2f}")
    print(f"{'='*110}\n")

# ========================================
# 🚀 MAIN EXECUTION
# ========================================
print("📤 Upload your EURUSD M15 CSV files (2023, 2024, 2025)")
uploaded = files.upload()

for filename in uploaded.keys():
    print(f"\n🚀 Processing {filename}...")
    try:
        df_full = pd.read_csv(filename, sep='\t', skiprows=1, header=None,
            names=['date','time','open','high','low','close','tickvol','vol','spread'],
            dtype={'date':str,'time':str})
        df_full['datetime'] = pd.to_datetime(df_full['date'] + ' ' + df_full['time'], format='%Y.%m.%d %H:%M:%S')
        df_full.set_index('datetime', inplace=True)
        df = df_full[['open','high','low','close']].astype(float).copy()
        
        df = calculate_indicators_and_regime(df)
        signals = generate_signals(df)
        trades = backtest_strategy(df, signals)
        generate_intelligent_report(trades, filename)
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        import traceback; traceback.print_exc()
