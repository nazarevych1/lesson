import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.io as pio
import pandas as pd
import vectorbt as vbt
import numpy as np
from datetime import time, timedelta, datetime
import time as _time
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union, Optional, Tuple, List
import seaborn as sns
import matplotlib.pyplot as plt


def save_backtesting_results(pf: vbt.Portfolio):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    stats = pf.stats()
    stats_df = stats.to_frame()

    with PdfPages(f"{output_dir}/portfolio_report.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(8.5, len(stats_df) * 0.4))
        ax.axis("off")
        table = ax.table(
            cellText=stats_df.values,
            colLabels=stats_df.columns,
            rowLabels=stats_df.index,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    pf.trades.records_readable.to_csv(f'{output_dir}/trades.csv')

def create_heatmap(results_df, metric_name, output_dir='optimization_output', figsize=(30, 30)):
    try:
        print(f"\n>>> Creating heatmap for: {metric_name}")
        
        # Check if metric exists in dataframe
        if metric_name not in results_df.columns:
            print(f"WARNING: '{metric_name}' not found in results. Skipping...")
            return False
        
        # Create pivot table
        heatmap_data = results_df.pivot_table(
            index='WICK_RATIO', 
            columns='OP_WICK_RATIO', 
            values=metric_name, 
            aggfunc='mean'
        )
        
        # Check if heatmap has data
        if heatmap_data.empty:
            print(f"WARNING: No data for '{metric_name}'. Skipping...")
            return False
        
        # Check for all NaN values
        if heatmap_data.isna().all().all():
            print(f"WARNING: All values are NaN for '{metric_name}'. Skipping...")
            return False
        
        # Print statistics
        valid_values = heatmap_data.stack().dropna()
        print(f"  Valid values: {len(valid_values)}")
        print(f"  Min: {valid_values.min():.2f}, Max: {valid_values.max():.2f}, Mean: {valid_values.mean():.2f}")
        
        # Create and save heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            center=0,
            cbar_kws={'label': metric_name}
        )
        plt.title(f"{metric_name} Heatmap")
        plt.xlabel("WICK_RATIO")
        plt.ylabel("OP_WICK_RATIO")
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        filepath = f"{output_dir}/{metric_name}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filepath}")
        return True
        
    except Exception as e:
        print(f"ERROR creating heatmap for '{metric_name}': {e}")
        plt.close()
        return False


def backtest_strategy(df_custom: pd.DataFrame, df_1min: pd.DataFrame, config: dict) -> vbt.Portfolio:
    df_custom = df_custom.copy()
    df_1min = df_1min.copy()

    FEES = config['FEES']
    SLIPPAGE = config['SLIPPAGE']
    SIZE = config['SIZE']
    SIZE_TYPE = config['SIZE_TYPE']
    INIT_BALANCE = config['INIT_BALANCE']
    WICK_RATIO = config['WICK_RATIO']
    OP_WICK_RATIO = config['OP_WICK_RATIO']
    RR = config['RR']
    TRADING_TIME_START = config['TRADING_TIME']['START']
    TRADING_TIME_END = config['TRADING_TIME']['END']

    bullish_entry_mask = (
    (df_custom['LWL'] >= WICK_RATIO * df_custom['Body']) &
    (df_custom['Direction'] == 1) &
    (df_custom['UWL'] * OP_WICK_RATIO <= df_custom['LWL'])
    )

    bearish_entry_mask = (
        (df_custom['UWL'] >= WICK_RATIO * df_custom['Body']) &
        (df_custom['Direction'] == -1) &
        (df_custom['LWL'] * OP_WICK_RATIO <= df_custom['UWL'])
    )

    df_custom['Bullish Entry'] = bullish_entry_mask
    df_custom['Bearish Entry'] = bearish_entry_mask

    df_custom = df_custom.drop(columns=['UWL', 'LWL', 'Body', 'Direction'])
    df_custom.loc[df_custom['Bullish Entry'] == True, 'SL'] = df_custom['Low']
    df_custom.loc[df_custom['Bearish Entry'] == True, 'SL'] = df_custom['High']

    df_custom.loc[df_custom['Bullish Entry'] == True, 'TP'] = df_custom['Close'] + (df_custom['Close'] - df_custom['Low']) * RR
    df_custom.loc[df_custom['Bearish Entry'] == True, 'TP'] = df_custom['Close'] - (df_custom['High'] - df_custom['Close']) * RR

    """4. Reindex to 1min timeframe. """
    df_1min['Bullish Entry'] = df_custom['Bullish Entry'].shift(1).reindex(df_1min.index, method='ffill').copy()
    df_1min['Bearish Entry'] = df_custom['Bearish Entry'].shift(1).reindex(df_1min.index, method='ffill').copy()
    df_1min['SL'] = df_custom['SL'].shift(1).reindex(df_1min.index, method='ffill').copy()
    df_1min['TP'] = df_custom['TP'].shift(1).reindex(df_1min.index, method='ffill').copy()
    df_1min['Date and Hour'] = df_1min.index.floor('h')
    df_1min = df_1min.dropna()

    """5. Main Loop"""

    index_arr_1min = df_1min.index.to_numpy()
    bullish_entry_arr_1min = df_1min['Bullish Entry'].to_numpy()
    bearish_entry_arr_1min = df_1min['Bearish Entry'].to_numpy()
    sl_arr_1min = df_1min['SL'].to_numpy()
    tp_arr_1min = df_1min['TP'].to_numpy()
    high_arr = df_1min['High'].to_numpy()
    low_arr = df_1min['Low'].to_numpy()
    close_arr = df_1min['Close'].to_numpy()
    date_and_hour_arr = df_1min['Date and Hour'].to_numpy()
    candle_time_arr = df_1min.index.time

    price_arr = np.full(len(index_arr_1min), np.nan)

    bullish_entries_arr = np.full(len(index_arr_1min), False)
    bearish_entries_arr = np.full(len(index_arr_1min), False)

    bullish_exit_arr = np.full(len(index_arr_1min), False)
    bearish_exit_arr = np.full(len(index_arr_1min), False)

    opened_trade_direction = None # bullish/bearish/None
    current_sl = None; current_tp = None # float/None

    opened_date_and_hour = {}

    trading_start_time = time(hour=int(TRADING_TIME_START.split(':')[0]), minute=int(TRADING_TIME_START.split(':')[1]))
    trading_end_time = time(hour=int(TRADING_TIME_END.split(':')[0]), minute=int(TRADING_TIME_END.split(':')[1]))

    for i in range(len(index_arr_1min)):
        if opened_trade_direction is None: # if trade is not opened, check conditions to open
            date_and_hour = date_and_hour_arr[i]
            bullish_signal = bullish_entry_arr_1min[i]
            bearish_signal = bearish_entry_arr_1min[i]
            
            if (not bullish_signal and not bearish_signal) or opened_date_and_hour.get(date_and_hour) == True: # no entry
                continue

            if not (trading_start_time <= candle_time_arr[i] <= trading_end_time):
                continue

            opened_trade_direction = 'bullish' if bullish_signal else 'bearish' # bullish/bearish entry
            bullish_entries_arr[i] = True if bullish_signal else False
            bearish_entries_arr[i] = True if bearish_signal else False
            price_arr[i] = close_arr[i]
            current_sl, current_tp = sl_arr_1min[i], tp_arr_1min[i]
            opened_date_and_hour[date_and_hour] = True
            # print(index_arr_1min[i], f'{opened_trade_direction} trade is opened on price: {price_arr[i]}, sl: {current_sl}, tp: {current_tp}')

            continue

        elif opened_trade_direction == 'bullish':
            close_ = False
            closing_price = None

            if low_arr[i] <= current_sl: # sl hit
                close_ = True
                closing_price = current_sl

            elif high_arr[i] >= current_tp: # tp hit
                close_ = True
                closing_price = current_tp

            
            if close_:
                price_arr[i] = closing_price
                # print(index_arr_1min[i], f'{opened_trade_direction} trade is closed on price: {price_arr[i]}')
                bullish_exit_arr[i], opened_trade_direction, current_sl, current_tp = True, None, None, None
                

        elif opened_trade_direction == 'bearish':
            close_ = False
            closing_price = None

            if high_arr[i] >= current_sl: # sl hit
                close_ = True
                closing_price = current_sl

            elif low_arr[i] <= current_tp: # tp hit
                close_ = True
                closing_price = current_tp

            
            if close_:
                price_arr[i] = closing_price
                # print(index_arr_1min[i], f'{opened_trade_direction} trade is closed on price: {price_arr[i]}')
                bearish_exit_arr[i], opened_trade_direction, current_sl, current_tp = True, None, None, None


    """5. Backtest with Vectorbt"""
    pf = vbt.Portfolio.from_signals(
        close=df_1min['Close'],
        price=price_arr,
        entries=bullish_entries_arr,
        exits=bullish_exit_arr,
        short_entries=bearish_entries_arr,
        short_exits=bearish_exit_arr,
        size=SIZE,
        size_type=SIZE_TYPE,
        fees=FEES,
        slippage=SLIPPAGE,
        init_cash=INIT_BALANCE,
        freq='1min'
    )

    return pf


def process_data(df_custom: pd.DataFrame, df_1min: pd.DataFrame, backtesting_dates_start: str, backtesting_dates_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_custom = df_custom.copy()
    df_1min = df_1min.copy()

    df_custom = df_custom.set_index('Time')
    df_custom.index = pd.to_datetime(df_custom.index)
    df_custom = df_custom[backtesting_dates_start: backtesting_dates_end]

    df_1min = df_1min.set_index('Time')
    df_1min.index = pd.to_datetime(df_1min.index)

    df_custom['Direction'] = -1
    df_custom.loc[df_custom['Close'] > df_custom['Open'], 'Direction'] = 1
    df_custom.loc[df_custom['Direction'] == 1, 'UWL'] = df_custom['High'] - df_custom['Close']
    df_custom.loc[df_custom['Direction'] == -1, 'UWL'] = df_custom['High'] - df_custom['Open']
    df_custom.loc[df_custom['Direction'] == 1, 'LWL'] = df_custom['Open'] - df_custom['Low']
    df_custom.loc[df_custom['Direction'] == -1, 'LWL'] = df_custom['Close'] - df_custom['Low']
    df_custom['Body'] = (df_custom['Close'] - df_custom['Open']).abs()

    return df_custom, df_1min


def optimize_strategy(df_custom: pd.DataFrame, df_1min: pd.DataFrame, config: dict) -> List[dict]:
    wick_ratio_combs = np.arange(
        config['WICK_RATIO']['START'],
        config['WICK_RATIO']['END'],
        config['WICK_RATIO']['STEP'],
    )

    op_wick_ratio_combs = np.arange(
        config['OP_WICK_RATIO']['START'],
        config['OP_WICK_RATIO']['END'],
        config['OP_WICK_RATIO']['STEP'],
    )

    results = []

    for wick_ratio in tqdm(wick_ratio_combs, desc='Wick Ratio Combinations'):
        wick_ratio = round(wick_ratio, 2)

        for op_wick_ratio in tqdm(op_wick_ratio_combs, leave=False, desc='Op Wick Ratio Combinations'):
            op_wick_ratio = round(op_wick_ratio, 2)

            udpated_config = config.copy()
            udpated_config.update({"WICK_RATIO": wick_ratio, "OP_WICK_RATIO": op_wick_ratio})
            
            pf = backtest_strategy(df_custom, df_1min, udpated_config)
            stats = pf.stats().to_dict()
            stats.update({'WICK_RATIO': wick_ratio, 'OP_WICK_RATIO': op_wick_ratio})
            results.append(stats)

    return results

