import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '')))

from src.functions import *
import vectorbt as vbt
import pandas as pd
import numpy as np
import yaml
from typing import Optional
from datetime import datetime, time

if __name__ == '__main__':
    with open('config.yaml', 'r') as fp:
        config = yaml.safe_load(fp)

    """1. Import Config"""
    DATA_FP = config['DATA_FP']
    DATA_TF = config['DATA_TF']
    DATA_1MIN_FP = config['DATA_1MIN_FP']
    BACKTESTING_DATES_START = config['BACKTESTING_DATES']['START']
    BACKTESTING_DATES_END = config['BACKTESTING_DATES']['END']

    df_custom = pd.read_csv(DATA_FP)

    df_1min = pd.read_csv(DATA_1MIN_FP)
    
    df_custom, df_1min = process_data(df_custom, df_1min, BACKTESTING_DATES_START, BACKTESTING_DATES_END)

    optimization_results = optimize_strategy(df_custom, df_1min, config)

    optimization_results_df = pd.DataFrame(optimization_results)
    output_dir = '/output'
    os.makedirs(output_dir, exist_ok=True)
    optimization_results_df.to_csv(f'{output_dir}/optimization_results.csv')
    # optimization_results_df = pd.read_csv('optimization/output/optimization_results.csv')

    metrics = [
        'Profit Factor',
        'Sharpe Ratio',
        'Total Trades',
        'Total Return [%]',
        'Max Drawdown [%]',
        'Win Rate [%]',   
    ]
    
    successful = 0
    for metric in metrics:
        if create_heatmap(optimization_results_df, metric):
            successful += 1
    

    # save_optimization_results(optimization_results)

