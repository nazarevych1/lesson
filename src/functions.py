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
from typing import Union, Optional


def save_backtesting_results(pf):
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
