
import pandas as pd
from lightweight_charts import Chart
import numpy as np
import pandas_ta as ta

np.random.seed(42)
if __name__ == '__main__':
    chart = Chart(inner_height=0.5, height=800)
    chart.legend(visible=True)

    df = pd.read_csv('ohlcv.csv')
    df[['macd', 'macd_hist', 'macd_signal']] = ta.macd(df.close, 20, 50, 10)
    df = df.tail(120).reset_index(drop=True)

    volume_df = df[['date', 'volume']]
    volume_df = pd.concat([volume_df, pd.Series(np.random.choice([-1, 1], size=len(volume_df)), name='factor')], axis=1)
    volume_df['pnl_value'] = volume_df['volume'] * volume_df['factor'].div(1_000)

    volume_df['color'] = np.where(volume_df.pnl_value >= 0, 'rgba(0,114,54,255)', 'rgba(201,29,35,255)')

    vol_chart = chart.create_subchart(position='bottom', width=1, height=0.25, sync=True)
    vol_chart = vol_chart.create_histogram(name='pnl_value')

    chart.set(df[['date', 'open', 'high', 'low', 'close', 'volume']])
    vol_chart.set(volume_df[['date', 'pnl_value', 'color']])
    vol_chart.horizontal_line(85_000.0, text='Outlier Wins')
    vol_chart.horizontal_line(-85_000.0, text='Outlier Losses')

    macd_chart = chart.create_subchart(position='bottom', width=1, height=0.25, sync=True)
    macd_chart.legend(visible=True)

    macd_line = macd_chart.create_line(name='macd', width=1)
    macd_signal = macd_chart.create_line(name='macd_signal', width=1, style='dotted')
    macd_hist = macd_chart.create_histogram(name='macd_hist')

    macd_df = df[['date', 'macd', 'macd_signal', 'macd_hist']]

    macd_line.set(macd_df[['date', 'macd']])
    macd_signal.set(macd_df[['date', 'macd_signal']])
    macd_hist.set(macd_df[['date', 'macd_hist']])

    chart.show(block=True)