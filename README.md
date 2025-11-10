This is a framework for creating/backtesting and running trading algorythms. It is intended to work in a familiar way for those who have used Tradingview's Pinescript, from writing the scripts to visualizing them in lightweight-charts. It doesn't pretend to mimic the language, but the structure should be familiar enough to you to feel comfortable.

**What does it do:**
It connects to crypto exchanges using the CCXT library and downloads the required historical data. It saves it in cache and keeps updating it every time it's launched. It runs a backtest of your script on it, and then (if ordered to) it continues running it in realtime, casting webhook alerts to a webhook service (Like [my own free webhook script](https://github.com/germangar/whook)). It can run on multiple simultaneous timeframes.


The project is still a work in progress so you'll most likely find a few nuissances. Specially when writing your script since I didn't add much error handling yet, so you'll find yourself reliant on Python's backtraces. However, **fetching the backtesting candles, fetching the real time price updates, running the backtest and running your strategy realtime should be reliable at this point.**

Drawing capabilities are still a work in progress. As of today it's capable of plots, histograms, lines (point to point) and markers (labels), as it's capable of creating subpanels and targetting these to them. Horizontal lines, boxes and tables are still in the to do list.

I do think it provides a very intuitive and simple way of creating, backtesting and running trading bots for those with knowledge of python and/or pinescript.

I will provide basic documentation later on, but as of now there is a ['example_misc.py'](https://github.com/germangar/algorizer/blob/main/example_misc.py) file which I commented extensively which should provide a starting point.

You can use template.py as base to write scripts.

Changes since the original readme:
- The strategy code supports now both liquidation and take profit/stoploss orders.
- Old values in arrays can be accesed with relative indexing now, but only as long as the array is a generatedSeries_c (which will be the case unless you use a workaround and declare your own numpy array).
- You can retrieve data from a generatedSeries_c from a different timeframe with relative indexing too (timestamp is preferable, tho).
- When backtesting the fast timeframe the index of the slow timeframe will be adjustes to the fast timeframe timestamp (timeframes aren't calculated simultaneously in backtesting, but this allows to use some generatedSeries_c features like retrieving by relative indexing)
In general series calculations can be used 'naturally' now, but be always careful when combining timeframes. Remember they aren't calculated simoultaneously in backtests.

<img width="2416" height="1282" alt="image" src="https://github.com/user-attachments/assets/24b78bab-1899-4eaf-91c5-98d9c18e4505" />



### Future plans (aka to do list) ###
- Add more indicators and drawing options, and improve the code interoperability in general.
- Add inputs with ranges and steps for future batch-backtesting
- Direct broker connection with the exchange for the strategy code to confirm operations
- Make the chart load bars in modular blocks so it doesn't take so long on high bar count.


. I will not make much work on the chart window UI. I'll make it able to change timeframe if the lightweight-charts gods allow it and that's it. But I'll gladly accept contributions on it. The script and the chart are in a client/server configuration so even whole new chart replacements other than lightweight-charts could be added</br>
. I'll also be happy to accept contributions in making it work with stocks. Only needs a fetcher file and a way to get the realtime price updates, but I have no idea where one can obtain that information in the world of stocks.

--------------------

Install the required modules with the usual requirements.txt

pip install -r requirements.txt



