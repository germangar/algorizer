This is a framework for creating/backtesting and running trading algorythms. It is intended to work in a familiar way for those who have used Tradingview's Pinescript, from writing the scripts to visualizing them in lightweight-charts. It doesn't intend to mimic the language, but the structure should be familiar enough to you to feel comfortable.

**What does it do:**
It connects to crypto exchanges using the CCXT library and downloads the historical data[1]. It saves it in cache and keeps updating it every time it's launched. It runs a backtest of your script on it, and then (if ordered to) it continues running it in realtime, casting webhook alerts to a webhook service (Like [my own free webhook script](https://github.com/germangar/whook)).

[1] It does not require an API key. Note: Not all exchanges provide historical data. If it fails try another exchange.

Strategies can be multi-timeframe. Trades can be oneway or hedged. It takes into account the fees cost by fetching them from the exchange. 

Most typical 'indicators' are already built in, and creating custom ones is relatively easy. 

I do think it provides a very intuitive and simple way of creating, backtesting and running trading bots for those with knowledge of python and/or pinescript.

The project is still a work in progress so you'll most likely find a few nuissances. However, **fetching the historical candles, fetching the real time price updates, running the backtest and running your strategy realtime is reliable.** Backtest and realtime executing match properly. There are **no lookahead nor repainting issues.**

Plotting capabilities: As of today it's capable of **plots, histograms, lines** (point to point) and **markers** (labels), as it's capable of creating subpanels and targetting these to them. Horizontal lines, boxes and tables remain in the to do list, and will probably stay there for quite some time.

I'm slowly adding documentation in the ['DOC.md'](https://github.com/germangar/algorizer/blob/main/DOC.md) file. It's not complete yet, but it offers a starting point. I also extensively commented the file ['example_misc.py'](https://github.com/germangar/algorizer/blob/main/example_misc.py) from which you can obtain a good amount of information and get a sense of how to create a strategy.

Use template.py as base to write scripts.


<img width="2118" height="1267" alt="image" src="https://github.com/user-attachments/assets/b1f69204-3e29-4865-a7d0-d4d0b5a66b35" />




### Future plans (aka to do list) ###
- Add more indicators and drawing options.
- Add inputs with ranges and steps for future batch-backtesting
- Direct broker connection with the exchange for the strategy code to confirm operations
- Low priority: Make the chart load bars in modular blocks so it doesn't take so long on high bar count (pandas is slow af).


. I will not make much work on the chart window UI. I'll make it able to change timeframe if the lightweight-charts gods allow it and that's it. But I'll gladly accept contributions on it. The script and the chart are in a client/server configuration so even whole new chart replacements other than lightweight-charts could be added</br>
. I'll also be happy to accept contributions in making it work with stocks. Only needs a fetcher file and a way to get the realtime price updates, but I have no idea where one can obtain that information in the world of stocks.

--------------------

Install the required modules with the usual requirements.txt

pip install -r requirements.txt



