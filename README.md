
> Changelog for 0.1.1:</br>
> - Fix: Add missing backtrace error feedback from asynchronous tasks execution.</br>
> - Fix: broker_event info dictionary had the key "source " with a trailing space. Remove the space. Fortunately the template also has the space in the docstrings, so as long as the users notice them it will be working. But the space shouldn't be there and I'm removing it.</br>
> - Fix: ta.SHIFT would return Nan when running in real time.</br>
> - Add: TEMA, VWMA, ALMA and LSMA to 'ta'</br>

</br></br></br>


Algorizer is a framework for creating/backtesting and running trading algorythms. It is intended to work in a familiar way for those who have used Tradingview's Pinescript, from writing the scripts to visualizing them in lightweight-charts. It doesn't intend to mimic the language, but the structure should be familiar enough to you to feel comfortable.
</br></br>

**What does it do:**
It connects to crypto exchanges using the CCXT library and downloads the historical data[1]. It saves it in cache and keeps updating it every time it's launched. It runs a backtest of your script on it, and then (if ordered to) it continues running it in realtime, casting webhook alerts to a webhook service (Like [my own free webhook script](https://github.com/germangar/whook)).

> [1] It does not require an API key and it will take a while to download for the first time. Note: Not all exchanges provide historical data. If it fails try another exchange.

Strategies can be multi-timeframe (minimum timeframe is 1 minute). Trades can be oneway or hedged. It takes into account the fees cost by fetching them from the exchange. 

Most typical 'indicators' are already built in, and creating custom ones is relatively easy. By default all series are calculated using Numpy. Talib is used for some of them when it's present in the system. Talib provides a marginal speed increase..


## ** Quick Start **

- Install the algorizer module
```python -m pip install algorizer```

- Create an empty python script and copy/paste the contents of the [script template](https://github.com/germangar/algorizer/blob/main/template.py)

- Run it to verify everything is correct. Start writing your strategy.

</br>

> Warning: In the v0.1.0 release there's a bug in the name of a key in the info dictionary passed to ```broker_event```.
> The key is ```'source '``` (with a trailing space) instead of ```'source'```. The trailing space is removed in the repository and will be removed in the next release. You should add the trailing space to the template if you are working with v0.1.0. The bug is only triggered when a ```broker_event``` happens, so only when an order is issued in real time execution.

</br>
I'm slowly adding documentation in the DOC file: https://github.com/germangar/algorizer/blob/main/DOC.md </br>
I also extensively commented the file example_misc.py: https://github.com/germangar/algorizer/blob/main/example_misc.py

</br></br></br>

The project is still a work in progress so you'll most likely find a few nuissances. However, **fetching the historical candles, fetching the real time price updates, running the backtest and running your strategy realtime is reliable.** Backtest and realtime execution match properly. There are **no lookahead nor repainting issues.**

Plotting capabilities: As of today it's capable of **plots, histograms, lines** (point to point) and **markers** (labels), as it's capable of creating subpanels and targetting these to them. Horizontal lines, boxes and tables remain in the to do list, and will probably stay there for quite some time.

</br>
<img width="2118" height="1267" alt="image" src="https://github.com/user-attachments/assets/b1f69204-3e29-4865-a7d0-d4d0b5a66b35" />
</br></br>

> Note on dependencies: Pandas is only used to load the data into the chart. Lightweight-chart requires it. It isn't used anywhere else. However, even if algorizer doesn't directly use it, CCXT does.
</br>


### Future plans (aka to do list) ###
- Add more indicators and drawing options.
- Add inputs with ranges and steps for future batch-backtesting
- Direct broker connection with the exchange for the strategy code to confirm operations
- Low priority: Make the chart load bars in modular blocks so it doesn't take so long on high bar count (pandas is slow af).


. I will not make much work on the chart window UI. I'll make it able to change timeframe if the lightweight-charts gods allow it and that's it. But I'll gladly accept contributions on it. The script and the chart are in a client/server configuration so even whole new chart replacements other than lightweight-charts could be added</br>
. I'll also be happy to accept contributions in making it work with stocks. Only needs a fetcher file and a way to get the realtime price updates, but I have no idea where one can obtain that information in the world of stocks.


