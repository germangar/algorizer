This is a framework for creating/backtesting and running trading algorythms. It is intended to work in a familiar way for those who have used Tradingview's Pinescript, from writing the scripts to visualizing them in lightweight-charts. It doesn't pretend to mimic the language, although the structure should be familiar enough to you. It tries to avoid 'the pandas ways' and be sequential.

As of now it connects to crypto exchanges and fetches the candles required for backtesting, and it can also run in realtime and send alerts to the usual webhooks (Like [my own](https://github.com/germangar/whook)).

The project is in early stages so you'll most likely find nuissances, specially at writing your script since I didn't add much error handling yet, so you'll find yourself reliant on Python's backtraces.

I do think it provides a very intuitive and simple way of creating, backtesting and running trading bots for those with knowledge of python and or pinescript.

I will provide basic documentation later on, but as of now there is a ['example_misc.py'](https://github.com/germangar/algorizer/blob/main/example_misc.py) file which I commented extensively which should provide a starting point.

You can use templaty.py as base to write scripts.

(lines and tables are not yet implemented. They're coming soon. As of now only plots, histograms and markers are available).

![algorizer screenshot](https://github.com/user-attachments/assets/e8a99250-9224-44f2-8c70-b738de66cddd)


Install the required modules with the usual requirements.txt


pip install -r requirements.txt

- I would love it if the project could also be used for stocks, but I'm not familiar with that type of brokers. So if anyone is willing to collaborate in adding a fetcher and realtime updates of candles for stoks it will be very welcome.


