@echo off
set coins=LDO CFX ETH XAUT BTC

for %%c in (%coins%) do (
    start "%%c Instance" python strat.py %%c
)