@echo off
set coins=LDO

for %%c in (%coins%) do (
	start "%%c Instance" python strat.py %%c
    #start "%%c Instance" pypy strat.py %%c
)