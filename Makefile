PY ?= python3

.PHONY: install test sanity quick train sac walkforward hpo frontier shadow offline clean

install:      ## dependencies
	$(PY) -m pip install -r requirements.txt
test:         ## unit tests: look-ahead, accounting, costs, rewards, overlay, DSR, splits, envs
	$(PY) -m pytest -q
sanity:       ## planted-signal, pure-noise, accounting, from-scratch agents
	$(PY) scripts/sanity.py
quick:        ## end-to-end smoke run (single split, 3 seeds, short training)
	$(PY) scripts/train.py --config configs/quick.yaml
train:        ## main experiment: RecurrentPPO, 5 seeds, walk-forward
	$(PY) scripts/train.py --config configs/default.yaml
sac:          ## challenger
	$(PY) scripts/train.py --config configs/sac.yaml
real:         ## 50-name multi-asset universe from yfinance
	$(PY) scripts/train.py --config configs/yfinance.yaml
hpo:          ## optuna on the validation block
	$(PY) scripts/hpo.py --config configs/quick.yaml --trials 20
frontier:     ## execution / market making / deep hedging vs classical baselines
	$(PY) scripts/frontier.py --task all
shadow:       ## daily loop in shadow mode (MODEL=runs/<run>/fold_0)
	$(PY) scripts/shadow_run.py --model $(MODEL) --broker shadow
offline:      ## offline RL from the decision log
	$(PY) scripts/offline_run.py --from-backtest
clean:
	rm -rf runs live_state data_cache .pytest_cache
