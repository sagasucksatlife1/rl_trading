PY ?= python3
RL = research/rl_allocator

.PHONY: install test snapshot backtest fit shadow paper research-sanity research-ab clean

# ---- shipped strategy: GBM score -> linear weights, no-trade band, risk overlay ----
install:     ## dependencies
	$(PY) -m pip install -r requirements.txt
test:        ## all tests (strategy + archived research)
	$(PY) -m pytest -q
snapshot:    ## hashed point-in-time data snapshot (SOURCE=sharadar DIR=vendor/sharadar)
	$(PY) scripts/snapshot.py build --source $(or $(SOURCE),synthetic_pit) $(if $(DIR),--dir $(DIR))
backtest:    ## walk-forward OOS backtest of the shipped rule vs alternatives, 1x/2x/3x costs
	$(PY) scripts/backtest_strategy.py --config configs/strategy.yaml
fit:         ## fit the deployable state (final GBM, HMM, normaliser) -> deploy/current
	$(PY) scripts/fit_strategy.py --config configs/strategy.yaml --out deploy/current
shadow:      ## daily loop, shadow mode (sends no orders)
	$(PY) scripts/shadow_run.py --deploy deploy/current --broker shadow
paper:       ## daily loop against the local paper broker
	$(PY) scripts/shadow_run.py --deploy deploy/current --broker paper

# ---- archived research: RL allocator (did not beat the shipped rule) ----
research-sanity:  ## planted/noise/accounting sanity tests for the RL allocator
	$(PY) $(RL)/scripts/sanity.py
research-ab:      ## Ticket 3 turnover A/B
	$(PY) $(RL)/scripts/ab_turnover.py --workers 4

clean:
	rm -rf runs live_state deploy data_cache .pytest_cache
