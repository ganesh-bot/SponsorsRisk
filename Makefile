.PHONY: smoke train-gru compare-all plots clean

smoke:
	python -m scripts.smoke_train

train-gru:
	python run_train.py

compare-all:
	python compare_all.py --epochs 30 --batch 256

compare-all-fast:
	python compare_all.py --epochs 3 --batch 128 --no-calibration

plots:
	python scripts/plot_curves.py --split val

clean:
	@echo "Removing intermediate artifacts..."
	@rm -rf results/probs/*.npy || true
