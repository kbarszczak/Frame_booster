code-check:
	uv run ruff check src/
	uv run pyright src/

code-clean:
	uv run ruff check --fix-only src/
	uv run ruff check --select I --fix src/
	uv run ruff format src/

train:
	uv run python src/pytorch/train.py \
		--version $(word 2,$(MAKECMDGOALS)) \
		--name fbnet_macos \
		--target data/tmp \
		--data data/vimeo90k_pytorch \
		--train train.txt \
		--test test.txt \
		--valid valid.txt \
		--visualization vis.txt \
		--device mps \
		--batch_size 2 \
		--epochs 10 \
		--width 256 \
		--height 144 \
		--log_freq 500 \
		--save_freq 1000

help:
	@echo "Available commands:"
	@echo "  code-check              - Run linting and type checking"
	@echo "  code-clean              - Clean code"
	@echo "  train <version>         - Train model (e.g., make train v7_1)"

.PHONY: code-check code-clean train help
