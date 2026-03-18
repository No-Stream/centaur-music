PYTHONPATH_PREFIX = PYTHONPATH=.
UV_RUN = $(PYTHONPATH_PREFIX) uv run
PIECE ?=
PLOT ?= 1
TESTS ?= tests/test_score.py tests/test_tuning.py

ifeq ($(PLOT),1)
RENDER_PLOT_FLAG = --plot
else
RENDER_PLOT_FLAG =
endif

.PHONY: list
list:
	$(UV_RUN) python main.py --list

.PHONY: lint
lint:
	$(UV_RUN) ruff check .

.PHONY: format
format:
	$(UV_RUN) ruff format .

.PHONY: test
test:
	$(UV_RUN) pytest $(TESTS)

.PHONY: test-selected
test-selected:
	$(UV_RUN) pytest $(TESTS)

.PHONY: render
render:
ifndef PIECE
	$(error PIECE is required, for example `make render PIECE=harmonic_window`)
endif
	$(UV_RUN) python main.py $(PIECE) $(RENDER_PLOT_FLAG)

.PHONY: render-sketches
render-sketches:
	$(UV_RUN) python main.py sketch_passacaglia $(RENDER_PLOT_FLAG)
	$(UV_RUN) python main.py sketch_invention $(RENDER_PLOT_FLAG)
	$(UV_RUN) python main.py sketch_arpeggios $(RENDER_PLOT_FLAG)
	$(UV_RUN) python main.py sketch_variations $(RENDER_PLOT_FLAG)
	$(UV_RUN) python main.py sketch_spiral $(RENDER_PLOT_FLAG)
	$(UV_RUN) python main.py sketch_interference $(RENDER_PLOT_FLAG)
	$(UV_RUN) python main.py sketch_arpeggios_cross $(RENDER_PLOT_FLAG)
	$(UV_RUN) python main.py sketch_spiral_arch $(RENDER_PLOT_FLAG)
	$(UV_RUN) python main.py sketch_interference_v2 $(RENDER_PLOT_FLAG)

.PHONY: render-all
render-all:
	$(UV_RUN) python main.py all $(RENDER_PLOT_FLAG)
