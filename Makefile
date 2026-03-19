.DEFAULT_GOAL := all

PYTHONPATH_PREFIX = PYTHONPATH=.
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
UV_RUN = UV_CACHE_DIR=$(UV_CACHE_DIR) MPLBACKEND=Agg $(PYTHONPATH_PREFIX) uv run
PIECE ?=
PLOT ?= 1
AT ?=
WINDOW ?= 8
START ?=
DUR ?=
TESTS ?= tests

ifeq ($(PLOT),1)
RENDER_PLOT_FLAG = --plot
else
RENDER_PLOT_FLAG =
endif

.PHONY: all
all: format-check lint compile typecheck test

.PHONY: check
check: all

.PHONY: list
list:
	$(UV_RUN) python main.py --list

.PHONY: lint
lint:
	$(UV_RUN) ruff check .

.PHONY: format-check
format-check:
	$(UV_RUN) ruff format --check .

.PHONY: format
format:
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .

.PHONY: compile
compile:
	$(UV_RUN) python -m compileall code_musics tests main.py

.PHONY: typecheck
typecheck:
	$(UV_RUN) basedpyright

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

.PHONY: inspect
inspect:
ifndef PIECE
	$(error PIECE is required, for example `make inspect PIECE=ji_chorale AT=2:10`)
endif
ifndef AT
	$(error AT is required, for example `make inspect PIECE=ji_chorale AT=2:10`)
endif
	$(UV_RUN) python main.py $(PIECE) --inspect-at "$(AT)" --inspect-window $(WINDOW)

.PHONY: snippet
snippet:
ifndef PIECE
	$(error PIECE is required, for example `make snippet PIECE=ji_chorale AT=2:10 WINDOW=12`)
endif
ifndef AT
	$(error AT is required, for example `make snippet PIECE=ji_chorale AT=2:10 WINDOW=12`)
endif
	$(UV_RUN) python main.py $(PIECE) --snippet-at "$(AT)" --snippet-window $(WINDOW) $(RENDER_PLOT_FLAG)

.PHONY: render-window
render-window:
ifndef PIECE
	$(error PIECE is required, for example `make render-window PIECE=ji_chorale START=130 DUR=12`)
endif
ifndef START
	$(error START is required, for example `make render-window PIECE=ji_chorale START=130 DUR=12`)
endif
ifndef DUR
	$(error DUR is required, for example `make render-window PIECE=ji_chorale START=130 DUR=12`)
endif
	$(UV_RUN) python main.py $(PIECE) --window-start "$(START)" --window-dur $(DUR) $(RENDER_PLOT_FLAG)

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
