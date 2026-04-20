.DEFAULT_GOAL := all

PYTHONPATH_PREFIX = PYTHONPATH=.
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
BRICASTI_IR_DIR ?= $(CURDIR)/irs/bricasti
UV_RUN = UV_CACHE_DIR=$(UV_CACHE_DIR) BRICASTI_IR_DIR=$(BRICASTI_IR_DIR) MPLBACKEND=Agg $(PYTHONPATH_PREFIX) uv run
PIECE ?=
PLOT ?= 1
AT ?=
WINDOW ?= 8
START ?=
DUR ?=
MIDI_FORMATS ?=
BIT_DEPTH ?= 24
DRY ?= 0
NO_MIX ?= 0
ANALYSIS ?= 1
TESTS ?= tests
OBLIQUE ?= 0
MUSICAL ?= 0
VISUAL ?= 0
IMAGE ?= 0
CONSTRAINT ?= 0
N ?=
MODELS ?=

ifeq ($(PLOT),1)
RENDER_PLOT_FLAG = --plot
else
RENDER_PLOT_FLAG =
endif

ifeq ($(ANALYSIS),0)
RENDER_ANALYSIS_FLAG = --no-analysis
else
RENDER_ANALYSIS_FLAG =
endif

ifneq ($(strip $(MIDI_FORMATS)),)
MIDI_FORMATS_FLAG = --midi-formats $(MIDI_FORMATS)
else
MIDI_FORMATS_FLAG =
endif

ifeq ($(DRY),1)
STEMS_DRY_FLAG = --dry
else
STEMS_DRY_FLAG =
endif

ifeq ($(NO_MIX),1)
STEMS_NO_MIX_FLAG = --no-mix
else
STEMS_NO_MIX_FLAG =
endif

.PHONY: all
all: format-check lint compile typecheck test

.PHONY: check
check: all

.PHONY: list
list:
	$(UV_RUN) python main.py --list

.PHONY: lint
lint: lint-py lint-md

.PHONY: lint-py
lint-py:
	$(UV_RUN) ruff check .

.PHONY: lint-md
lint-md:
	@if command -v markdownlint-cli2 >/dev/null 2>&1; then \
		markdownlint-cli2 "docs/*.md" "*.md"; \
	else \
		echo "markdownlint-cli2 not installed — skipping markdown lint (install via: brew install markdownlint-cli2)"; \
	fi

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

# Run a read-only smoke-test script under scratch/ without a permission prompt.
#
# INTENDED USE ONLY:
#   - Read-only inspection (import a module, call a pure function, print the result)
#   - DSP/engine smoke tests (render a short buffer, assert finite/bounded)
#   - Character measurements (FFT a sine through a filter, print harmonic levels)
#
# NOT FOR:
#   - Writing files to the repo (including logs, audio, plots, caches)
#   - Running renders that touch `renders/`, `midi/`, `stems/`, etc.
#   - Any DAG/piece render, evaluation, MIDI or stem export — use `make render`,
#     `make midi`, `make stems`, `make evaluate` targets for those
#   - Network calls, subprocess spawning, env mutation
#   - Anything with side effects you wouldn't be happy running twice by accident
#
# The scratch/ directory is gitignored so these scripts never leave your tree.
# If a script needs to grow beyond a pure smoke test, promote it to a proper
# test under tests/ or a named make target instead.
#
# Usage:
#   make scratch SCRIPT=scratch/smoke_filters.py
.PHONY: scratch
scratch:
ifndef SCRIPT
	$(error SCRIPT is required, for example `make scratch SCRIPT=scratch/smoke.py`)
endif
	@case "$(SCRIPT)" in \
		scratch/*) ;; \
		*) echo "SCRIPT must be under scratch/ (got: $(SCRIPT))"; exit 1 ;; \
	esac
	$(UV_RUN) python $(SCRIPT)

.PHONY: render
render:
ifndef PIECE
	$(error PIECE is required, for example `make render PIECE=harmonic_window`)
endif
	$(UV_RUN) python main.py $(PIECE) $(RENDER_PLOT_FLAG) $(RENDER_ANALYSIS_FLAG)

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

.PHONY: midi
midi:
ifndef PIECE
	$(error PIECE is required, for example `make midi PIECE=ji_chorale`)
endif
	$(UV_RUN) python main.py $(PIECE) --export-midi $(MIDI_FORMATS_FLAG)

.PHONY: midi-snippet
midi-snippet:
ifndef PIECE
	$(error PIECE is required, for example `make midi-snippet PIECE=ji_chorale AT=2:10 WINDOW=12`)
endif
ifndef AT
	$(error AT is required, for example `make midi-snippet PIECE=ji_chorale AT=2:10 WINDOW=12`)
endif
	$(UV_RUN) python main.py $(PIECE) --export-midi $(MIDI_FORMATS_FLAG) --snippet-at "$(AT)" --snippet-window $(WINDOW)

.PHONY: midi-window
midi-window:
ifndef PIECE
	$(error PIECE is required, for example `make midi-window PIECE=ji_chorale START=130 DUR=12`)
endif
ifndef START
	$(error START is required, for example `make midi-window PIECE=ji_chorale START=130 DUR=12`)
endif
ifndef DUR
	$(error DUR is required, for example `make midi-window PIECE=ji_chorale START=130 DUR=12`)
endif
	$(UV_RUN) python main.py $(PIECE) --export-midi $(MIDI_FORMATS_FLAG) --window-start "$(START)" --window-dur $(DUR)

.PHONY: stems
stems:
ifndef PIECE
	$(error PIECE is required, for example `make stems PIECE=ji_chorale`)
endif
	$(UV_RUN) python main.py $(PIECE) --export-stems --stem-bit-depth $(BIT_DEPTH) $(STEMS_DRY_FLAG) $(STEMS_NO_MIX_FLAG)

.PHONY: stems-snippet
stems-snippet:
ifndef PIECE
	$(error PIECE is required, for example `make stems-snippet PIECE=ji_chorale AT=2:10 WINDOW=12`)
endif
ifndef AT
	$(error AT is required, for example `make stems-snippet PIECE=ji_chorale AT=2:10 WINDOW=12`)
endif
	$(UV_RUN) python main.py $(PIECE) --export-stems --stem-bit-depth $(BIT_DEPTH) $(STEMS_DRY_FLAG) $(STEMS_NO_MIX_FLAG) --snippet-at "$(AT)" --snippet-window $(WINDOW)

.PHONY: stems-window
stems-window:
ifndef PIECE
	$(error PIECE is required, for example `make stems-window PIECE=ji_chorale START=130 DUR=12`)
endif
ifndef START
	$(error START is required, for example `make stems-window PIECE=ji_chorale START=130 DUR=12`)
endif
ifndef DUR
	$(error DUR is required, for example `make stems-window PIECE=ji_chorale START=130 DUR=12`)
endif
	$(UV_RUN) python main.py $(PIECE) --export-stems --stem-bit-depth $(BIT_DEPTH) $(STEMS_DRY_FLAG) $(STEMS_NO_MIX_FLAG) --window-start "$(START)" --window-dur $(DUR)

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

.PHONY: evaluate
evaluate:
ifndef PIECE
	$(error PIECE is required, for example `make evaluate PIECE=slow_glass`)
endif
	$(UV_RUN) python -m code_musics.evaluate $(PIECE) \
		$(if $(MODELS),--models $(MODELS),)

.PHONY: evaluate-all
evaluate-all:
	$(UV_RUN) python -m code_musics.evaluate all \
		$(if $(MODELS),--models $(MODELS),)

.PHONY: inspire
inspire:
	$(UV_RUN) python -m code_musics.inspire \
		$(if $(N),--n=$(N),) \
		$(if $(filter-out 0,$(OBLIQUE)),--oblique=$(OBLIQUE),) \
		$(if $(filter-out 0,$(MUSICAL)),--musical=$(MUSICAL),) \
		$(if $(filter-out 0,$(VISUAL)),--visual=$(VISUAL),) \
		$(if $(filter-out 0,$(IMAGE)),--image=$(IMAGE),) \
		$(if $(filter-out 0,$(CONSTRAINT)),--constraint=$(CONSTRAINT),)
