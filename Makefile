.PHONY: test-fourier
test-fourier:
	$(call run_in_venv,"cmd/test_fourier.py")


# 1 - script path
define run_in_venv
	@(export DATA_DIR="${PWD}/data" && export FT_MDL="${PWD}/data/ft_cc.en.300_freqprune_100K_20K_pq_100.bin" && . .venv/bin/activate && PYTHONPATH="${PYTHONPATH}:${PWD}" && python3 $(1))
endef
