# Mira Core — convenience targets. See README "Setup".
# `make install` runs the idempotent installer; the rest wrap the mira CLI.
.PHONY: install serve chat doctor

install:        ## Install deps + config (flags: ARGS="--with-ollama --with-launchagent")
	bash scripts/setup.sh $(ARGS)

serve:          ## Start the web server (port 8000)
	uv run python mira_cli.py serve

chat:           ## Start the interactive CLI
	uv run python mira_cli.py chat

doctor:         ## Health check the install
	uv run python mira_cli.py doctor
