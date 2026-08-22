IMAGE ?= mindpalace
PALACE ?= .venv/bin/python -m palace.cli.app
CONFIG ?= config.yaml
DATA_DIR ?= $(CURDIR)/data

DOCKER_RUN = docker run --rm --network=host \
	-v "$(DATA_DIR):/app/data" \
	$(IMAGE)

.PHONY: chat index index-force search doctor stats \
	docker-build docker-chat docker-index docker-search \
	check-query setup clean

setup: .venv/touchfile

.venv/touchfile: pyproject.toml
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	touch .venv/touchfile

clean:
	rm -rf .venv
	rm -rf *.egg-info

chat:
	$(PALACE) chat --config "$(CONFIG)"

index:
	$(PALACE) index --config "$(CONFIG)"

index-force:
	$(PALACE) index --reindex --config "$(CONFIG)"

search: check-query
	$(PALACE) search "$(QUERY)" --top-k 5 --config "$(CONFIG)"

doctor:
	$(PALACE) doctor --config "$(CONFIG)"

stats:
	$(PALACE) stats --config "$(CONFIG)"

docker-build:
	docker build --tag "$(IMAGE)" .

docker-chat:
	docker run --rm --interactive --tty --network=host \
		-v "$(DATA_DIR):/app/data" \
		$(IMAGE) chat --config config.yaml

docker-index:
	$(DOCKER_RUN) index --config config.yaml

docker-search: check-query
	$(DOCKER_RUN) search "$(QUERY)" --top-k 5 --config config.yaml

check-query:
	@test -n "$(strip $(QUERY))" || \
		{ echo 'QUERY is required (for example: make search QUERY="sleep patterns")' >&2; exit 2; }
