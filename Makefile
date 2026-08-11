.PHONY: changelog

changelog:
	git-cliff --config cliff.toml --output CHANGELOG.md
