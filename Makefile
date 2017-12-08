default: clean ## Build the c++ rlnc extension module
	python3 setup.py build_ext --inplace
clean: ## remove all build artifacts, including the c++ extension module
	python3 setup.py clean

help:
	@grep -P '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
