SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = portfolio_management
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean html

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  html      to build the HTML documentation"
	@echo "  clean     to clean the built docs"

clean:
	rm -rf $(BUILDDIR)/*

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
