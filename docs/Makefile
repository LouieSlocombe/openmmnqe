# Minimal makefile for Sphinx documentation.
#
# Strict by default, so a local build fails on the same warnings that the CI
# docs job and Read the Docs fail on. Override while drafting with:
#
#     make -C docs html SPHINXOPTS=

SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
