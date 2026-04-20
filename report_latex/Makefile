# Simple Makefile wrapper around latexmk.
#   make        -> build PDF
#   make watch  -> continuous build on file changes
#   make clean  -> remove intermediates
#   make purge  -> remove intermediates and the final PDF

MAIN := main
TEX  := $(MAIN).tex
PDF  := $(MAIN).pdf

.PHONY: all watch clean purge

all: $(PDF)

$(PDF): $(TEX) preamble.tex references.bib $(wildcard sections/*.tex) $(wildcard figures/*)
	latexmk -pdf -bibtex $(MAIN)

watch:
	latexmk -pdf -bibtex -pvc $(MAIN)

clean:
	latexmk -c $(MAIN)
	rm -f *.bbl *.run.xml *.bcf

purge: clean
	rm -f $(PDF)
