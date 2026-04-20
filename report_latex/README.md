# Course Project Report (LaTeX source)

This directory contains the LaTeX source of the course project report
"A Three-Stage SFT+GRPO Pipeline for Mathematical Reasoning on
Qwen2.5-1.5B".

## Layout

```
report_latex/
├── main.tex             # entry point, \input's all sections
├── preamble.tex         # packages, page setup, custom commands
├── references.bib       # bibliography (BibLaTeX format)
├── latexmkrc            # latexmk configuration (pdflatex + biber)
├── Makefile             # convenience targets: all / watch / clean / purge
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_data.tex
│   ├── 03_method.tex
│   ├── 04_reward.tex
│   ├── 05_experiments.tex
│   ├── 06_results.tex
│   ├── 07_general_eval.tex
│   ├── 08_failure.tex
│   └── 09_conclusion.tex
└── figures/             # PNG figures referenced from the sections
    ├── eval_score_comparison.png
    ├── response_len.png
    └── reward.png
```

## Build options

### Local build (TeX Live recommended)

Required programs: `latexmk`, `pdflatex`, `biber`. On Debian/Ubuntu:

```
sudo apt-get install texlive-latex-recommended texlive-latex-extra \
                     texlive-bibtex-extra texlive-fonts-extra \
                     biber latexmk
```

Then run from this directory:

```
make            # build main.pdf
make watch      # continuous re-build on file changes
make clean      # remove .aux/.bbl/.log/...
make purge      # also remove main.pdf
```

### Overleaf

1. Compress this `report_latex/` directory into a zip archive.
2. On Overleaf: *New Project* -> *Upload Project* -> pick the zip.
3. Set the *Compiler* to **pdfLaTeX** in the project menu.
4. Click *Recompile*. Bibliography is handled automatically.

## Editing tips

* Each numbered file in `sections/` is one chapter. Edits stay local.
* New citations go into `references.bib`; cite with
  `\cite{key}` or `\parencite{key}`.
* New figures go into `figures/` and are inserted with
  `\includegraphics[width=...]{figures/<name>.png}`.
* The `\modelname{}` and `\code{}` macros (defined in `preamble.tex`) are
  used to keep visual style consistent for model names and code-like
  identifiers.

## Author / cover information

The title block in `main.tex` uses a placeholder author:

```
\author{Group X\thanks{Replace with author names, IDs, course code, and affiliation.}}
```

Replace this with the actual group members, student IDs, course code, and
affiliation before submission.
