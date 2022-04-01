src = $(wildcard src/*.cpp)
obj = $(src:.cpp=.o)

CC = mpic++

CXXFLAGS = -g -Wall -std=c++11 -fopenmp -DPRINT

PROPOSAL = proposal

main: $(obj)
	$(CC) -o $@ $^ $(CXXFLAGS)

.PHONY: clean

clean:
	rm -f $(obj) main
	rm -f *.aux *.lof *.log *.lot *.toc *.bbl *.blg *.pdf

run: $(PROPOSAL).pdf

$(PROPOSAL).pdf: $(PROPOSAL).bbl $(PROPOSAL).tex
	pdflatex $(PROPOSAL).tex -draftmode
	pdflatex $(PROPOSAL).tex

$(PROPOSAL).bbl: $(PROPOSAL).aux
	bibtex $(PROPOSAL).aux

$(PROPOSAL).aux: $(PROPOSAL).bib
	pdflatex $(PROPOSAL).tex -draftmode
	pdflatex $(PROPOSAL).tex -draftmode
