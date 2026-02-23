# When a MermaidJS source file (*.mmd) is updated, generate a corresponding PNG file.
%.png: %.mmd
	npx -p @mermaid-js/mermaid-cli mmdc --input $< --output $@ --theme default --backgroundColor transparent --scale 4

# A top level target to mark that all diagrams have a correpsonding png that should exist if it doesn't already.
diagrams: $(patsubst %.mmd,%.png,$(wildcard *.mmd))

all: diagrams

.PHONY: diagrams all

clean:
	rm -f *.png
