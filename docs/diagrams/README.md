# Project Diagrams

This directory contains Mermaid.JS source files and generated PNG diagrams.

## Generating Diagrams

Generate all diagrams:

```bash
make diagrams
```
Or when running from project root:

```bash
make -C docs/diagrams diagrams
# OR simply
make -C docs/diagrams # Defaults to 'all' target which depends on 'diagrams'
```

Generate a specific diagram:
```bash
make diagram-name.png
```

## Maintenance

When updating diagrams:
1. Edit the `.mmd` source file
2. Run `make diagrams` to regenerate PNGs
3. Commit both `.mmd` and `.png` files

The diagrams use Mermaid.JS flowchart syntax. See [Mermaid documentation](https://mermaid.js.org/intro/) for reference.
