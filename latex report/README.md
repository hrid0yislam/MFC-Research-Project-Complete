# LaTeX Report - Experiment 1

## Overview
This folder contains the LaTeX source code for the comprehensive Experiment 1 report on "Electrochemical Performance Evaluation of Carbon Black-Modified Electrodes in Dairy Wastewater Microbial Fuel Cell Systems".

## Files
- `experiment_1_report.tex` - Main LaTeX document
- `README.md` - This file

## Compilation Instructions

### Method 1: Using pdflatex (Recommended)
```bash
cd "latex report"
pdflatex experiment_1_report.tex
pdflatex experiment_1_report.tex  # Run twice for proper cross-references
```

### Method 2: Using LaTeX online editors
1. **Overleaf**: Upload the .tex file to [Overleaf](https://www.overleaf.com)
2. **ShareLaTeX**: Upload to your preferred online LaTeX editor
3. **TeXstudio**: Open in TeXstudio and compile

### Method 3: Using LaTeX distribution
- **Windows**: Install MiKTeX or TeX Live
- **Mac**: Install MacTeX
- **Linux**: Install texlive-full package

## Required Packages
The document uses the following LaTeX packages (automatically installed with most distributions):
- `geometry` - Page layout
- `amsmath, amsfonts, amssymb` - Mathematical symbols
- `graphicx` - Graphics support
- `booktabs` - Professional tables
- `siunitx` - Scientific units
- `hyperref` - Hyperlinks and cross-references
- `fancyhdr` - Headers and footers
- `float` - Float positioning

## Document Features
- Professional title page with abstract
- Table of contents
- Properly formatted scientific tables
- Cross-referenced sections and tables
- Scientific units using siunitx package
- Professional bibliography
- Academic formatting throughout

## Output
Compilation will produce `experiment_1_report.pdf` containing the complete research report.

## Notes
- The document is set up for A4 paper size
- All tables and figures are properly numbered and referenced
- The bibliography includes example references (replace with actual citations)
- Headers include project and institution information

## Contact
For questions about the LaTeX formatting or compilation issues, contact the research team at UiT Narvik. 