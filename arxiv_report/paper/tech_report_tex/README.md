# LaTeX Technical Report Workspace

This folder is a self-contained LaTeX workspace for the arXiv-style technical report.

## Main Entry

- `main.tex`

## Layout

- `sections/`: main body converted from `scene_aware_workflow_report_en.md`.
- `appendix/`: reproducibility and implementation details.
- `figures/generated/`: copied PDF/PNG figures generated from evaluation artifacts.
- `tables/generated/`: copied booktabs table fragments.
- `data/audit_csv/`: copied CSVs used to audit plotted values.
- `bib/references.bib`: verified bibliography entries.
- `notes/`: source Markdown drafts, review notes, and reference verification notes.

## Build

Preferred local build:

```bash
latexmk -pdf main.tex
```

Windows PowerShell:

```powershell
.\scripts\build_pdf.ps1
```

Linux/macOS:

```bash
bash scripts/build_pdf.sh
```

The folder can also be uploaded to Overleaf with `main.tex` as the root file.
