# linkedin_drafts/ — conventions

Read this before writing a new weekly draft.

## Deliverables per week

1. `YYYY-MM-DD.md` — the draft (post text, suggested visual, angle, continuity notes).
2. `visuals/YYYY-MM-DD_<slug>.png` — **the visual as a standalone image.**
   Formulas and diagrams must never live in the post body: LinkedIn strips
   formatting, and Yan will not paste math into post text. The post text stays
   plain prose; the concept ships as an attached image.
3. `linkedin_drafts_archive.pdf` — regenerated archive of every draft to date.

## Regenerating

```bash
python3 scripts/build_linkedin_visuals.py   # renders every visual to visuals/
python3 scripts/build_linkedin_pdf.py       # rebuilds the PDF archive
```

Add a new render function to `build_linkedin_visuals.py` for each week's visual,
then call it from `__main__`. The `canvas()` / `box()` / `arrow()` helpers keep
the house style consistent: DejaVu Sans, warm off-white ground, blue/green
accents, square 1:1 (LinkedIn's best in-feed crop).

## Guardrails on visuals

Same rules as the post text. Illustrative shapes only — no real tickers,
weights, dollar amounts, correlations, thresholds, live P&L, or infra detail.
Label any chart that isn't a real backtest as illustrative on the image itself.

## Status

Everything here is a draft for Yan's approval. Nothing is posted automatically.
