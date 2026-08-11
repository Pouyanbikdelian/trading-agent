#!/usr/bin/env python3
"""Compile linkedin_drafts/*.md into a single, nicely typeset PDF archive.

Why a script and not a one-off: this runs weekly, so the archive needs to be
regenerable after every new draft lands. Requires xelatex (TeX Live) on PATH.

Usage:  python3 scripts/build_linkedin_pdf.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "linkedin_drafts"

SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def esc(text: str) -> str:
    """LaTeX-escape, preserving **bold** and *italic* markdown."""
    tokens: list[str] = []

    def stash(wrapper: str):
        def _sub(m: re.Match) -> str:
            tokens.append(wrapper % esc(m.group(1)))
            return f"\x01{len(tokens) - 1}\x02"

        return _sub

    # NB: re.sub with a function repl inserts the token verbatim, so these
    # wrappers must carry a single backslash. A doubled one becomes a LaTeX
    # linebreak and blows up whenever bold text starts a paragraph.
    text = re.sub(r"\*\*(.+?)\*\*", stash(r"\textbf{%s}"), text, flags=re.S)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", stash(r"\emph{%s}"), text)
    out = "".join(SPECIALS.get(c, c) for c in text)
    return re.sub(r"\x01(\d+)\x02", lambda m: tokens[int(m.group(1))], out)


def parse(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")

    def section(name: str) -> str:
        m = re.search(rf"^## {name}\s*\n(.*?)(?=^## |^---\s*$|\Z)", raw, flags=re.S | re.M)
        return m.group(1).strip() if m else ""

    def field(label: str) -> str:
        m = re.search(rf"\*\*{label}:?\*\*:?\s*(.+)", raw)
        return m.group(1).strip() if m else ""

    notes = re.search(r"^\*Continuity notes(.+?)\*\s*$", raw, flags=re.S | re.M)
    return {
        "date": path.stem,
        "topic": field(r"Topic \(rotation\)"),
        "post": section("POST"),
        "visual": section("Suggested visual"),
        "angle": section("Angle"),
        "notes": ("Continuity notes" + notes.group(1)).strip() if notes else "",
    }


PREAMBLE = r"""
\documentclass[11pt]{article}
\usepackage[a4paper,margin=22mm,top=20mm,bottom=20mm]{geometry}
\usepackage{fontspec}
\usepackage{xcolor}
\frenchspacing
\usepackage{fancyhdr}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\setmainfont{DejaVu Serif}[Scale=0.92]
\setsansfont{DejaVu Sans}[Scale=0.92]
\setmonofont{DejaVu Sans Mono}[Scale=0.85]
\definecolor{ink}{HTML}{1A1A1A}
\definecolor{accent}{HTML}{0A66C2}
\definecolor{muted}{HTML}{6B6B6B}
\definecolor{boxbg}{HTML}{F4F7FB}
\color{ink}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{\sffamily\footnotesize\color{muted}\thepage}
\newcommand{\postbox}[1]{%
  \begingroup\sffamily
  \setlength{\fboxsep}{10pt}%
  \noindent\colorbox{boxbg}{%
    \begin{minipage}{\dimexpr\linewidth-2\fboxsep\relax}
    \color{ink}\raggedright\frenchspacing #1
    \end{minipage}}%
  \endgroup}
\newcommand{\lbl}[1]{{\sffamily\bfseries\footnotesize\color{accent}\MakeUppercase{#1}}\par\vspace{-0.15em}}
"""


def build(outdir: Path) -> Path:
    posts = [parse(p) for p in sorted(DRAFTS.glob("20*.md"))]
    if not posts:
        sys.exit("no drafts found in linkedin_drafts/")

    body = [PREAMBLE, r"\begin{document}"]
    body.append(
        r"\begin{center}{\sffamily\bfseries\Large LinkedIn drafts — archive}\par"
        r"\vspace{2pt}{\sffamily\color{muted}\small Yan · automated trading system series"
        rf" · {len(posts)} posts · {posts[0]['date']} to {posts[-1]['date']}"
        r"\\ Drafts for approval. Nothing in here has been published.}\end{center}"
        r"\vspace{4pt}\hrule\vspace{6pt}"
    )

    for i, p in enumerate(posts):
        if i:
            body.append(r"\clearpage")
        body.append(
            rf"{{\sffamily\bfseries\large {esc(p['date'])}}}\hfill"
            rf"{{\sffamily\footnotesize\color{{muted}} {esc(p['topic'])}}}\par"
            r"\vspace{2pt}\hrule\vspace{10pt}"
        )
        body.append(r"\lbl{Post — ready to paste}")
        body.append(r"\postbox{" + esc(p["post"]).replace("\n\n", "\\par\\vspace{0.5em}\n") + "}")
        body.append(r"\vspace{6pt}")
        for label, key in (("Suggested visual", "visual"), ("Why it lands", "angle")):
            if p[key]:
                body.append(rf"\lbl{{{label}}}{esc(p[key])}\par\vspace{{4pt}}")
        if p["notes"]:
            body.append(
                r"\vspace{2pt}{\sffamily\footnotesize\color{muted}" + esc(p["notes"]) + r"}\par"
            )
    body.append(r"\end{document}")

    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / "linkedin_drafts_archive.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        tex = Path(tmp) / "archive.tex"
        tex.write_text("\n".join(body), encoding="utf-8")
        for _ in range(2):
            r = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
        if r.returncode != 0:
            sys.exit(r.stdout[-3000:])
        shutil.copy(Path(tmp) / "archive.pdf", target)
    return target


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(DRAFTS))
    print(build(Path(ap.parse_args().outdir)))
