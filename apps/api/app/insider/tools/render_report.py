#!/usr/bin/env python3
"""Render an insider-sale research report JSON into a clean, readable PDF.

Usage:  python3 tools/render_report.py <report.json> <out.pdf>

The routine (Claude) does the research and writes <report.json>; this script owns
the presentation so every report looks consistent. Pure-Python (reportlab only) so
it runs in the cloud routine with just `pip install reportlab` — no system libs.

report.json shape (all fields optional except recommendation/one_liner/ticker):
{
  "ticker": "HAL", "company": "Halliburton Co", "generated_utc": "2026-07-09T21:00Z",
  "recommendation": "SELL",            # BUY | SELL | HOLD | NEUTRAL | AVOID
  "confidence": "Medium",              # Low | Medium | High
  "one_liner": "One-sentence bottom line the reader sees first.",
  "summary_bullets": ["...", "..."],   # 3-6 at-a-glance points
  "transaction": {"person":"", "title":"", "date":"", "code":"", "plan_10b5_1": false,
                  "shares":0, "price":0, "amount":0, "ownership":"", "stake_context":""},
  "sections": [{"heading":"Signal read", "body":"para\n\npara\n- bullet", "sources":["url"]}],
  "sources": ["url", ...],
  "caveats": "text"
}
"""
import json, sys, datetime, re
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                HRFlowable, ListFlowable, ListItem)

REC_COLORS = {  # recommendation -> (bg, text)
    "BUY":     (colors.HexColor("#166534"), colors.white),
    "SELL":    (colors.HexColor("#991b1b"), colors.white),
    "AVOID":   (colors.HexColor("#991b1b"), colors.white),
    "HOLD":    (colors.HexColor("#92400e"), colors.white),
    "NEUTRAL": (colors.HexColor("#92400e"), colors.white),
}
INK   = colors.HexColor("#111827")
MUTE  = colors.HexColor("#6b7280")
RULE  = colors.HexColor("#d1d5db")
PANEL = colors.HexColor("#f3f4f6")
ACCENT= colors.HexColor("#1f2937")


def styles():
    ss = getSampleStyleSheet()
    S = {}
    S["title"]   = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                                  fontSize=19, leading=23, textColor=INK, spaceAfter=2)
    S["sub"]     = ParagraphStyle("s", fontName="Helvetica", fontSize=9.5, leading=13,
                                  textColor=MUTE, spaceAfter=2)
    S["recbig"]  = ParagraphStyle("rb", fontName="Helvetica-Bold", fontSize=22, leading=24,
                                  textColor=colors.white)
    S["reconf"]  = ParagraphStyle("rc", fontName="Helvetica", fontSize=10, leading=13,
                                  textColor=colors.white)
    S["recline"] = ParagraphStyle("rl", fontName="Helvetica-Bold", fontSize=12, leading=16,
                                  textColor=colors.white)
    S["h2"]      = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13, leading=16,
                                  textColor=ACCENT, spaceBefore=14, spaceAfter=4)
    S["body"]    = ParagraphStyle("b", fontName="Helvetica", fontSize=10, leading=15,
                                  textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6)
    S["bullet"]  = ParagraphStyle("bu", fontName="Helvetica", fontSize=10, leading=14,
                                  textColor=INK)
    S["kv"]      = ParagraphStyle("kv", fontName="Helvetica", fontSize=9.5, leading=13, textColor=INK)
    S["kvb"]     = ParagraphStyle("kvb", fontName="Helvetica-Bold", fontSize=9.5, leading=13, textColor=INK)
    S["src"]     = ParagraphStyle("src", fontName="Helvetica", fontSize=8, leading=11, textColor=MUTE)
    S["caveat"]  = ParagraphStyle("cav", fontName="Helvetica-Oblique", fontSize=8.5, leading=12, textColor=MUTE)
    return S


def esc(t):
    """XML-escape arbitrary text for reportlab Paragraph (safe for &, <, >)."""
    return escape(str("" if t is None else t))


def inline(t):
    """Escape XML, then render simple markdown (**bold**, _italic_) as reportlab markup."""
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![A-Za-z0-9_])_(?!_)(.+?)(?<!_)_(?![A-Za-z0-9_])", r"<i>\1</i>", t)
    return t


def money(v):
    try:    return "${:,.0f}".format(float(v))
    except (TypeError, ValueError): return str(v or "—")


def num(v):
    try:    return "{:,.0f}".format(float(v))
    except (TypeError, ValueError): return str(v or "—")


def body_flowables(text, S):
    """Turn a mini-markdown body (blank-line paragraphs, '- ' bullets) into flowables."""
    out, bullets = [], []
    def flush_bullets():
        if bullets:
            out.append(ListFlowable([ListItem(Paragraph(b, S["bullet"]), leftIndent=10)
                                     for b in bullets], bulletType="bullet", start="•",
                                    leftIndent=12, spaceAfter=6))
            bullets.clear()
    for block in (text or "").split("\n"):
        line = block.strip()
        if not line:
            flush_bullets(); continue
        if line.startswith("- ") or line.startswith("• "):
            bullets.append(inline(line[2:].strip()))
        else:
            flush_bullets(); out.append(Paragraph(inline(line), S["body"]))
    flush_bullets()
    return out


def build(data, out_pdf):
    S = styles()
    story = []
    ticker = data.get("ticker", "?")
    company = data.get("company", "")
    gen = data.get("generated_utc") or datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")

    # ── Header ────────────────────────────────────────────────
    story.append(Paragraph("Insider Sale — Research Report", S["sub"]))
    story.append(Paragraph("{} ({})".format(company or ticker, ticker), S["title"]))
    story.append(Paragraph("Generated {} · Research synthesis, not financial advice".format(gen), S["sub"]))
    story.append(Spacer(1, 10))

    # ── Recommendation banner ─────────────────────────────────
    rec = str(data.get("recommendation", "NEUTRAL")).upper()
    bg, _ = REC_COLORS.get(rec, REC_COLORS["NEUTRAL"])
    conf = data.get("confidence", "—")
    left = Paragraph(rec, S["recbig"])
    right = Paragraph("Confidence: {}".format(conf), S["reconf"])
    line = Paragraph(inline(data.get("one_liner", "")), S["recline"])
    banner = Table([[left, right], [line, ""]], colWidths=[3.6*inch, 3.0*inch])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("SPAN", (0,1), (1,1)),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (1,0), (1,0), "RIGHT"),
        ("LEFTPADDING", (0,0), (-1,-1), 14), ("RIGHTPADDING", (0,0), (-1,-1), 14),
        ("TOPPADDING", (0,0), (-1,0), 12), ("BOTTOMPADDING", (0,1), (-1,1), 12),
        ("TOPPADDING", (0,1), (-1,1), 2),
    ]))
    story.append(banner)
    story.append(Spacer(1, 12))

    # ── At a glance ───────────────────────────────────────────
    bullets = data.get("summary_bullets") or []
    if bullets:
        story.append(Paragraph("At a glance", S["h2"]))
        story.append(ListFlowable([ListItem(Paragraph(inline(b), S["bullet"]), leftIndent=10) for b in bullets],
                                  bulletType="bullet", start="•", leftIndent=12))
        story.append(Spacer(1, 6))

    # ── Transaction facts ─────────────────────────────────────
    t = data.get("transaction") or {}
    if t:
        plan = "Yes — Rule 10b5-1 plan (pre-scheduled)" if t.get("plan_10b5_1") else "No — discretionary"
        rows = [
            ["Reporting person", t.get("person", "—"), "Title", t.get("title", "—")],
            ["Transaction date", t.get("date", "—"), "Code", t.get("code", "—")],
            ["Shares sold", num(t.get("shares")), "Price", money(t.get("price"))],
            ["Total value", money(t.get("amount")), "Ownership", t.get("ownership", "—")],
            ["10b5-1 plan?", plan, "Remaining stake", t.get("stake_context", "—")],
        ]
        data_rows = [[Paragraph(esc(r[0]), S["kvb"]), Paragraph(esc(r[1]), S["kv"]),
                      Paragraph(esc(r[2]), S["kvb"]), Paragraph(esc(r[3]), S["kv"])] for r in rows]
        tbl = Table(data_rows, colWidths=[1.25*inch, 2.1*inch, 1.15*inch, 2.0*inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), PANEL),
            ("BOX", (0,0), (-1,-1), 0.5, RULE), ("INNERGRID", (0,0), (-1,-1), 0.5, colors.white),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7),
            ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(Paragraph("The transaction", S["h2"]))
        story.append(tbl)

    # ── Sections ──────────────────────────────────────────────
    for sec in data.get("sections") or []:
        story.append(Paragraph(esc(sec.get("heading", "")), S["h2"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceAfter=6))
        story.extend(body_flowables(sec.get("body", ""), S))
        srcs = sec.get("sources") or []
        if srcs:
            story.append(Paragraph("Sources: " + "  ·  ".join(esc(u) for u in srcs), S["src"]))
            story.append(Spacer(1, 2))

    # ── Caveats + sources ─────────────────────────────────────
    if data.get("caveats"):
        story.append(Paragraph("Caveats", S["h2"]))
        story.extend(body_flowables(data["caveats"], S))
    allsrc = data.get("sources") or []
    if allsrc:
        story.append(Paragraph("All sources", S["h2"]))
        story.append(ListFlowable([ListItem(Paragraph(esc(u), S["src"]), leftIndent=8) for u in allsrc],
                                  bulletType="bullet", start="•", leftIndent=10))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5); canvas.setFillColor(MUTE)
        canvas.drawString(0.9*inch, 0.5*inch, "Insider-sale research · automated · not financial advice")
        canvas.drawRightString(LETTER[0]-0.9*inch, 0.5*inch, "Page %d" % doc.page)
        canvas.restoreState()

    doc = SimpleDocTemplate(out_pdf, pagesize=LETTER,
                            leftMargin=0.9*inch, rightMargin=0.9*inch,
                            topMargin=0.8*inch, bottomMargin=0.8*inch,
                            title="%s insider-sale report" % ticker)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: render_report.py <report.json> <out.pdf>", file=sys.stderr); sys.exit(2)
    with open(sys.argv[1]) as f:
        build(json.load(f), sys.argv[2])
    print("wrote", sys.argv[2])
