"""Render the metrics report Markdown using the Codex artifact runtime."""
from pathlib import Path
import html
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/Study_RAG_Metrics_Validation.md"
OUT = ROOT / "output/pdf/Study_RAG_Metrics_Validation.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
font_root = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("CalibriReport", str(font_root / "calibri.ttf")))
pdfmetrics.registerFont(TTFont("CalibriReportBold", str(font_root / "calibrib.ttf")))
pdfmetrics.registerFontFamily("CalibriReport", normal="CalibriReport", bold="CalibriReportBold")
styles = {
    "body": ParagraphStyle("body", fontName="CalibriReport", fontSize=10.3, leading=14, spaceAfter=7),
    "title": ParagraphStyle("title", fontName="CalibriReportBold", fontSize=24, leading=28, spaceAfter=12),
    "h1": ParagraphStyle("h1", fontName="CalibriReportBold", fontSize=14, leading=18, spaceBefore=13, spaceAfter=7, keepWithNext=True),
    "h2": ParagraphStyle("h2", fontName="CalibriReportBold", fontSize=11.8, leading=15, spaceBefore=10, spaceAfter=6, keepWithNext=True),
    "cell": ParagraphStyle("cell", fontName="CalibriReport", fontSize=9.2, leading=12, spaceAfter=0),
    "head": ParagraphStyle("head", fontName="CalibriReportBold", fontSize=9.2, leading=12, spaceAfter=0),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.4, leading=12, spaceAfter=4, wordWrap="CJK"),
    "url": ParagraphStyle("url", fontName="CalibriReport", fontSize=9, leading=12, spaceAfter=8, wordWrap="CJK"),
}


def para(text, style="body"):
    clean = text.replace("`", "").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    return Paragraph(html.escape(clean), styles[style])


story = []
lines = SOURCE.read_text(encoding="utf-8").splitlines()
i = 0
code = False
while i < len(lines):
    line = lines[i]
    if line.startswith("```"):
        code = not code
    elif line.startswith("| "):
        records = []
        while i < len(lines) and lines[i].startswith("| "):
            row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r"[-: ]+", c) for c in row):
                records.append(row)
            i += 1
        count = len(records[0])
        widths = [112, 175, A4[0] - 84 - 287] if count == 3 else [(A4[0] - 84) / count] * count
        table = LongTable([[para(c, "head" if j == 0 else "cell") for c in row] for j, row in enumerate(records)],
                          colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#D9D9D9")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7E7E7")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.extend([table, Spacer(1, 9)])
        continue
    elif code and line:
        story.append(para(line, "code"))
    elif line.startswith("# "):
        story.append(para(line[2:], "title"))
    elif line.startswith("## "):
        story.append(para(line[3:], "h1"))
    elif line.startswith("### "):
        story.append(para(line[4:], "h2"))
    elif line:
        story.append(para(line, "url" if "https://" in line else "body"))
    i += 1


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("CalibriReport", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(42, 25, "Study RAG Metrics and Validation")
    canvas.drawRightString(A4[0] - 42, 25, str(document.page))
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=42, leftMargin=42,
                  topMargin=40, bottomMargin=42, title="Study RAG Metrics and Validation",
                  author="Study RAG Project").build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
