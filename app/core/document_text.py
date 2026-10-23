"""Full text readers for original Office and text documents; no active HTML."""
import posixpath
import re
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

LIMIT = 25_000_000


class PlainHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "head"}:
            self.hidden += 1
        if not self.hidden and tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "head"}:
            self.hidden = max(0, self.hidden - 1)
        if not self.hidden and tag in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def full_document_text(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".html", ".htm"}:
        if path.stat().st_size > LIMIT:
            raise ValueError("Source is too large for the full text view.")
        raw = path.read_bytes()
        text = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig", errors="replace")
        if suffix in {".html", ".htm"}:
            parser = PlainHTML()
            parser.feed(text)
            text = "".join(parser.parts)
        return [{"title": "Full document", "text": text.strip()}]
    if suffix not in {".docx", ".pptx"}:
        return []
    with ZipFile(path) as archive:
        consumed = 0

        def xml(name):
            nonlocal consumed
            entry = archive.getinfo(name)
            consumed += entry.file_size
            if consumed > LIMIT:
                raise ValueError("Source is too large for the full text view.")
            data = archive.read(entry)
            if b"<!DOCTYPE" in data.upper():
                raise ValueError("Document type declarations are not supported.")
            return ET.fromstring(data)

        if suffix == ".docx":
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            names = ["word/document.xml"] + sorted(
                n for n in archive.namelist() if re.fullmatch(r"word/(header|footer)\d+\.xml", n)
                or n in {"word/footnotes.xml", "word/endnotes.xml"})
            sections = []
            for name in names:
                title = "Document" if name == "word/document.xml" else Path(name).stem
                paragraphs = []
                for p in xml(name).findall(".//w:p", ns):
                    text = "".join(node.text or "" if node.tag == f"{{{ns['w']}}}t" else
                                   "\t" if node.tag == f"{{{ns['w']}}}tab" else
                                   "\n" if node.tag == f"{{{ns['w']}}}br" else "" for node in p.iter())
                    if not text.strip():
                        continue
                    style = p.find("w:pPr/w:pStyle", ns)
                    style_name = style.attrib.get(f"{{{ns['w']}}}val", "") if style is not None else ""
                    heading = style_name.lower().startswith(("heading", "title"))
                    if paragraphs and (heading or len(paragraphs) >= 12):
                        sections.append({"title": title, "text": "\n\n".join(paragraphs)})
                        paragraphs = []
                        title = text if heading else f"Section {len(sections) + 1}"
                    elif heading:
                        title = text
                    paragraphs.append(text)
                if paragraphs:
                    sections.append({"title": title, "text": "\n\n".join(paragraphs)})
            return sections
        ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
              "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
              "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
        relations = {r.attrib["Id"]: r.attrib.get("Target", "")
                     for r in xml("ppt/_rels/presentation.xml.rels")
                     if r.attrib.get("TargetMode") != "External"}
        sections = []
        for index, slide in enumerate(xml("ppt/presentation.xml").findall(".//p:sldId", ns), 1):
            target = relations[slide.attrib[f"{{{ns['r']}}}id"]]
            name = posixpath.normpath(posixpath.join("ppt", target)) if not target.startswith("/") else target.lstrip("/")
            if not name.startswith("ppt/slides/"):
                raise ValueError("Invalid slide relationship.")
            paragraphs = ["".join(t.text or "" for t in p.findall(".//a:t", ns))
                          for p in xml(name).findall(".//a:p", ns)]
            sections.append({"title": f"Slide {index}", "text": "\n\n".join(paragraphs)})
        return sections
