"""Convert original Office layout to PDF locally, with a content-addressed cache.

The isolated worker uses LibreOffice when installed, or Windows Microsoft Office.
No document is uploaded to an external viewer service.
"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

_conversion_lock = threading.Lock()


class OfficePreviewError(RuntimeError):
    pass


def ensure_pdf(source: Path, cache: Path) -> Path:
    if source.suffix.lower() == ".pdf":
        return source
    if source.suffix.lower() not in {".docx", ".pptx"}:
        raise ValueError("This source does not have an Office page preview.")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{digest}.pdf"
    with _conversion_lock:
        if target.is_file() and target.stat().st_size > 0:
            return target
        with tempfile.TemporaryDirectory(prefix="office-", dir=cache) as working:
            output = Path(working) / "preview.pdf"
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            result = subprocess.run(
                [sys.executable, "-m", "app.core.office_preview", str(source.resolve()), str(output.resolve())],
                capture_output=True, timeout=90, **options,
            )
            if result.returncode or not output.is_file() or not output.read_bytes().startswith(b"%PDF"):
                raise OfficePreviewError("Original Office pages could not be rendered. Microsoft Office (Windows) or LibreOffice is required.")
            output.replace(target)
    return target


def convert(source: Path, output: Path) -> None:
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if executable:
        profile = (output.parent / "lo-profile").as_uri()
        result = subprocess.run([executable, f"-env:UserInstallation={profile}", "--headless",
                                 "--convert-to", "pdf", "--outdir", str(output.parent), str(source)],
                                capture_output=True, timeout=75)
        generated = output.parent / f"{source.stem}.pdf"
        if result.returncode or not generated.is_file():
            raise RuntimeError("LibreOffice conversion failed.")
        generated.replace(output)
        return
    if os.name != "nt":
        raise RuntimeError("Install LibreOffice to render Office documents.")
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    app = document = None
    previous_update_links = None
    owns_app = True
    try:
        if source.suffix.lower() == ".docx":
            app = win32com.client.DispatchEx("Word.Application")
            app.Visible = False
            app.DisplayAlerts = 0
            app.AutomationSecurity = 3
            previous_update_links = app.Options.UpdateLinksAtOpen
            app.Options.UpdateLinksAtOpen = False
            document = app.Documents.Open(str(source), ConfirmConversions=False, ReadOnly=True,
                                          AddToRecentFiles=False, PasswordDocument="preview-no-password")
            document.ExportAsFixedFormat(str(output), 17)
        else:
            try:
                win32com.client.GetActiveObject("PowerPoint.Application")
                owns_app = False
            except pythoncom.com_error:
                pass
            app = win32com.client.DispatchEx("PowerPoint.Application")
            previous_security = app.AutomationSecurity
            app.AutomationSecurity = 3
            try:
                document = app.Presentations.Open(str(source), ReadOnly=True, Untitled=False, WithWindow=False)
                document.SaveAs(str(output), 32)
            finally:
                app.AutomationSecurity = previous_security
    finally:
        if document is not None:
            document.Close(False) if source.suffix.lower() == ".docx" else document.Close()
        if app is not None and owns_app:
            if previous_update_links is not None:
                app.Options.UpdateLinksAtOpen = previous_update_links
            app.Quit()
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
