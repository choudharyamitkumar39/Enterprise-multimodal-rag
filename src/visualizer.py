"""
PDF Visualizer & Bounding Box Highlighting Module.
Renders high-resolution page previews with translucent bounding box highlights
to provide verifiable, click-to-audit source citations in the Streamlit UI.
"""

import os
import io
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image


def highlight_page_passage(
    pdf_path: Path,
    page_number: int,
    search_snippet: Optional[str] = None,
    bbox: Optional[List[float]] = None,
    dpi: int = 150,
) -> Optional[bytes]:
    """
    Opens a PDF page, highlights the specified passage or bounding box
    with a translucent yellow highlighter rect, and returns the PNG image bytes.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return None

    try:
        doc = fitz.open(str(pdf_path))
        if page_number < 1 or page_number > len(doc):
            return None

        page = doc[page_number - 1]

        # 1. Highlight via direct bounding box if provided
        if bbox and len(bbox) == 4:
            rect = fitz.Rect(bbox)
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=(1.0, 0.85, 0.0))  # Amber/Yellow
            annot.update()

        # 2. Or search snippet text occurrences on the page
        elif search_snippet and len(search_snippet.strip()) > 10:
            # Clean snippet for fuzzy phrase search
            clean_phrase = search_snippet.strip().split("\n")[0][:60]
            text_instances = page.search_for(clean_phrase)
            for inst in text_instances[:4]:
                annot = page.add_highlight_annot(inst)
                annot.set_colors(stroke=(1.0, 0.85, 0.0))
                annot.update()

        # Render page to high-res pixmap
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes
    except Exception as e:
        return None
