#!/usr/bin/env python3
# create_cv.py - corrected version for ReportLab style collision

import json
from pathlib import Path
from datetime import datetime
import logging

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import Color
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Frame, Image, Spacer

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Load data (robust)
# ---------------------------
def load_latest_data():
    base_dir = Path(__file__).resolve().parent
    utils_dir = base_dir / "utils"
    cv_path = utils_dir / "generated_cv.json"
    job_info_path = utils_dir / "jobs_informations.json"

    logger.info(f"create_cv.py path: {Path(__file__).resolve()}")
    logger.info(f"Expected utils directory: {utils_dir}")
    logger.info(f"Looking for CV file at: {cv_path}")
    logger.info(f"Looking for jobs file at: {job_info_path}")

    if not utils_dir.exists():
        raise FileNotFoundError(f"'utils' directory not found at: {utils_dir}")

    logger.info("Contents of utils/:")
    for p in sorted(utils_dir.iterdir()):
        logger.info(f" - {p.name}")

    if not cv_path.exists():
        raise FileNotFoundError(f"generated_cv.json not found at: {cv_path}")
    if not job_info_path.exists():
        raise FileNotFoundError(f"jobs_informations.json not found at: {job_info_path}")

    cv_data = json.loads(cv_path.read_text(encoding="utf-8"))
    jobs = json.loads(job_info_path.read_text(encoding="utf-8"))
    job_info = jobs[-1] if isinstance(jobs, list) and jobs else (jobs if isinstance(jobs, dict) else {})

    logger.info("Loaded CV and job info successfully.")
    return cv_data, job_info

# ---------------------------
# Helpers
# ---------------------------
def hex_to_color(hex_str):
    hex_str = hex_str.lstrip("#")
    r, g, b = tuple(int(hex_str[i:i+2], 16) / 255 for i in (0, 2, 4))
    return Color(r, g, b)



# ---------------------------
# PDF builder (robust styles)
# ---------------------------
def create_dynamic_pdf(cv_data, job_info, filename="cv_dynamic.pdf"):
    PAGE_WIDTH, PAGE_HEIGHT = A4

    # Theme & layout
    primarycolor = hex_to_color(job_info.get("primaryColor", "#27374D"))
    secondarycolor = hex_to_color(job_info.get("secondaryColor", "#344966"))
    # With this robust fallback
    requested_font = job_info.get("font", "Helvetica").lower()
    if requested_font in ["helvetica", "times-roman", "courier"]:
        font_name = requested_font.capitalize()  # Helvetica, Times-Roman, Courier
    else:
        font_name = "Helvetica"  # fallback if unknown

    MARGIN_LEFT = 12 * mm
    MARGIN_RIGHT = 12 * mm
    MARGIN_TOP = 8 * mm
    MARGIN_BOTTOM = 8 * mm
    HEADER_HEIGHT = 42 * mm
    GAP = 6 * mm

    CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    COL_WIDTH = (CONTENT_WIDTH - GAP) * 0.58
    RIGHT_COL_WIDTH = (CONTENT_WIDTH - GAP) - COL_WIDTH

    # Load sample stylesheet and update/add styles safely
    styles = getSampleStyleSheet()

    # Helper: set or add style safely
    def set_style(name: str, **kwargs):
        if name in styles:
            s = styles[name]
            for k, v in kwargs.items():
                setattr(s, k, v)
        else:
            styles.add(ParagraphStyle(name=name, **kwargs))

    # Use Helvetica-Bold for bold text
    bold_font = "Helvetica-Bold"

    # Set styles
    set_style('Name', fontName=bold_font, fontSize=20, textColor=primarycolor, leading=22)
    set_style('Job', fontName=font_name, fontSize=10, textColor=colors.black)
    set_style('Section', fontName=bold_font, fontSize=11, textColor=secondarycolor, spaceBefore=6)
    set_style('Normal', fontName=font_name, fontSize=9, leading=11)
    set_style('Bullet', fontName=font_name, fontSize=9, leading=11, leftIndent=8)

    # Canvas
    c = canvas.Canvas(filename, pagesize=A4)

    # Header area
    header_top = PAGE_HEIGHT - MARGIN_TOP
    img_h = HEADER_HEIGHT * 0.85
    img_w = 0.18 * CONTENT_WIDTH
    img_x = MARGIN_LEFT
    img_y = header_top - img_h

    # Photo handling
    has_photo = job_info.get("hasPhoto", False)
    if has_photo:
        photo_dir = Path("uploads/photos")
        if photo_dir.exists():
            photos = sorted(photo_dir.glob("*"), key=lambda x: x.stat().st_mtime)
            if photos:
                img_path = photos[-1]
                try:
                    img = Image(str(img_path), width=img_w, height=img_h)
                    img.drawOn(c, img_x, img_y)
                except Exception as e:
                    logger.warning(f"Failed to draw photo {img_path}: {e}")
    else:
        # draw placeholder circle
        c.setStrokeColor(primarycolor)
        cx = img_x + img_w/2
        cy = img_y + img_h/2
        c.circle(cx, cy, img_w/2, stroke=1, fill=0)

    # Name & summary
    name = cv_data.get("name", "John Doe")
    summary = cv_data.get("summary", "")
    p_name = Paragraph(f"<b>{name}</b>", styles['Name'])
    p_summary = Paragraph(summary, styles['Job'])
    frame = Frame(img_x + img_w + 8 * mm, img_y, CONTENT_WIDTH - img_w - 20 * mm, img_h, showBoundary=0)
    frame.addFromList([p_name, p_summary], c)

    # Left column content
    def build_left():
        story = []
        story.append(Paragraph("Profile", styles["Section"]))
        story.append(Paragraph(summary or "No summary provided.", styles["Normal"]))
        story.append(Spacer(1, 6))

        # Education
        story.append(Paragraph("Education", styles["Section"]))
        for edu in cv_data.get("education", []):
            line = f"<b>{edu.get('degree', '')}</b> — {edu.get('institution', '')} ({edu.get('year', '')})"
            story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 6))

        # Experience
        story.append(Paragraph("Professional Experience", styles["Section"]))
        for exp in cv_data.get("experience", []):
            title = exp.get("title", "")
            company = exp.get("company", "")
            start = exp.get("start", "")
            end = exp.get("end", "")
            desc = exp.get("description", "")
            story.append(Paragraph(f"<b>{title}</b> — {company} ({start} – {end})", styles["Normal"]))
            story.append(Paragraph(desc or "", styles["Bullet"]))
            story.append(Spacer(1, 4))
        return story

    # Right column content
    def build_right():
        story = []
        story.append(Paragraph("Skills", styles["Section"]))
        if cv_data.get("skills"):
            story.append(Paragraph(", ".join(cv_data["skills"]), styles["Normal"]))
        else:
            story.append(Paragraph("No skills listed.", styles["Normal"]))
        story.append(Spacer(1, 6))

        if cv_data.get("languages"):
            story.append(Paragraph("Languages", styles["Section"]))
            story.append(Paragraph(", ".join(cv_data["languages"]), styles["Normal"]))
            story.append(Spacer(1, 6))

        if cv_data.get("certifications"):
            story.append(Paragraph("Certifications", styles["Section"]))
            for cert in cv_data["certifications"]:
                story.append(Paragraph(cert, styles["Normal"]))
        return story

    # Layout frames
    left_frame = Frame(MARGIN_LEFT, MARGIN_BOTTOM, COL_WIDTH, PAGE_HEIGHT - HEADER_HEIGHT - 20 * mm, showBoundary=0)
    right_frame = Frame(MARGIN_LEFT + COL_WIDTH + GAP, MARGIN_BOTTOM, RIGHT_COL_WIDTH, PAGE_HEIGHT - HEADER_HEIGHT - 20 * mm, showBoundary=0)

    left_frame.addFromList(build_left(), c)
    right_frame.addFromList(build_right(), c)

    c.save()
    logger.info(f"✅ CV saved to {filename}")




# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    cv_data, job_info = load_latest_data()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"cv_dynamic_{timestamp}.pdf"
    create_dynamic_pdf(cv_data, job_info, output_file)
