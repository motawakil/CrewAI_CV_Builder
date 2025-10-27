import json
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import Color
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Frame, Image, Spacer

# -------------------------------------------------------------------------
# Load CV Data
# -------------------------------------------------------------------------
def load_latest_data():
    cv_path = Path("generated_cv.json")
    job_info_path = Path("job_informations.json")

    if not cv_path.exists():
        raise FileNotFoundError("generated_cv.json not found.")
    if not job_info_path.exists():
        raise FileNotFoundError("job_informations.json not found.")

    cv_data = json.loads(cv_path.read_text(encoding="utf-8"))
    jobs = json.loads(job_info_path.read_text(encoding="utf-8"))
    job_info = jobs[-1] if isinstance(jobs, list) and jobs else jobs

    return cv_data, job_info


# -------------------------------------------------------------------------
# Color Helper
# -------------------------------------------------------------------------
def hex_to_color(hex_str):
    """Convert #RRGGBB to reportlab Color"""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        hex_str = "27374D"
    r, g, b = tuple(int(hex_str[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return Color(r, g, b)


# -------------------------------------------------------------------------
# Utility for checking meaningful value
# -------------------------------------------------------------------------
def is_meaningful(val):
    if not val:
        return False
    if not isinstance(val, str):
        return True
    v = val.strip()
    if not v:
        return False
    if v.lower() in ("unknown", "n/a", "na"):
        return False
    return True


# -------------------------------------------------------------------------
# PDF Builder
# -------------------------------------------------------------------------
def create_dynamic_pdf(cv_data, job_info, filename="cv_dynamic.pdf"):
    PAGE_WIDTH, PAGE_HEIGHT = A4

    # --- Theme settings ---
    primary = hex_to_color(job_info.get("primaryColor", "#27374D"))
    secondary = hex_to_color(job_info.get("secondaryColor", "#344966"))
    font_name = "Helvetica"
    bold_font = "Helvetica-Bold"

    MARGIN_LEFT = 12 * mm
    MARGIN_RIGHT = 12 * mm
    MARGIN_TOP = 8 * mm
    MARGIN_BOTTOM = 8 * mm
    HEADER_HEIGHT = 42 * mm
    GAP = 6 * mm

    CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    COL_WIDTH = (CONTENT_WIDTH - GAP) * 0.58
    RIGHT_COL_WIDTH = (CONTENT_WIDTH - GAP) - COL_WIDTH

    # ---------------------------------------------------------------------
    # Safe style registration helper
    # ---------------------------------------------------------------------
    styles = getSampleStyleSheet()

    def set_style(name, **kwargs):
        if name in styles:
            s = styles[name]
            for k, v in kwargs.items():
                setattr(s, k, v)
        else:
            styles.add(ParagraphStyle(name=name, **kwargs))

    set_style('Name', fontName=bold_font, fontSize=20, textColor=primary, leading=22)
    set_style('Job', fontName=font_name, fontSize=10, textColor=colors.black)
    set_style('Section', fontName=bold_font, fontSize=11, textColor=secondary, spaceBefore=6)
    set_style('Normal', fontName=font_name, fontSize=9, leading=11)
    set_style('Bullet', fontName=font_name, fontSize=9, leading=11, leftIndent=8)
    set_style('Contact', fontName=font_name, fontSize=9, leading=11, textColor=colors.black)

    # ---------------------------------------------------------------------
    # Begin PDF
    # ---------------------------------------------------------------------
    c = canvas.Canvas(filename, pagesize=A4)

    # ---------------------------------------------------------------------
    # Header (photo, name, summary, contact)
    # ---------------------------------------------------------------------
    header_top = PAGE_HEIGHT - MARGIN_TOP
    img_h = HEADER_HEIGHT * 0.85
    img_w = 0.18 * CONTENT_WIDTH
    img_x = MARGIN_LEFT
    img_y = header_top - img_h

    has_photo = job_info.get("hasPhoto", False)
    # If hasPhoto and there's a file, draw it; else draw circle placeholder
    if has_photo:
        photo_dir = Path("uploads/photos")
        photos = sorted(photo_dir.glob("*"), key=lambda x: x.stat().st_mtime) if photo_dir.exists() else []
        if photos:
            img_path = photos[-1]
            try:
                img = Image(str(img_path), width=img_w, height=img_h)
                img.drawOn(c, img_x, img_y)
            except Exception:
                c.setStrokeColor(primary)
                cx = img_x + img_w / 2
                cy = img_y + img_h / 2
                c.circle(cx, cy, img_w / 2, stroke=1, fill=0)
        else:
            c.setStrokeColor(primary)
            cx = img_x + img_w / 2
            cy = img_y + img_h / 2
            c.circle(cx, cy, img_w / 2, stroke=1, fill=0)
    else:
        c.setStrokeColor(primary)
        cx = img_x + img_w / 2
        cy = img_y + img_h / 2
        c.circle(cx, cy, img_w / 2, stroke=1, fill=0)

    # --- Name + Summary + Contact ---
    name = cv_data.get("name", "John Doe")
    summary = cv_data.get("summary", "")
    p_name = Paragraph(f"<b>{name}</b>", styles['Name'])
    p_summary = Paragraph(summary, styles['Job']) if is_meaningful(summary) else None

    # Contact fields: prefer top-level email/phone, fallback to cv_data['other']
    email = cv_data.get("email")
    phone = cv_data.get("phone")
    other = cv_data.get("other", {}) or {}
    # Accept secondary email field
    email2 = other.get("email2")
    portfolio = other.get("portfolio")
    linkedin = other.get("linkedin")
    redbubble = other.get("redbubble")
    dob = other.get("date_of_birth")
    address = other.get("address")

    # Build a list of contact fragments (only include meaningful ones)
    contact_items = []

    if is_meaningful(email):
        # clickable mailto link
        contact_items.append(f"<a href='mailto:{email}' color='blue'>✉ {email}</a>")
    if is_meaningful(email2) and email2 != email:
        contact_items.append(f"<a href='mailto:{email2}' color='blue'>✉ {email2}</a>")
    if is_meaningful(phone):
        # tel link
        tel_link = phone.replace(" ", "")
        contact_items.append(f"<a href='tel:{tel_link}' color='blue'>☎ {phone}</a>")
    if is_meaningful(portfolio):
        contact_items.append(f"<a href='{portfolio}' color='blue'>🔗 Portfolio</a>")
    if is_meaningful(linkedin):
        contact_items.append(f"<a href='{linkedin}' color='blue'>in: LinkedIn</a>")
    if is_meaningful(redbubble):
        # show only if meaningful and not 'Unknown'
        if is_meaningful(redbubble):
            contact_items.append(f"🎨 {redbubble}")
    if is_meaningful(address):
        contact_items.append(f"📍 {address}")
    if is_meaningful(dob):
        contact_items.append(f"🎂 {dob}")

    # Join contact items with separators
    contact_line = " — ".join(contact_items) if contact_items else ""

    # Draw the frame to the right of the photo for name/summary/contact
    frame = Frame(img_x + img_w + 8 * mm, img_y, CONTENT_WIDTH - img_w - 20 * mm, img_h, showBoundary=0)
    header_flow = [p_name]
    if p_summary:
        header_flow.append(p_summary)
    if contact_line:
        header_flow.append(Spacer(1, 4))
        header_flow.append(Paragraph(contact_line, styles['Contact']))
    frame.addFromList(header_flow, c)

    # ---------------------------------------------------------------------
    # Left Column (Profile, Contact block, Education, Experience)
    # ---------------------------------------------------------------------
    def build_left():
        story = []

        # Profile / Summary (on left as well if exists)
        if is_meaningful(summary):
            story.append(Paragraph("Profile", styles["Section"]))
            story.append(Paragraph(summary, styles["Normal"]))
            story.append(Spacer(1, 6))

        # Add a Contact section on the left for ATS / clarity
        if contact_line:
            story.append(Paragraph("Contact", styles["Section"]))
            # prefer showing each contact item on new line for readability
            for item in contact_items:
                story.append(Paragraph(item, styles["Normal"]))
            story.append(Spacer(1, 6))

        # Education
        educations = cv_data.get("education", [])
        if educations:
            story.append(Paragraph("Education", styles["Section"]))
            for edu in educations:
                line = f"<b>{edu.get('degree', '')}</b>"
                details = []
                if is_meaningful(edu.get('institution', '')):
                    details.append(edu.get('institution', ''))
                if is_meaningful(edu.get('year', '')):
                    details.append(str(edu.get('year', '')))
                if details:
                    line += " — " + " | ".join(details)
                story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 6))

        # Experience
        experiences = cv_data.get("experience", [])
        if experiences:
            story.append(Paragraph("Professional Experience", styles["Section"]))
            for exp in experiences:
                title = exp.get("title", "")
                company = exp.get("company", "")
                start = exp.get("start", "")
                end = exp.get("end", "")
                desc = exp.get("description", "")
                meta = " — ".join([p for p in (company, f"{start} – {end}" if (is_meaningful(start) or is_meaningful(end)) else None) if p])
                header = f"<b>{title}</b>" + (f" {meta}" if meta else "")
                story.append(Paragraph(header, styles["Normal"]))
                if is_meaningful(desc):
                    story.append(Paragraph(desc.replace("\n", "<br/>"), styles["Bullet"]))
                story.append(Spacer(1, 4))

        return story

    # ---------------------------------------------------------------------
    # Right Column (Skills, Languages, Certifications)
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # Layout columns
    # ---------------------------------------------------------------------
    left_frame = Frame(MARGIN_LEFT, MARGIN_BOTTOM, COL_WIDTH,
                       PAGE_HEIGHT - HEADER_HEIGHT - 20 * mm, showBoundary=0)
    right_frame = Frame(MARGIN_LEFT + COL_WIDTH + GAP, MARGIN_BOTTOM,
                        RIGHT_COL_WIDTH, PAGE_HEIGHT - HEADER_HEIGHT - 20 * mm, showBoundary=0)

    left_frame.addFromList(build_left(), c)
    right_frame.addFromList(build_right(), c)

    # ---------------------------------------------------------------------
    # Footer: timestamp
    # ---------------------------------------------------------------------
    footer_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    c.setFont(font_name, 7)
    c.setFillColor(colors.grey)
    c.drawRightString(PAGE_WIDTH - MARGIN_RIGHT, MARGIN_BOTTOM / 2, footer_text)

    c.save()
    print(f"✅ CV saved to {filename}")


# -------------------------------------------------------------------------
# MAIN (for standalone testing)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    cv_data, job_info = load_latest_data()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"cv_dynamic_{timestamp}.pdf"
    create_dynamic_pdf(cv_data, job_info, output_file)
    print("✅ PDF generation completed.")
