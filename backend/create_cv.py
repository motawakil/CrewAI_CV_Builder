import json
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.colors import Color
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Frame, Spacer
from reportlab.lib.utils import ImageReader
from PIL import Image as PILImage, ImageDraw

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
# Create circular mask for image
# -------------------------------------------------------------------------
def create_circular_image(img_path, size):
    """Create a circular cropped image"""
    try:
        img = PILImage.open(img_path).convert('RGB')
        
        # Resize to square
        img = img.resize((size, size), PILImage.LANCZOS)
        
        # Create circular mask
        mask = PILImage.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        
        # Apply mask
        output = PILImage.new('RGB', (size, size), (255, 255, 255))
        output.paste(img, (0, 0))
        output.putalpha(mask)
        
        return output
    except Exception as e:
        print(f"Error creating circular image: {e}")
        return None


# -------------------------------------------------------------------------
# PDF Builder
# -------------------------------------------------------------------------
def create_dynamic_pdf(cv_data, job_info, filename="cv_dynamic.pdf"):
    PAGE_WIDTH, PAGE_HEIGHT = A4

    # --- Theme settings with elegant fonts ---
    primary = hex_to_color(job_info.get("primaryColor", "#1a1a2e"))
    secondary = hex_to_color(job_info.get("secondaryColor", "#16213e"))
    accent = hex_to_color("#0f3460")
    
    font_name = "Times-Roman"
    bold_font = "Times-Bold"
    italic_font = "Times-Italic"

    MARGIN_LEFT = 15 * mm
    MARGIN_RIGHT = 15 * mm
    MARGIN_TOP = 12 * mm
    MARGIN_BOTTOM = 12 * mm
    HEADER_HEIGHT = 52 * mm
    GAP = 8 * mm

    CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    COL_WIDTH = (CONTENT_WIDTH - GAP) * 0.60
    RIGHT_COL_WIDTH = (CONTENT_WIDTH - GAP) - COL_WIDTH

    # Image circle size
    IMAGE_CIRCLE_SIZE = 38 * mm

    # ---------------------------------------------------------------------
    # Enhanced style registration
    # ---------------------------------------------------------------------
    styles = getSampleStyleSheet()

    def set_style(name, **kwargs):
        if name in styles:
            s = styles[name]
            for k, v in kwargs.items():
                setattr(s, k, v)
        else:
            styles.add(ParagraphStyle(name=name, **kwargs))

    set_style('Name', fontName=bold_font, fontSize=26, textColor=primary, leading=30, spaceAfter=6)
    set_style('Section', fontName=bold_font, fontSize=12, textColor=secondary, spaceBefore=12, spaceAfter=8)
    set_style('Normal', fontName=font_name, fontSize=10, leading=14, spaceAfter=4)
    set_style('Bullet', fontName=font_name, fontSize=9.5, leading=14, leftIndent=12, spaceAfter=3)
    set_style('ContactSmall', fontName=font_name, fontSize=9.5, leading=13, textColor=colors.HexColor('#3a3a3a'))
    set_style('SubText', fontName=font_name, fontSize=9, leading=12, textColor=colors.HexColor('#5a5a5a'))

    # ---------------------------------------------------------------------
    # Begin PDF
    # ---------------------------------------------------------------------
    c = canvas.Canvas(filename, pagesize=A4)

    # ---------------------------------------------------------------------
    # Header (circular photo, name, contact line by line)
    # ---------------------------------------------------------------------
    header_top = PAGE_HEIGHT - MARGIN_TOP
    
    # Circular photo settings
    photo_radius = IMAGE_CIRCLE_SIZE / 2
    photo_x = MARGIN_LEFT
    photo_y = header_top - IMAGE_CIRCLE_SIZE - 3 * mm
    
    # Draw photo or placeholder
    has_photo = job_info.get("hasPhoto", False)
    photo_drawn = False
    
    if has_photo:
        photo_dir = Path("uploads/photos")
        if photo_dir.exists():
            photos = sorted(photo_dir.glob("*"), key=lambda x: x.stat().st_mtime)
            if photos:
                img_path = str(photos[-1])
                try:
                    # Create circular image
                    circular_img = create_circular_image(img_path, int(IMAGE_CIRCLE_SIZE * 2.83))  # Convert mm to pixels
                    
                    if circular_img:
                        # Save temp circular image
                        temp_path = Path("temp_circular.png")
                        circular_img.save(temp_path, "PNG")
                        
                        # Draw the circular image
                        c.drawImage(str(temp_path), photo_x, photo_y, 
                                  width=IMAGE_CIRCLE_SIZE, height=IMAGE_CIRCLE_SIZE, 
                                  mask='auto', preserveAspectRatio=True)
                        
                        # Draw border
                        c.setStrokeColor(primary)
                        c.setLineWidth(2)
                        c.circle(photo_x + photo_radius, photo_y + photo_radius, photo_radius, stroke=1, fill=0)
                        
                        photo_drawn = True
                        
                        # Clean up temp file
                        if temp_path.exists():
                            temp_path.unlink()
                except Exception as e:
                    print(f"Error drawing photo: {e}")
    
    # Draw placeholder circle if no photo
    if not photo_drawn:
        c.setStrokeColor(primary)
        c.setLineWidth(2)
        c.setFillColor(colors.white)
        c.circle(photo_x + photo_radius, photo_y + photo_radius, photo_radius, stroke=1, fill=1)

    # --- Name + Contact (line by line) ---
    name = cv_data.get("name", "John Doe")
    p_name = Paragraph(f"<b>{name}</b>", styles['Name'])

    # Contact fields
    email = cv_data.get("email")
    phone = cv_data.get("phone")
    other = cv_data.get("other", {}) or {}
    email2 = other.get("email2")
    portfolio = other.get("portfolio")
    linkedin = other.get("linkedin")
    redbubble = other.get("redbubble")
    dob = other.get("date_of_birth")
    address = other.get("address")

    # Build contact items (line by line)
    contact_flow = []
    if is_meaningful(email):
        contact_flow.append(Paragraph(f"Email: <a href='mailto:{email}' color='#0066cc'>{email}</a>", styles['ContactSmall']))
    if is_meaningful(email2) and email2 != email:
        contact_flow.append(Paragraph(f"Email: <a href='mailto:{email2}' color='#0066cc'>{email2}</a>", styles['ContactSmall']))
    if is_meaningful(phone):
        tel_link = phone.replace(" ", "")
        contact_flow.append(Paragraph(f"Phone: <a href='tel:{tel_link}' color='#0066cc'>{phone}</a>", styles['ContactSmall']))
    if is_meaningful(portfolio):
        display_url = portfolio.replace("https://", "").replace("http://", "")
        contact_flow.append(Paragraph(f"Portfolio: <a href='{portfolio}' color='#0066cc'>{display_url}</a>", styles['ContactSmall']))
    if is_meaningful(linkedin):
        display_url = linkedin.replace("https://", "").replace("http://", "")
        contact_flow.append(Paragraph(f"LinkedIn: <a href='{linkedin}' color='#0066cc'>{display_url}</a>", styles['ContactSmall']))
    if is_meaningful(redbubble):
        contact_flow.append(Paragraph(f"Redbubble: {redbubble}", styles['ContactSmall']))
    if is_meaningful(address):
        contact_flow.append(Paragraph(f"Address: {address}", styles['ContactSmall']))
    if is_meaningful(dob):
        contact_flow.append(Paragraph(f"Date of Birth: {dob}", styles['ContactSmall']))

    # Draw header frame to the right of photo
    header_frame_x = photo_x + IMAGE_CIRCLE_SIZE + 12 * mm
    header_frame_width = CONTENT_WIDTH - IMAGE_CIRCLE_SIZE - 12 * mm
    header_frame = Frame(header_frame_x, photo_y, header_frame_width, IMAGE_CIRCLE_SIZE + 3 * mm, showBoundary=0)
    
    header_content = [p_name, Spacer(1, 3)]
    header_content.extend(contact_flow)
    
    header_frame.addFromList(header_content, c)

    # Draw horizontal line under header
    line_y = photo_y - 8
    c.setStrokeColor(accent)
    c.setLineWidth(0.8)
    c.line(MARGIN_LEFT, line_y, PAGE_WIDTH - MARGIN_RIGHT, line_y)

    # ---------------------------------------------------------------------
    # Left Column (Profile, Education, Experience)
    # ---------------------------------------------------------------------
    def build_left():
        story = []

        # Profile/Summary
        summary = cv_data.get("summary", "")
        if is_meaningful(summary):
            story.append(Paragraph("<b>PROFILE</b>", styles["Section"]))
            story.append(Paragraph(summary, styles["Normal"]))
            story.append(Spacer(1, 8))

        # Education
        educations = cv_data.get("education", [])
        if educations:
            story.append(Paragraph("<b>EDUCATION</b>", styles["Section"]))
            for edu in educations:
                degree = edu.get('degree', '')
                institution = edu.get('institution', '')
                year = edu.get('year', '')
                
                if degree:
                    story.append(Paragraph(f"<b>{degree}</b>", styles["Normal"]))
                
                details = []
                if is_meaningful(institution):
                    details.append(institution)
                if is_meaningful(year):
                    details.append(str(year))
                
                if details:
                    story.append(Paragraph(" | ".join(details), styles["SubText"]))
                    story.append(Spacer(1, 2))
                
                story.append(Spacer(1, 6))
            story.append(Spacer(1, 4))

        # Experience
        experiences = cv_data.get("experience", [])
        if experiences:
            story.append(Paragraph("<b>PROFESSIONAL EXPERIENCE</b>", styles["Section"]))
            for exp in experiences:
                title = exp.get("title", "")
                company = exp.get("company", "")
                start = exp.get("start", "")
                end = exp.get("end", "")
                desc = exp.get("description", "")
                
                if title:
                    story.append(Paragraph(f"<b>{title}</b>", styles["Normal"]))
                
                meta_parts = []
                if is_meaningful(company):
                    meta_parts.append(company)
                if is_meaningful(start) or is_meaningful(end):
                    date_range = f"{start} – {end}".strip(" –")
                    meta_parts.append(date_range)
                
                if meta_parts:
                    story.append(Paragraph(" | ".join(meta_parts), styles["SubText"]))
                    story.append(Spacer(1, 2))
                
                if is_meaningful(desc):
                    story.append(Spacer(1, 2))
                    story.append(Paragraph(desc.replace("\n", "<br/>"), styles["Bullet"]))
                
                story.append(Spacer(1, 10))

        return story

    # ---------------------------------------------------------------------
    # Right Column (Skills, Languages, Certifications)
    # ---------------------------------------------------------------------
    def build_right():
        story = []
        
        # Skills
        story.append(Paragraph("<b>SKILLS</b>", styles["Section"]))
        if cv_data.get("skills"):
            for skill in cv_data["skills"]:
                story.append(Paragraph(f"• {skill}", styles["Normal"]))
        else:
            story.append(Paragraph("No skills listed", styles["Normal"]))
        story.append(Spacer(1, 8))

        # Languages
        if cv_data.get("languages"):
            story.append(Paragraph("<b>LANGUAGES</b>", styles["Section"]))
            for lang in cv_data["languages"]:
                story.append(Paragraph(f"• {lang}", styles["Normal"]))
            story.append(Spacer(1, 8))

        # Certifications
        if cv_data.get("certifications"):
            story.append(Paragraph("<b>CERTIFICATIONS</b>", styles["Section"]))
            for cert in cv_data["certifications"]:
                story.append(Paragraph(f"• {cert}", styles["Normal"]))
                story.append(Spacer(1, 2))
        
        return story

    # ---------------------------------------------------------------------
    # Layout columns
    # ---------------------------------------------------------------------
    content_top = line_y - 8
    content_height = content_top - MARGIN_BOTTOM
    
    left_frame = Frame(MARGIN_LEFT, MARGIN_BOTTOM, COL_WIDTH, content_height, showBoundary=0)
    right_frame = Frame(MARGIN_LEFT + COL_WIDTH + GAP, MARGIN_BOTTOM, RIGHT_COL_WIDTH, content_height, showBoundary=0)

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
