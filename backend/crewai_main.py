"""
crewai_main.py
Sequential pipeline:
  1) Read profile.json + PDF -> consolidated paragraph + extracted fields
  2) Read jobs_informations.json (latest job) -> job analysis
  3) Build a tailored CV JSON + markdown, save outputs

Requirements:
  pip install google-genai python-dotenv pydantic PyPDF2
  Put GEMINI_API_KEY=... in .env
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime


from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from PyPDF2 import PdfReader

# Load environment
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("crewai")

# Ensure debug output folder
DEBUG_DIR = Path("debug_outputs")
DEBUG_DIR.mkdir(exist_ok=True)

def write_debug_file(name: str, content: str):
    try:
        (DEBUG_DIR / name).write_text(content or "", encoding="utf-8")
        logger.info(f"Wrote debug file: {DEBUG_DIR / name}")
    except Exception as e:
        logger.warning(f"Failed writing debug file {name}: {e}")

# Attempt to import genai
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False
    logger.error("google-genai not installed or import failed. Install with: pip install google-genai")

# ----------------------
# Pydantic models (v2)
# ----------------------
class ProfileResult(BaseModel):
    paragraph: str
    extracted: Dict[str, Any] = Field(default_factory=dict)
    pdf_preview: Optional[str] = None

class JobAnalysisResult(BaseModel):
    job_id: int
    summary: str
    purpose: str
    seniority: str
    required_skills: List[str]
    local_context: str
    hiring_advice: str

class CVModel(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: Optional[str] = None
    experience: List[Dict[str, Any]] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    other: Dict[str, Any] = Field(default_factory=dict)

# ----------------------
# Helpers
# ----------------------
def extract_text_from_pdf(path: str) -> str:
    if not os.path.exists(path):
        logger.warning(f"PDF not found: {path}")
        return ""
    try:
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages)
    except Exception as e:
        logger.error(f"Error reading PDF {path}: {e}")
        return ""

def safe_json_load_from_text(text: str) -> Optional[dict]:
    """Try to extract and parse a JSON object from text. Return dict or None."""
    if not text:
        return None
    text = text.strip()
    # Remove code fences if any
    for fence in ("```json", "```", "```text"):
        text = text.replace(fence, "")
    # Find first "{" and last "}" and try to parse
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        candidate = text[start:end]
        return json.loads(candidate)
    except Exception:
        # last resort: try parsing entire string (if it's just JSON)
        try:
            return json.loads(text)
        except Exception:
            return None

# ----------------------
# Gemini client wrapper
# ----------------------
class GeminiClient:
    def __init__(self, model: str = "gemini-2.0-flash"):
        if not GENAI_AVAILABLE:
            raise RuntimeError("google-genai package not available.")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment.")
        # Initialize client
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_output_tokens: int = 512) -> str:
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
        except Exception as e:
            logger.error(f"LLM generate_content call failed: {e}")
            raise
        # Prefer .text
        text = getattr(resp, "text", None)
        if text:
            return text
        # fallback nested
        try:
            output = getattr(resp, "output", None)
            if output and len(output) > 0:
                content = getattr(output[0], "content", None)
                if content and len(content) > 0:
                    return getattr(content[0], "text", str(resp))
        except Exception:
            pass
        return str(resp)

# ----------------------
# Agent with three tasks
# ----------------------
class Agent:
    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client

    def task_build_profile(self, profile_json_path: str, pdf_path: str) -> ProfileResult:
        # Read profile JSON
        if not os.path.exists(profile_json_path):
            raise FileNotFoundError(profile_json_path)
        with open(profile_json_path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        # Extract pdf text
        pdf_text = extract_text_from_pdf(pdf_path)
        write_debug_file("task1_profile.json", json.dumps(profile, ensure_ascii=False, indent=2))
        write_debug_file("task1_pdf_excerpt.txt", pdf_text[:6000])

        # Build prompt
        prompt = (
            "You are an assistant that must extract a concise candidate profile and provide structured fields.\n"
            "INPUT: a JSON profile and a PDF resume text.\n"
            "OUTPUT FORMAT: first print a short paragraph (2-4 sentences) describing the candidate, then on a new line print a JSON object with keys: "
            "\"name\", \"email\", \"phone\", \"top_skills\" (list), \"top_roles\" (list), \"education\" (list of strings), \"languages\" (list).\n\n"
            "PROFILE JSON:\n" + json.dumps(profile, ensure_ascii=False) + "\n\n"
            "PDF TEXT (excerpt):\n" + (pdf_text[:3000] if pdf_text else "") + "\n\n"
            "Return EXACTLY: paragraph then JSON (no extra commentary)."
        )

        generated = self.llm.generate(prompt) if self.llm else ""
        write_debug_file("task1_raw_llm.txt", generated or "")
        logger.info(f"Task1: LLM returned {len(generated or '')} chars")

        extracted_json = safe_json_load_from_text(generated or "")
        paragraph = ""
        if extracted_json is None:
            # if parsing failed, treat full response as paragraph only
            paragraph = (generated or "").strip()
            extracted_json = {}
            write_debug_file("task1_llm_unparsed.txt", generated or "")
            logger.warning("Task1: Could not parse JSON from LLM; saved raw output for inspection.")
        else:
            # attempt to isolate paragraph (text before JSON start)
            idx = (generated or "").rfind("{")
            paragraph = (generated or "")[:idx].strip() if idx != -1 else ""

        return ProfileResult(paragraph=paragraph, extracted=extracted_json, pdf_preview=(pdf_text[:1000] if pdf_text else None))

    def task_analyze_job(self, jobs_json_path: str) -> JobAnalysisResult:
        if not os.path.exists(jobs_json_path):
            raise FileNotFoundError(jobs_json_path)
        with open(jobs_json_path, "r", encoding="utf-8") as f:
            jobs = json.load(f)
        if not jobs:
            raise ValueError("jobs_informations.json is empty")

        # choose latest job using createdAt if possible
        def parse_dt(j):
            try:
                return datetime.datetime.fromisoformat(j.get("createdAt"))
            except Exception:
                return datetime.datetime.min

        latest = sorted(jobs, key=parse_dt)[-1]

        write_debug_file("task2_latest_job.json", json.dumps(latest, ensure_ascii=False, indent=2))

        prompt = (
            "You are a senior recruiter and analyst. Read the job posting below and return a JSON object with these keys:\n"
            "summary (1-2 sentences), purpose, seniority (Junior/Mid/Senior), required_skills (list), local_context, hiring_advice.\n"
            "Return only the JSON object.\n\n"
            "JOB POSTING JSON:\n" + json.dumps(latest, ensure_ascii=False) + "\n\n"
        )

        generated = self.llm.generate(prompt) if self.llm else ""
        write_debug_file("task2_raw_llm.txt", generated or "")
        logger.info(f"Task2: LLM returned {len(generated or '')} chars")

        parsed = safe_json_load_from_text(generated or "")
        if parsed is None:
            logger.warning("Task2: Could not parse JSON from LLM; saving raw.")
            write_debug_file("task2_llm_unparsed.txt", generated or "")
            # fallback minimal object
            parsed = {
                "summary": (generated or "")[:400],
                "purpose": "",
                "seniority": "Unknown",
                "required_skills": [],
                "local_context": "",
                "hiring_advice": ""
            }

        return JobAnalysisResult(
            job_id=latest.get("id", -1),
            summary=parsed.get("summary", ""),
            purpose=parsed.get("purpose", ""),
            seniority=parsed.get("seniority", "Unknown"),
            required_skills=parsed.get("required_skills", []),
            local_context=parsed.get("local_context", ""),
            hiring_advice=parsed.get("hiring_advice", "")
        )

    def task_build_cv(self, profile_res: ProfileResult, job_analysis: JobAnalysisResult) -> CVModel:
        # Build prompt for CV: request strict JSON then markdown delimited
        prompt = (
            "You are an expert CV writer. Using the PROFILE and JOB ANALYSIS below, produce:\n"
            "1) A strict JSON object with keys: name, email, phone, summary, experience (list of {title, company, start, end, description}), "
            "skills (list), education (list of {degree, institution, year}), languages (list), certifications (list), other (dict).\n"
            "2) On a new line, the marker ---MARKDOWN_CV--- and then a readable markdown CV.\n"
            "Return exactly the JSON, then the marker, then the markdown.\n\n"
            "PROFILE PARAGRAPH:\n" + profile_res.paragraph + "\n\n"
            "PROFILE EXTRACTED:\n" + json.dumps(profile_res.extracted, ensure_ascii=False) + "\n\n"
            "JOB ANALYSIS:\n" + json.dumps(job_analysis.model_dump(), ensure_ascii=False) + "\n\n"
            "If you must guess missing dates or fields, append ' (estimated)'."
        )

        generated = self.llm.generate(prompt) if self.llm else ""
        write_debug_file("task3_raw_llm.txt", generated or "")
        logger.info(f"Task3: LLM returned {len(generated or '')} chars")

        # split JSON and markdown
        json_part = None
        md_part = ""
        if generated:
            marker = "---MARKDOWN_CV---"
            if marker in generated:
                json_text, md_part = generated.split(marker, 1)
                json_text = json_text.strip()
            else:
                # try to extract JSON block
                parsed = safe_json_load_from_text(generated)
                if parsed is not None:
                    json_part = parsed
                    # markdown fallback: everything else
                    md_part = ""
                else:
                    # unable to find JSON
                    json_text = "{}"
            if json_part is None:
                # if we have json_text try to parse
                if 'json_text' in locals():
                    try:
                        json_part = json.loads(json_text)
                    except Exception:
                        json_part = safe_json_load_from_text(json_text) or {}
                        write_debug_file("task3_failed_json.txt", json_text)
        else:
            json_part = {}

        # Validate into CVModel
        try:
            cv = CVModel(**(json_part or {}))
        except ValidationError as e:
            logger.warning(f"Task3: Validation error building CVModel: {e}. Using tolerant fields.")
            cv = CVModel(
                name=(json_part or {}).get("name"),
                email=(json_part or {}).get("email"),
                phone=(json_part or {}).get("phone"),
                summary=(json_part or {}).get("summary") or profile_res.paragraph,
                experience=(json_part or {}).get("experience", []),
                skills=(json_part or {}).get("skills", []),
                education=(json_part or {}).get("education", []),
                languages=(json_part or {}).get("languages", []),
                certifications=(json_part or {}).get("certifications", []),
                other=(json_part or {}).get("other", {})
            )

        # Save outputs
        try:
            (Path("generated_cv.json")).write_text(cv.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Saved generated_cv.json")
        except Exception as e:
            logger.error(f"Failed saving generated_cv.json: {e}")

        # Save markdown if available, else create simple markdown
        try:
            md_path = Path("generated_cv.md")
            if md_part and md_part.strip():
                md_path.write_text(md_part.strip(), encoding="utf-8")
            else:
                # simple markdown fallback
                with md_path.open("w", encoding="utf-8") as f:
                    f.write(f"# {cv.name or 'Candidate'}\n\n")
                    if cv.email: f.write(f"- Email: {cv.email}\n")
                    if cv.phone: f.write(f"- Phone: {cv.phone}\n")
                    f.write("\n## Summary\n\n")
                    f.write((cv.summary or "") + "\n\n")
                    f.write("## Skills\n\n")
                    for s in cv.skills:
                        f.write(f"- {s}\n")
                    f.write("\n## Experience\n\n")
                    for e in cv.experience:
                        f.write(f"**{e.get('title','')}**, {e.get('company','')}\n\n{e.get('description','')}\n\n")
            logger.info("Saved generated_cv.md")
        except Exception as e:
            logger.error(f"Failed saving generated_cv.md: {e}")

        # If empty, write debug note
        if not cv.name and not cv.email and not cv.skills:
            write_debug_file("task3_cv_empty_notice.txt", "CV appears empty. Check task3_raw_llm.txt and task3_failed_json.txt for details.")
            logger.warning("Generated CV seems to be missing main fields. See debug_outputs/ for details.")

        return cv


import json
import datetime
import logging
from pathlib import Path

def save_final_results(cv_data: dict, output_dir: str = "."):
    """
    Save the final generated CV data into a JSON file with a timestamped filename.

    Args:
        cv_data (dict): The final CV dictionary (e.g., model_dump() result).
        output_dir (str): Directory where the JSON file should be saved. Default: current folder.

    Returns:
        str: The path to the saved JSON file.
    """

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"generated_cv_{timestamp}.json"

    # Save JSON in readable format (UTF-8, pretty printed)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cv_data, f, ensure_ascii=False, indent=2)

    # Log info for confirmation
    logging.info(f"✅ Final CV saved to: {output_path}")
    logging.info("📄 CV Content Preview:\n" + json.dumps(cv_data, ensure_ascii=False, indent=2)[:1500])

    return str(output_path)




# ----------------------
# Process runner (simple sequential)
# ----------------------
class Process:
    def __init__(self, name: str):
        self.name = name
        self.steps = []

    def add_step(self, name: str, func):
        self.steps.append((name, func))

    def run(self, initial_input: Dict[str, Any]):
        state = {"input": initial_input}
        logger.info(f"[Process:{self.name}] Starting with input: {initial_input}")
        for idx, (name, func) in enumerate(self.steps, start=1):
            logger.info(f"[Process:{self.name}] Running step {idx}: {name}")
            try:
                result = func(state)
            except Exception as e:
                logger.error(f"[Process:{self.name}] Step '{name}' failed: {e}", exc_info=True)
                raise
            state[name] = result
            logger.info(f"[Process:{self.name}] Step '{name}' done (type={type(result).__name__})")
        logger.info(f"[Process:{self.name}] Completed successfully.")
        return state

# ----------------------
# Main
# ----------------------
def run_job_analyzer():
    PROFILE_JSON = "backend/utils/profile.json"
    JOBS_JSON = "backend/utils/jobs_informations.json"
    PDF_PATH = "uploads/Motaouakel_ElMaimouni_CV.pdf"
    OUTPUT_JSON = "backend/utils/generated_cv.json"

    # Initialize Gemini client
    gemini = GeminiClient(model="gemini-2.0-flash")
    agent = Agent(llm_client=gemini)

    p = Process("profile_job_cv_pipeline")
    
    # Step 1 - build profile
    p.add_step("step1_build_profile", lambda state: agent.task_build_profile(PROFILE_JSON, PDF_PATH))
    # Step 2 - analyze job
    p.add_step("step2_analyze_job", lambda state: agent.task_analyze_job(JOBS_JSON))
    # Step 3 - build CV
    def step3(state):
        prof = state.get("step1_build_profile")
        job = state.get("step2_analyze_job")
        if prof is None or not isinstance(prof, ProfileResult):
            prof = agent.task_build_profile(PROFILE_JSON, PDF_PATH)
        if job is None or not isinstance(job, JobAnalysisResult):
            job = agent.task_analyze_job(JOBS_JSON)
        return agent.task_build_cv(prof, job)

    p.add_step("step3_build_cv", step3)

    # Run process
    final_state = p.run({"start": True})

    # Get final CV result
    cv = final_state.get("step3_build_cv")

    if isinstance(cv, CVModel):
        logger.info("✅ CV successfully generated")
        logger.info(cv.model_dump_json(indent=2, ensure_ascii=False))

        # ✅ Save clean JSON only
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(cv.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"💾 CV saved to {OUTPUT_JSON}")
    else:
        logger.warning("⚠️ Step 3 did not return a valid CVModel. Please check the model output.")

    logger.info("🎯 Process complete — only 'generated_cv.json' saved.")
    

if __name__ == "__main__":
    run_job_analyzer()


