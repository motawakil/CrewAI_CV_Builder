
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
    required_skills: List[str] = Field(default_factory=list)
    local_context: str
    hiring_advice: str

class CVModel(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    headline: Optional[str] = None           # NEW: short job-adapted headline
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

cv_languages = ["fr", "en"]

class Agent:
    def __init__(self, llm_client: Optional[Any] = None):
        """
        llm_client is expected to have a .generate(prompt, **kwargs) -> str method.
        """
        self.llm = llm_client

    # -----------------------
    # Utility helpers
    # -----------------------
    def _sanitize_forbidden(self, text: str) -> str:
        if not text:
            return text or ""
        forbidden = ["estimated", "estimate", "approx", "approximate", "approximately"]
        out = text
        for w in forbidden:
            # remove whole-word occurrences case-insensitively
            out = out.replace(w, "")
            out = out.replace(w.capitalize(), "")
            out = out.replace(w.upper(), "")
        return out

    def _choose_most_relevant_skills(self, candidate_skills: List[str], job_skills: List[str], max_n: int = 8) -> List[str]:
        if not candidate_skills:
            return []
        cand = [str(s).strip() for s in candidate_skills if s and str(s).strip()]
        job = [str(s).strip() for s in (job_skills or []) if s and str(s).strip()]
        selected = []

        # 1) pick skills that match job skills (job skill appears in candidate skill or vice-versa)
        for j in job:
            if len(selected) >= max_n:
                break
            for c in cand:
                if c in selected:
                    continue
                # simple containment match, case-insensitive
                if j.lower() in c.lower() or c.lower() in j.lower():
                    selected.append(c)
                    break

        # 2) fallback: best remaining candidate skills (preserve original order)
        for c in cand:
            if len(selected) >= max_n:
                break
            if c not in selected:
                selected.append(c)

        return selected[:max_n]

    # -----------------------
    # Task 1: Build Profile
    # -----------------------
    def task_build_profile(self, profile_json_path: str, pdf_path: str) -> ProfileResult:
        # load profile json
        if not os.path.exists(profile_json_path):
            raise FileNotFoundError(profile_json_path)
        with open(profile_json_path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        # extract pdf text (helper provided by your codebase)
        pdf_text = extract_text_from_pdf(pdf_path)
        logger.info(f"task_build_profile: PDF excerpt (first 1000 chars):\n{(pdf_text or '')[:1000]}")

        write_debug_file("task1_profile.json", json.dumps(profile, ensure_ascii=False, indent=2))
        write_debug_file("task1_pdf_excerpt.txt", (pdf_text or "")[:6000])

        # Prompt: strict instructions, forbid uncertain words
        prompt = (
            "You are an expert HR extractor. Extract all candidate details from the provided JSON and PDF text.\n\n"
            "RULES:\n"
            "- Return EXACTLY: first a concise professional summary (3-5 sentences) tailored to a hiring manager, then a JSON object on the next line.\n"
            "- DO NOT include the words 'estimated', 'approx', 'approximate', 'approximately' or synonyms in ANY output.\n"
            "- If a field is missing, set it to an empty value or 'Unknown'.\n"
            "- top_skills should contain ONLY the most relevant skills (not dozens); limit to a reasonable number.\n\n"
            "EXPECTED JSON KEYS:\n"
            "name, email, phone, top_skills (list), top_roles (list), education (list of strings), languages (list), "
            "studies (list), interests (list), certificates (list), projects (list), other (dict)\n\n"
            "SOURCE PROFILE JSON:\n" + json.dumps(profile, ensure_ascii=False) + "\n\n"
            "PDF TEXT (excerpt):\n" + (pdf_text[:5000] if pdf_text else "") + "\n\n"
            "Return the summary paragraph, then the JSON. Nothing else."
        )

        # Call LLM
        if self.llm:
            # if your client accepts generation params, pass them (ex: temperature=0)
            # generated = self.llm.generate(prompt, temperature=0)
            generated = self.llm.generate(prompt)
        else:
            generated = ""

        write_debug_file("task1_raw_llm.txt", generated or "")
        logger.info(f"task_build_profile: LLM returned {len(generated or '')} chars")

        # sanitize and parse
        sanitized = self._sanitize_forbidden(generated or "")
        extracted_json = safe_json_load_from_text(sanitized)
        paragraph = ""

        if extracted_json is None:
            # fallback: attempt to split by first JSON brace
            idx = (sanitized).rfind("{")
            if idx != -1:
                paragraph = sanitized[:idx].strip()
                try:
                    extracted_json = json.loads(sanitized[idx:])
                except Exception:
                    extracted_json = {}
            else:
                paragraph = sanitized.strip()
                extracted_json = {}
            write_debug_file("task1_llm_unparsed.txt", generated or "")
            logger.warning("task_build_profile: Could not parse JSON from LLM cleanly; saved raw output.")
        else:
            # isolate paragraph (text before first json object)
            idx = (sanitized).rfind("{")
            paragraph = sanitized[:idx].strip() if idx != -1 else ""

        # normalize and trim top_skills
        if isinstance(extracted_json.get("top_skills"), list):
            extracted_json["top_skills"] = [str(x).strip() for x in extracted_json["top_skills"] if x][:8]

        return ProfileResult(
            paragraph=paragraph,
            extracted=extracted_json or {},
            pdf_preview=(pdf_text[:1000] if pdf_text else None)
        )

    # -----------------------
    # Task 2: Analyze Job
    # -----------------------
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
            "You are a senior recruiter and analyst. Read the job posting and return ONLY a JSON object with keys:\n"
            "summary (1-2 sentences), purpose, seniority (Junior/Mid/Senior), required_skills (list - include only the MOST important 6 or fewer), local_context, hiring_advice.\n"
            "RULES:\n"
            "- Do NOT use 'estimated', 'approx', 'approximate' or similar words.\n"
            "- Keep required_skills focused (max 6), the most critical for hiring decisions.\n"
            "- Return only the JSON object and nothing else.\n\n"
            "JOB POSTING JSON:\n" + json.dumps(latest, ensure_ascii=False) + "\n\n"
        )

        if self.llm:
            # generated = self.llm.generate(prompt, temperature=0)
            generated = self.llm.generate(prompt)
        else:
            generated = ""

        write_debug_file("task2_raw_llm.txt", generated or "")
        logger.info(f"task_analyze_job: LLM returned {len(generated or '')} chars")

        sanitized = self._sanitize_forbidden(generated or "")
        parsed = safe_json_load_from_text(sanitized)

        if parsed is None:
            write_debug_file("task2_llm_unparsed.txt", generated or "")
            logger.warning("task_analyze_job: Could not parse JSON; using fallback structure.")
            parsed = {
                "summary": (sanitized or "")[:300],
                "purpose": "",
                "seniority": "Unknown",
                "required_skills": [],
                "local_context": "",
                "hiring_advice": ""
            }

        # Defensive trimming of required_skills
        req_skills = parsed.get("required_skills") or []
        if isinstance(req_skills, list):
            req_skills = [str(s).strip() for s in req_skills if s][:6]
        else:
            req_skills = []

        return JobAnalysisResult(
            job_id=latest.get("id", -1),
            summary=parsed.get("summary", "") or "",
            purpose=parsed.get("purpose", "") or "",
            seniority=parsed.get("seniority", "Unknown") or "Unknown",
            required_skills=req_skills,
            local_context=parsed.get("local_context", "") or "",
            hiring_advice=parsed.get("hiring_advice", "") or ""
        )

      # -----------------------
    # Task 3: Build CV (Enhanced with cv_languages)
    # -----------------------
    def task_build_cv(self, profile_res: ProfileResult, job_analysis: JobAnalysisResult, cv_languages: Optional[List[str]] = None) -> CVModel:
        """
        Build a complete CV model (JSON + markdown) based on profile and job analysis.
        Optionally generate it in one or more target languages (e.g., ['fr', 'en']).
        """
        # Define the target languages
        target_langs = cv_languages or ["en"]
        lang_instruction = ""
        if len(target_langs) == 1:
            lang_instruction = f"Write the CV entirely in {target_langs[0].upper()}."
        else:
            lang_instruction = (
                "Generate the CV in multiple languages. "
                + ", ".join([l.upper() for l in target_langs])
                + ". Each version should start with '### CV in [LANG]'."
            )

        # Build prompt
        prompt = (
            "You are an expert multilingual CV writer. "
            "Using the PROFILE and JOB ANALYSIS below, produce:\n"
            "1) A strict JSON object with keys: name, email, phone, headline, summary, "
            "experience (list of {title, company, start, end, description}), skills (list - only the most relevant, max 8), "
            "education (list of {degree, institution, year}), languages (list), certifications (list), other (dict).\n"
            "2) On a NEW LINE, the marker ---MARKDOWN_CV--- and then a readable markdown CV.\n\n"
            "RULES:\n"
            f"- {lang_instruction}\n"
            "- 'headline' should be a short phrase (3–6 words) tailored to the job.\n"
            "- 'summary' should be professional and adapted to the job posting.\n"
            "- Avoid uncertain terms like 'estimated' or 'approx'.\n"
            "- Return exactly: the JSON, then the marker, then the markdown. Nothing else.\n\n"
            "PROFILE PARAGRAPH:\n" + (profile_res.paragraph or "") + "\n\n"
            "PROFILE EXTRACTED (JSON):\n" + json.dumps(profile_res.extracted or {}, ensure_ascii=False) + "\n\n"
            "JOB ANALYSIS (JSON):\n" + json.dumps(job_analysis.model_dump(), ensure_ascii=False) + "\n\n"
        )

        # Generate
        generated = self.llm.generate(prompt) if self.llm else ""
        write_debug_file("task3_raw_llm.txt", generated or "")
        logger.info(f"task_build_cv: LLM returned {len(generated or '')} chars")

        # Rest of your parsing logic unchanged
        sanitized = self._sanitize_forbidden(generated or "")
        json_part = {}
        md_part = ""

        if sanitized:
            marker = "---MARKDOWN_CV---"
            if marker in sanitized:
                json_text, md_part = sanitized.split(marker, 1)
                json_text = json_text.strip()
            else:
                json_text = sanitized.strip()

            parsed = safe_json_load_from_text(json_text)
            json_part = parsed or {}
        else:
            json_part = {}

        # Select best skills based on job analysis
        reduced_skills = self._choose_most_relevant_skills(
            json_part.get("skills", []),
            job_analysis.required_skills,
            max_n=8
        )

        # Build validated CV
        candidate = {
            "name": json_part.get("name"),
            "email": json_part.get("email"),
            "phone": json_part.get("phone"),
            "headline": json_part.get("headline"),
            "summary": json_part.get("summary") or profile_res.paragraph,
            "experience": json_part.get("experience", []),
            "skills": reduced_skills,
            "education": json_part.get("education", []),
            "languages": json_part.get("languages", target_langs),
            "certifications": json_part.get("certifications", []),
            "other": json_part.get("other", {}),
        }

        # Validate with Pydantic
        try:
            cv = CVModel(**candidate)
        except ValidationError as e:
            logger.warning(f"CV validation failed: {e}")
            cv = CVModel(**{k: v for k, v in candidate.items() if k in CVModel.model_fields})

        # Save both JSON and Markdown outputs
        output_dir = Path("backend/stored_cvs")
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = (cv.name or "cv_unnamed").replace(" ", "_")

        json_path = output_dir / f"{base_name}.json"
        md_path = output_dir / f"{base_name}.md"

        json_path.write_text(cv.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(md_part.strip() or "No Markdown output.", encoding="utf-8")

        logger.info(f"✅ Saved multilingual CV ({', '.join(target_langs)}): {json_path} / {md_path}")
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

import json
import logging
from pathlib import Path

from typing import Tuple, Any, Dict

# Your project LLM / agent / process imports (adjust module paths as needed)
# from your_agent_module import Agent, GeminiClient, ProfileResult, JobAnalysisResult, CVModel
# from your_process_framework import Process

# Import the PDF builder function from create_cv.py (must be in PYTHONPATH)
from backend.create_cv import create_dynamic_pdf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OUTPUT path used by pipeline to save the clean generated JSON
OUTPUT_JSON = Path("backend/utils/generated_cv.json")
JOBS_JSON = Path("backend/utils/jobs_informations.json")


def load_latest_data_from_utils() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load the generated CV JSON and the latest job info from backend/utils.

    Returns (cv_data, job_info).
    Raises FileNotFoundError if files are missing or JSON parse errors.
    """
    base_dir = Path(__file__).resolve().parent  # directory containing crewai_main.py
    utils_dir = base_dir / "utils"

    cv_path = utils_dir / "generated_cv.json"
    jobs_path = utils_dir / "jobs_informations.json"

    logger.info(f"Looking for generated CV at: {cv_path}")
    logger.info(f"Looking for jobs JSON at: {jobs_path}")

    if not utils_dir.exists():
        raise FileNotFoundError(f"'utils' directory does not exist at {utils_dir}")

    if not cv_path.exists():
        raise FileNotFoundError(f"generated_cv.json not found at {cv_path}")
    if not jobs_path.exists():
        raise FileNotFoundError(f"jobs_informations.json not found at {jobs_path}")

    # Read CV JSON
    try:
        cv_text = cv_path.read_text(encoding="utf-8")
        cv_data = json.loads(cv_text)
    except Exception as e:
        raise RuntimeError(f"Failed to read/parse {cv_path}: {e}") from e

    # Read jobs JSON and select last entry
    try:
        jobs_text = jobs_path.read_text(encoding="utf-8")
        jobs = json.loads(jobs_text)
    except Exception as e:
        raise RuntimeError(f"Failed to read/parse {jobs_path}: {e}") from e

    if isinstance(jobs, list):
        job_info = jobs[-1] if jobs else {}
    elif isinstance(jobs, dict):
        job_info = jobs
    else:
        job_info = {}

    logger.info("Loaded CV and job info from utils successfully.")
    print(f"CV data elements : {cv_data}")
    print(f"Job info elements : {job_info}")
    return cv_data, job_info

from pathlib import Path
import json

import logging

logger = logging.getLogger(__name__)

def run_job_analyzer():
    PROFILE_JSON = "backend/utils/profile.json"
    JOBS_JSON = "backend/utils/jobs_informations.json"
    PDF_PATH = "uploads/uploaded_profile_cv.pdf"
    OUTPUT_JSON = "backend/utils/generated_cv.json"

    # Initialize Gemini client and agent (adjust to your actual constructors)
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

    logger.info("▶️ Starting pipeline process...")
    final_state = p.run({"start": True})
    logger.info("▶️ Pipeline finished.")

    # Get final CV result
    cv = final_state.get("step3_build_cv")

    if isinstance(cv, CVModel):
        try:
            # Save clean JSON in backend/utils (optional for debugging)
            cv_data = cv.model_dump()
            out_path = Path(OUTPUT_JSON)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(cv_data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"💾 Clean CV JSON saved to {out_path}")

            # Try to reload data safely
            try:
                cv_loaded, job_info = load_latest_data_from_utils()
            except Exception as e:
                logger.warning(f"Could not reload files from utils: {e}. Using memory fallback.")
                cv_loaded = cv_data
                jobs_path = Path(JOBS_JSON)
                if jobs_path.exists():
                    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
                    job_info = jobs[-1] if isinstance(jobs, list) and jobs else jobs
                else:
                    job_info = {}

            # -------------------------------
            # 🗂️ Save PDF inside stored_cvs/
            # -------------------------------
            stored_dir = Path("backend/stored_cvs")
            stored_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            base_name = (cv.name or "generated_cv").replace(" ", "_")
            pdf_filename = stored_dir / f"{base_name}_{timestamp}.pdf"

            logger.info(f"🖨️ Generating PDF at {pdf_filename} ...")
            try:
                create_dynamic_pdf(cv_loaded, job_info, str(pdf_filename))
                logger.info(f"✅ PDF created: {pdf_filename}")
            except Exception as e:
                logger.exception(f"Failed to create PDF from CV/job info: {e}")

        except Exception as e:
            logger.exception(f"Failed saving generated CV JSON: {e}")
    else:
        logger.warning("⚠️ Step 3 did not return a valid CVModel. Please check debug outputs for raw LLM responses.")

    logger.info("🎯 run_job_analyzer completed.")


if __name__ == "__main__":
    run_job_analyzer()
