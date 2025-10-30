# 🧠 CrewAI CV Project

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![License](https://img.shields.io/badge/license-MIT-blue)
![Open Source](https://img.shields.io/badge/Open%20Source-100%25-brightgreen)

**An open-source AI-powered CV generator that creates perfectly tailored resumes in seconds using CrewAI multi-agent system.**

Save hours of CV customization with intelligent AI that analyzes jobs and generates professional PDFs instantly.

✨ **Generate tailored CVs in ~10 seconds** • 🆓 **100% Free with Gemini API** • 🎨 **Fully Customizable**

</div>


---

## 📋 Table of Contents

1. [Demo Video](#-demo-video)
2. [About](#-about)
3. [Screenshots](#-screenshots)
4. [Features](#-features)
5. [How It Works](#-how-it-works)
6. [Project Structure](#-project-structure)
7. [Technologies Used](#-technologies-used)
8. [Installation](#%EF%B8%8F-installation)
9. [Usage](#-usage)
10. [API Reference](#-api-reference)
11. [CrewAI Agents](#-crewai-agents)
12. [License](#-license)
13. [Contact](#-contact)
14. [Acknowledgements](#-acknowledgements)

---

## 🎥 Demo Video

> **Coming Soon!** - A complete walkthrough video will be added here soon.

**What the demo will show:**
1. Setting up API keys via the web interface
2. Uploading a CV and creating a profile
3. Pasting a job description and customizing design
4. Watching the AI generate a tailored CV in real-time
5. Downloading and reviewing the final PDF


---

## 🎯 About

### The Problem
Creating tailored CVs for multiple job applications is **time-consuming**:
- ⏱️ Manual CV adaptation takes 15-30 minutes per application
- 🎨 Professional formatting requires design skills
- 📝 Matching skills to job requirements is repetitive

### The Solution
An **open-source, non-profit solution** using **CrewAI multi-agent AI system**:
- ⚡ **~10 seconds per CV** - Fast generation with AI processing
- 🤖 **3-Agent Pipeline** - Profile extraction → Job analysis → CV building
- 🎨 **Professional PDFs** - ReportLab-generated with custom styling
- 🆓 **100% Free** - Use Gemini API at no cost
- 🌐 **Open Source** - Community-driven project

---

## 📸 Screenshots

### API Key Configuration
Manage multiple AI service providers (Gemini, OpenAI, Anthropic, SerpAPI, Custom Search).

<img width="1552" height="860" alt="API Keys Configuration" src="https://github.com/user-attachments/assets/a2efd851-193a-4810-9cb3-3f464eb9f37b" />

### Profile Management
Upload CV (PDF) for automatic extraction and add detailed description.

<img width="1381" height="869" alt="Profile Management" src="https://github.com/user-attachments/assets/215991c3-6a90-4fa3-8040-71e2c8300a80" />

### CV Generation
Paste job description, customize colors/fonts, select language, add photo.

<img width="1022" height="882" alt="CV Generation" src="https://github.com/user-attachments/assets/548aa07f-623a-426e-9539-410e1c0ac18e" />

### Results & Downloads
View, download, and manage all generated CVs.

<img width="1540" height="696" alt="Results Page" src="https://github.com/user-attachments/assets/38fec3d5-6bb4-44bc-9c91-6d032ef8f4d3" />

---

## ✨ Features

### CV Customization
- 🎨 **Custom Colors** - Primary and secondary color themes
- 🔤 **Font Selection** - Arial, Times-Roman, and more
- 🌍 **Multi-Language** - French, English support (configurable)
- 📸 **Profile Photo** - Optional circular profile image with border
- 📐 **Professional Layout** - Two-column design with optimized spacing

### Intelligent Processing
- 🤖 **Automatic PDF Reading** - Extracts data from uploaded CVs using PyPDF2
- 📊 **Job Analysis** - Analyzes descriptions to extract key requirements
- 🎯 **Skill Matching** - Prioritizes relevant skills (max 8) based on job needs
- 📝 **Dynamic Content** - Adapts experience and summary to job posting
- ✅ **ATS-Friendly** - Clean structure for Applicant Tracking Systems

### Data Management
- 💾 **Profile Storage** - Saves user profile in JSON format
- 📁 **Job History** - Tracks all job postings with metadata
- 📦 **CV Library** - All generated CVs stored with timestamps
- 🔐 **API Key Management** - Secure storage in JSON and `.env` sync
- 🗂️ **File Organization** - Organized uploads and outputs

### Technical Features
- ⚡ **Background Processing** - Non-blocking CV generation via threading
- 🌐 **CORS Enabled** - Ready for frontend integration
- 📄 **PDF Generation** - ReportLab with precise layout control
- 🔄 **Auto-reload** - Development mode with live updates
- 📚 **API Documentation** - Auto-generated Swagger/ReDoc

---

## 🔄 How It Works

### Three-Agent Sequential Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  USER INPUT: Job Description + Preferences              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  AGENT 1: Profile Miner                                 │
│  • Reads backend/utils/profile.json                     │
│  • Extracts text from uploads/uploaded_profile_cv.pdf   │
│  • Uses PyPDF2 for PDF parsing                          │
│  • Outputs: Structured profile data + paragraph summary │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  AGENT 2: Job Analyzer                                  │
│  • Reads backend/utils/jobs_informations.json (latest)  │
│  • Extracts required skills (max 6 most important)      │
│  • Identifies seniority level, purpose, context         │
│  • Outputs: Job requirements + hiring insights          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  AGENT 3: CV Builder                                    │
│  • Matches profile to job requirements                  │
│  • Selects top 8 relevant skills                        │
│  • Generates headline + tailored summary                │
│  • Creates both JSON and Markdown outputs               │
│  • Saves to backend/stored_cvs/                         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  PDF GENERATION (create_cv.py)                          │
│  • Loads generated_cv.json + job_informations.json      │
│  • ReportLab creates A4 PDF with custom styling         │
│  • Circular profile photo with border (if enabled)      │
│  • Two-column layout: main + sidebar                    │
│  • Saves to backend/stored_cvs/ with timestamp          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
               READY!
```

**Process Details:**
- **Sequential Execution**: Each agent's output feeds the next
- **Background Threading**: FastAPI spawns daemon thread for non-blocking generation
- **Data Flow**: profile.json → jobs_informations.json → generated_cv.json → PDF
- **Error Handling**: Debug outputs saved to `debug_outputs/` folder

---

## 📁 Project Structure

```
CREWAI_PROJET/
│
├── backend/
│   ├── __pycache__/             # Python cache (gitignored)
│   ├── stored_cvs/              # Generated CV PDFs (gitignored)
│   │   ├── *.pdf                # Final CV outputs with timestamps
│   │   ├── *.json               # CV data in JSON format
│   │   └── *.md                 # CV in Markdown format
│   │
│   ├── uploads/                 # User uploads (gitignored)
│   │   ├── photos/              # Profile photos
│   │   └── uploaded_profile_cv.pdf  # Fixed name for uploaded CV
│   │
│   ├── utils/                   # Data storage
│   │   ├── api_keys.json        # API keys (gitignored)
│   │   ├── profile.json         # User profile (gitignored)
│   │   ├── jobs_informations.json   # Jobs database (gitignored)
│   │   └── generated_cv.json    # Latest CV output (gitignored)
│   │
│   ├── __init__.py
│   ├── app.py                   # FastAPI application (15 endpoints)
│   ├── create_cv.py             # PDF generation with ReportLab
│   └── crewai_main.py           # 3-agent pipeline orchestration
│
├── debug_outputs/               # Debug logs from agents
│   ├── task1_*.txt/json         # Agent 1 outputs
│   ├── task2_*.txt/json         # Agent 2 outputs
│   └── task3_*.txt              # Agent 3 outputs
│
├── frontend/
│   ├── css/
│   │   └── style.css            # UI styling
│   ├── js/
│   │   └── script.js            # Frontend logic
│   └── index.html               # Main interface
│
├── generated_results/           # Historical CV generations
│   └── generated_cv_*.json      # Timestamped outputs
│
├── uploads/                     # Alternative upload location
│   └── photos/                  # Profile images
│
├── venv/                        # Virtual environment (gitignored)
├── .env                         # Environment variables (gitignored)
├── cv_tech_template.pdf         # Template reference
├── generated_cv.json            # Root level output
├── generated_cv.md              # Markdown version
├── jobs_informations.json       # Jobs at root level
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

---

## 🛠️ Technologies Used

### Backend Stack
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern async Python web framework
- **[Uvicorn](https://www.uvicorn.org/)** - ASGI server
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation (v2)
- **[Python-dotenv](https://pypi.org/project/python-dotenv/)** - Environment management

### AI & Processing
- **[Google Gemini](https://deepmind.google/technologies/gemini/)** (gemini-2.0-flash) - Primary LLM
- **[google-genai](https://pypi.org/project/google-genai/)** - Python SDK for Gemini
- **[PyPDF2](https://pypdf2.readthedocs.io/)** - PDF text extraction

### PDF Generation
- **[ReportLab](https://www.reportlab.com/)** - PDF creation library
- **[Pillow (PIL)](https://pillow.readthedocs.io/)** - Image processing for circular photos

### Frontend
- **HTML5 + CSS3** - Clean UI
- **JavaScript (Vanilla)** - Dynamic interactions
- **No frameworks** - Lightweight and fast

### Supported AI APIs
- **Gemini API** - Primary (free tier available)
- **OpenAI API** - GPT models
- **Anthropic API** - Claude models
- **SerpAPI** - Web search
- **Custom Search API** - Google search

---

## ⚙️ Installation

### Prerequisites
- **Python 3.8+** (tested with 3.10+)
- **pip** package manager
- **Git**
- **Gemini API key** (free at [Google AI Studio](https://makersuite.google.com/app/apikey))

### Step 1: Clone Repository

```bash
git clone https://github.com/motawakil/CrewAI_CV_Project.git
cd CrewAI_CV_Project
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages installed:**
```
fastapi
uvicorn[standard]
pydantic
python-dotenv
google-genai
PyPDF2
reportlab
pillow
python-multipart
```

### Step 4: Create Directories

```bash
mkdir -p backend/utils backend/uploads/photos backend/stored_cvs debug_outputs
```

### Step 5: Configure API Keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Or configure via web interface after starting the app.**

### Step 6: Run the Application

```bash
uvicorn backend.app:app --reload --port 8000
```

The application will start at `http://localhost:8000`

### Step 7: Access the Interface

- **Web UI:** Open `frontend/index.html` in your browser
- **API Docs:** `http://localhost:8000/docs` (Swagger)
- **Alternative Docs:** `http://localhost:8000/redoc`

---

## 💻 Usage

### Quick Start

1. **Configure API Key**
   - Open web interface
   - Navigate to **Paramètres** (Settings)
   - Enter your Gemini API key
   - Click **Sauvegarder** (Save)

2. **Set Up Profile**
   - Go to **Profil** section
   - Upload your CV (PDF) - saves as `uploaded_profile_cv.pdf`
   - Write a detailed description (2000 char limit)
   - Click **Sauvegarder le Profil**

3. **Generate CV**
   - Navigate to **Générer** (Generate)
   - Paste job description (truncated to 2000 chars)
   - Add job URL (optional)
   - Choose colors (default: `#6366f1`, `#ec4899`)
   - Select language: `fr` or `en`
   - Pick font: `Arial` or `Times-Roman`
   - Upload photo (optional)
   - Click generate
   - **Wait ~10-15 seconds** (background processing)

4. **Download CV**
   - Go to **Résultats** (Results)
   - See all generated CVs with timestamps
   - Click **Télécharger** to download PDF
   - Click delete icon to remove CV

---

## 🔌 API Reference

### API Key Management

```http
GET  /api/load-keys/           # Load saved API keys
POST /api/save-keys/           # Save API keys (JSON + .env)
POST /api/reset-keys/          # Reset all keys to null
```

### Profile Management

```http
POST /api/save-profile/        # Save profile (Form: description, cvFile)
GET  /api/load-profile/        # Load profile data
POST /api/upload-cv/           # Upload PDF (saves as uploaded_profile_cv.pdf)
POST /api/reset-profile/       # Delete profile.json
```

### Job & CV Generation

```http
POST   /api/save-job-info/     # Save job + trigger CV generation (background)
GET    /api/load-jobs/         # Get all jobs from jobs_informations.json
POST   /api/upload-photo/      # Upload profile photo
GET    /api/list-cvs/          # List all PDFs in backend/stored_cvs/
GET    /api/download-cv/{filename}  # Download specific PDF
DELETE /api/delete-cv/{filename}    # Delete CV file
```

### Static Files

```http
GET /uploads/{path}            # Access uploaded photos and files
```

### Request Examples

**Save Job Info:**
```bash
curl -X POST "http://localhost:8000/api/save-job-info/" \
  -F "jobDescription=Senior Python Developer with AI experience..." \
  -F "jobLink=https://company.com/job" \
  -F "primaryColor=#6366f1" \
  -F "secondaryColor=#ec4899" \
  -F "language=fr" \
  -F "font=Arial" \
  -F "hasPhoto=true"
```

**List Generated CVs:**
```bash
curl "http://localhost:8000/api/list-cvs/"
```

**Response:**
```json
{
  "status": "ok",
  "cvs": [
    {
      "id": 0,
      "filename": "Name_20251026_204620.pdf",
      "title": "Name_20251026_204620",
      "timestamp": 1730000000,
      "url": "/api/download-cv/Name_20251026_204620.pdf"
    }
  ]
}
```

---

## 🤖 CrewAI Agents

### Agent 1: Profile Miner

**File:** `crewai_main.py` → `task_build_profile()`

**Purpose:** Extract and structure candidate profile data

**Inputs:**
- `backend/utils/profile.json` - User description
- `uploads/uploaded_profile_cv.pdf` - Uploaded CV

**Process:**
- Uses PyPDF2's `PdfReader` to extract PDF text
- Sends profile JSON + PDF text to Gemini
- Prompts for strict JSON extraction
- Forbids uncertain words ("estimated", "approx")
- Limits `top_skills` to 8 items

**Outputs:**
- `ProfileResult` with:
  - `paragraph` - Professional summary (3-5 sentences)
  - `extracted` - Dict with name, email, phone, top_skills, education, etc.
  - `pdf_preview` - First 1000 chars of PDF

**Debug Files:**
- `debug_outputs/task1_profile.json`
- `debug_outputs/task1_pdf_excerpt.txt`
- `debug_outputs/task1_raw_llm.txt`

### Agent 2: Job Analyzer

**File:** `crewai_main.py` → `task_analyze_job()`

**Purpose:** Analyze job posting and extract requirements

**Inputs:**
- `backend/utils/jobs_informations.json` - Latest job (by `createdAt`)

**Process:**
- Sorts jobs by timestamp, selects most recent
- Sends job JSON to Gemini
- Requests structured analysis
- Limits `required_skills` to 6 most critical

**Outputs:**
- `JobAnalysisResult` with:
  - `job_id` - Job ID from JSON
  - `summary` - 1-2 sentence overview
  - `purpose` - Role purpose
  - `seniority` - Junior/Mid/Senior
  - `required_skills` - Top 6 skills (list)
  - `local_context` - Geographic/cultural context
  - `hiring_advice` - Recommendations

**Debug Files:**
- `debug_outputs/task2_latest_job.json`
- `debug_outputs/task2_raw_llm.txt`

### Agent 3: CV Builder

**File:** `crewai_main.py` → `task_build_cv()`

**Purpose:** Generate tailored CV matching profile to job

**Inputs:**
- `ProfileResult` from Agent 1
- `JobAnalysisResult` from Agent 2
- `cv_languages` - Target languages (default: `["fr", "en"]`)

**Process:**
- Combines profile + job analysis in single prompt
- Requests JSON (CVModel structure) + Markdown
- Selects top 8 relevant skills using `_choose_most_relevant_skills()`
- Validates output with Pydantic `CVModel`
- Generates multilingual versions if requested

**Outputs:**
- `CVModel` (Pydantic) with fields:
  - `name`, `email`, `phone`
  - `headline` - Short job-adapted phrase (3-6 words)
  - `summary` - Professional summary adapted to job
  - `experience` - List of dicts (title, company, start, end, description)
  - `skills` - Top 8 relevant skills
  - `education` - List of dicts (degree, institution, year)
  - `languages` - List of languages
  - `certifications` - List
  - `other` - Dict for additional fields

**Saved Files:**
- `backend/stored_cvs/{name}.json` - CV data
- `backend/stored_cvs/{name}.md` - Markdown version
- `backend/utils/generated_cv.json` - Latest CV (clean)

**Debug Files:**
- `debug_outputs/task3_raw_llm.txt`

### Pipeline Orchestration

**Function:** `run_job_analyzer()` in `crewai_main.py`

**Triggered by:** `POST /api/save-job-info/` via background thread

**Sequential Steps:**
1. Initialize `GeminiClient(model="gemini-2.0-flash")`
2. Create `Agent` with LLM client
3. Create `Process("profile_job_cv_pipeline")`
4. Add 3 steps: build_profile → analyze_job → build_cv
5. Execute with `process.run({"start": True})`
6. Save outputs to `backend/utils/generated_cv.json`
7. Call `create_dynamic_pdf()` to generate PDF
8. Save to `backend/stored_cvs/{name}_{timestamp}.pdf`

**Threading:** 
```python
import threading
threading.Thread(target=run_job_analyzer, daemon=True).start()
```

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Motaouakel El Maimouni

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 Contact

**Motaouakel El Maimouni**
- GitHub: [@motawakil](https://github.com/motawakil)
- Project: [https://github.com/motawakil/CrewAI_CV_Project](https://github.com/motawakil/CrewAI_CV_Project)

---

## 🙏 Acknowledgements

- **[CrewAI](https://www.crewai.com/)** - Multi-agent orchestration
- **[FastAPI](https://fastapi.tiangolo.com/)** - Web framework
- **[Google Gemini](https://deepmind.google/technologies/gemini/)** - AI model
- **[ReportLab](https://www.reportlab.com/)** - PDF generation
- **[PyPDF2](https://pypdf2.readthedocs.io/)** - PDF parsing

---

<div align="center">

⭐ **Star this repository if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/motawakil/CrewAI_CV_Project?style=social)](https://github.com/motawakil/CrewAI_CV_Project)

**Made with ❤️ for the job-seeking community**

---

**Stop wasting hours on CV formatting. Start applying to more jobs today!**

[**Get Started**](#%EF%B8%8F-installation) • [**View Demo**](#-demo-video) • [**Read Docs**](#-api-reference)

</div>
