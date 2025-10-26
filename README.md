# 🧠 CrewAI CV Project

This project is an AI-powered CV generator and job analyzer built with **FastAPI**, **CrewAI**, and **Gemini/LLM agents**.  
It analyzes job descriptions, generates tailored CVs dynamically as PDFs, and supports multi-step AI agent workflows.

---

## 🚀 Features

- Job description analysis using LLM agents  
- Dynamic CV generation (PDF with ReportLab)  
- Modular architecture (agents, utils, backend)  
- JSON-based storage for job info and CV data  
- FastAPI backend ready for API integrations  
- Supports sequential AI task pipelines  

---

## 🧩 Project Structure

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/CrewAI_CV_Project.git
cd CrewAI_CV_Project



2. Create and Activate Virtual Environment
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Application
uvicorn backend.app:app --reload --port 8000
