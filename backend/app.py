from fastapi import FastAPI, Request , UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from dotenv import load_dotenv
from .crewai_main import run_job_analyzer  # <-- importer la fonction
from fastapi.responses import JSONResponse
# Charger .env
load_dotenv()

app = FastAPI()

# 🔓 Activer CORS pour autoriser ton frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← en dev seulement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fichier de sauvegarde des clés locales
KEYS_FILE = "backend/utils/api_keys.json"

class APIKeys(BaseModel):
    gemini: str | None = None
    openai: str | None = None
    anthropic: str | None = None
    serpapi: str | None = None
    customsearch: str | None = None


def load_local_keys():
    """Lire les clés depuis le fichier JSON local."""
    if os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"gemini": None, "openai": None, "anthropic": None, "serpapi": None, "customsearch": None}


def save_local_keys(keys: dict):
    """Sauvegarder les clés dans le JSON et mettre à jour le .env."""
    os.makedirs(os.path.dirname(KEYS_FILE), exist_ok=True)
    with open(KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=4)

    # 🔁 Mettre à jour le fichier .env
    lines = []
    with open(".env", "r", encoding="utf-8") as env_file:
        for line in env_file:
            if line.startswith("GEMINI_API_KEY="):
                line = f"GEMINI_API_KEY={keys.get('gemini','')}\n"
            elif line.startswith("OPENAI_API_KEY="):
                line = f"OPENAI_API_KEY={keys.get('openai','')}\n"
            elif line.startswith("ANTHROPIC_API_KEY="):
                line = f"ANTHROPIC_API_KEY={keys.get('anthropic','')}\n"
            elif line.startswith("SERPAPI_API_KEY="):
                line = f"SERPAPI_API_KEY={keys.get('serpapi','')}\n"
            elif line.startswith("CUSTOMSEARCH_API_KEY="):
                line = f"CUSTOMSEARCH_API_KEY={keys.get('customsearch','')}\n"

            lines.append(line)

    with open(".env", "w", encoding="utf-8") as env_file:
        env_file.writelines(lines)


@app.get("/api/load-keys/")
async def load_keys():
    try:
        keys = load_local_keys()
        return {"status": "ok", "keys": keys}
    except Exception as e:
        print("Error loading keys:", e)
        return {"status": "error", "message": str(e)}


@app.post("/api/save-keys/")
async def save_keys(data: APIKeys):
    try:
        keys_dict = data.dict()
        save_local_keys(keys_dict)
        return {"status": "ok", "message": "API keys saved successfully"}
    except Exception as e:
        print("Error saving keys:", e)
        return {"status": "error", "message": str(e)}


@app.post("/api/reset-keys/")
async def reset_keys():
    try:
        keys = {"gemini": None, "openai": None, "anthropic": None, "serpapi": None, "customsearch": None}
        save_local_keys(keys)
        return {"status": "ok", "message": "API keys reset successfully"}
    except Exception as e:
        print("Error resetting keys:", e)
        return {"status": "error", "message": str(e)}







PROFILE_FILE = "backend/utils/profile.json"
UPLOAD_DIR = "backend/uploads"

# -------------------- PROFILE MANAGEMENT -------------------- #

@app.post("/api/save-profile/")
async def save_profile(description: str = Form(...), cvFile: str = Form("")):
    """
    Sauvegarde la description et le nom du fichier CV.
    """
    try:
        profile_data = {
            "cvFile": cvFile,
            "description": description,
            "savedAt": __import__("datetime").datetime.now().isoformat()
        }
        os.makedirs(os.path.dirname(PROFILE_FILE), exist_ok=True)
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=4)

        return {"status": "ok", "message": "Profil sauvegardé avec succès", "profile": profile_data}
    except Exception as e:
        print("Error saving profile:", e)
        return {"status": "error", "message": str(e)}


@app.get("/api/load-profile/")
async def load_profile():
    """
    Charge les informations du profil.
    """
    try:
        if not os.path.exists(PROFILE_FILE):
            return {"status": "ok", "profile": {"cvFile": None, "description": ""}}
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"status": "ok", "profile": data}
    except Exception as e:
        print("Error loading profile:", e)
        return {"status": "error", "message": str(e)}


@app.post("/api/upload-cv/")
async def upload_cv(file: UploadFile = File(...)):
    """
    Upload le CV PDF vers le dossier local.
    """
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        return {"status": "ok", "message": "CV uploadé avec succès", "fileName": file.filename}
    except Exception as e:
        print("Error uploading CV:", e)
        return {"status": "error", "message": str(e)}


@app.post("/api/reset-profile/")
async def reset_profile():
    """
    Réinitialise les informations du profil.
    """
    try:
        if os.path.exists(PROFILE_FILE):
            os.remove(PROFILE_FILE)
        return {"status": "ok", "message": "Profil réinitialisé"}
    except Exception as e:
        print("Error resetting profile:", e)
        return {"status": "error", "message": str(e)}




from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
import os, json
from datetime import datetime


# --- Paths
JOBS_FILE = "backend/utils/jobs_informations.json"
PHOTO_DIR = "backend/uploads/photos"


# --- CV GENERATION & JOB INFO STORAGE
@app.post("/api/save-job-info/")
async def save_job_info(
    jobDescription: str = Form(""),
    jobLink: str = Form(""),
    primaryColor: str = Form("#6366f1"),
    secondaryColor: str = Form("#ec4899"),
    language: str = Form("fr"),
    font: str = Form("Arial"),
    hasPhoto: str = Form("false")
):
    print("\n=== [DEBUG] /api/save-job-info/ called ===")
    print(f"[DEBUG] jobDescription (len={len(jobDescription)}): {jobDescription[:80]}")
    print(f"[DEBUG] jobLink: {jobLink}")
    print(f"[DEBUG] primaryColor: {primaryColor}, secondaryColor: {secondaryColor}")
    print(f"[DEBUG] language: {language}, font: {font}")
    print(f"[DEBUG] hasPhoto (raw): {hasPhoto}")

    try:
        os.makedirs(os.path.dirname(JOBS_FILE), exist_ok=True)

        # Charger JSON existant
        if os.path.exists(JOBS_FILE):
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    print(f"[DEBUG] Existing JSON loaded ({len(data)} jobs).")
                except json.JSONDecodeError:
                    print("[ERROR] Corrupted JSON file, resetting it.")
                    data = []
        else:
            print("[DEBUG] JSON file not found, creating new.")
            data = []

        # Conversion du booléen
        hasPhoto_bool = str(hasPhoto).lower() in ["true", "1", "yes"]

        # Création de la nouvelle entrée
        new_job = {
            "id": len(data) + 1,
            "jobDescription": jobDescription[:2000],
            "jobLink": jobLink,
            "primaryColor": primaryColor,
            "secondaryColor": secondaryColor,
            "language": language,
            "font": font,
            "hasPhoto": hasPhoto_bool,
            "createdAt": datetime.now().isoformat()
        }

        data.append(new_job)

        with open(JOBS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"[DEBUG] Job successfully saved (total={len(data)}).")

        # ------------------------ 🔹 Lancer CrewAI ------------------------
        # On lance l'analyse du dernier job, en asynchrone pour ne pas bloquer
        import threading
        threading.Thread(target=run_job_analyzer, daemon=True).start()
        print("[DEBUG] CrewAI job analyzer started in background thread.")

        return {"status": "ok", "message": "Job ajouté avec succès", "job": new_job}

    except Exception as e:
        print("[ERROR] Exception in save_job_info:", e)
        return {"status": "error", "message": str(e)}
    





@app.post("/api/upload-photo/")
async def upload_photo(file: UploadFile = File(...)):
    print("\n=== [DEBUG] /api/upload-photo/ called ===")
    try:
        os.makedirs(PHOTO_DIR, exist_ok=True)
        file_path = os.path.join(PHOTO_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        print(f"[DEBUG] Photo saved: {file.filename} ({len(content)} bytes)")
        return {"status": "ok", "message": "Photo uploadée avec succès", "fileName": file.filename}

    except Exception as e:
        print("[ERROR] upload_photo exception:", e)
        return {"status": "error", "message": str(e)}


@app.get("/api/load-jobs/")
async def load_jobs():
    print("\n=== [DEBUG] /api/load-jobs/ called ===")
    try:
        if not os.path.exists(JOBS_FILE):
            print("[DEBUG] jobs_informations.json not found → returning empty list.")
            return {"status": "ok", "jobs": []}
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[DEBUG] Loaded {len(data)} jobs from file.")
        return {"status": "ok", "jobs": data}
    except Exception as e:
        print("[ERROR] load_jobs exception:", e)
        return {"status": "error", "message": str(e)}

# --- Static serving for photos


app.mount("/uploads", StaticFiles(directory="backend/uploads"), name="uploads")


from fastapi.responses import FileResponse, JSONResponse


STORED_CVS_DIR = "backend/stored_cvs"

# ------------------------ 🔹 Lister les CVs disponibles ------------------------ #
@app.get("/api/list-cvs/")
async def list_cvs():
    """
    Retourne la liste des CVs PDF disponibles dans le dossier backend/stored_cvs/
    """
    try:
        if not os.path.exists(STORED_CVS_DIR):
            os.makedirs(STORED_CVS_DIR)

        files = [
            {
                "id": i,
                "filename": f,
                "title": os.path.splitext(f)[0],
                "date": os.path.getmtime(os.path.join(STORED_CVS_DIR, f)),
                "timestamp": int(os.path.getmtime(os.path.join(STORED_CVS_DIR, f))),
                "url": f"/api/download-cv/{f}"
            }
            for i, f in enumerate(sorted(os.listdir(STORED_CVS_DIR), reverse=True))
            if f.lower().endswith(".pdf")
        ]

        return {"status": "ok", "cvs": files}

    except Exception as e:
        print("Error listing CVs:", e)
        return {"status": "error", "message": str(e)}


# ------------------------ 🔹 Télécharger un CV PDF ------------------------ #
@app.get("/api/download-cv/{filename}")
async def download_cv(filename: str):
    """
    Permet de télécharger un CV PDF spécifique depuis backend/stored_cvs/
    """
    file_path = os.path.join(STORED_CVS_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "File not found"})
    return FileResponse(file_path, media_type="application/pdf", filename=filename)

@app.delete("/api/delete-cv/{filename}")
async def delete_cv(filename: str):
    try:
        file_path = os.path.join(STORED_CVS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"status": "ok", "message": f"{filename} supprimé"}
        return {"status": "error", "message": "Fichier non trouvé"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
