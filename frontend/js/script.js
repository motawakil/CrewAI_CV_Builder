        // Storage keys
        const STORAGE_KEYS = {
            apiKeys: 'cvgen_api_keys',
            profile: 'cvgen_profile',
            cvs: 'cvgen_generated_cvs'
        };

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadAPIKeys();
            loadProfile();
            loadResults();
            setupDragAndDrop();
        });

        // Navigation
        function showSection(sectionId) {
            // Hide all sections
            document.querySelectorAll('.section').forEach(section => {
                section.classList.remove('active');
            });
            
            // Remove active class from all nav buttons
            document.querySelectorAll('.nav-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Show selected section
            document.getElementById(sectionId).classList.add('active');
            
            // Activate corresponding nav button
            event.target.classList.add('active');
            
            // Refresh results if viewing results section
            if (sectionId === 'results') {
                loadResults();
            }
        }


        // ------------------------------- API Keys Management ------------------------------- // 


const API_BASE = "http://127.0.0.1:8000";

async function saveAPIKeys() {
    const apiKeys = {
        gemini: document.getElementById('gemini-key').value.trim(),
        openai: document.getElementById('openai-key').value.trim(),
        anthropic: document.getElementById('anthropic-key').value.trim(),
        serpapi: document.getElementById('serpapi-key').value.trim(),
        customsearch: document.getElementById('customsearch-key').value.trim(),
    };

    try {
        const res = await fetch(`${API_BASE}/api/save-keys/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(apiKeys),
        });

        const data = await res.json();
        if (data.status === "ok") {
            showToast("✅ Clés API sauvegardées avec succès !", "success");
        } else {
            showToast("⚠️ Erreur lors de la sauvegarde", "error");
        }
    } catch (err) {
        console.error(err);
        showToast("❌ Erreur réseau lors de la sauvegarde", "error");
    }
}

async function resetAPIKeys() {
    if (!confirm("⚠️ Réinitialiser toutes les clés API ?")) return;

    const res = await fetch(`${API_BASE}/api/reset-keys/`, { method: "POST" });
    const data = await res.json();
    showToast("🔄 " + data.message, "success");

    document.getElementById('gemini-key').value = '';
    document.getElementById('openai-key').value = '';
    document.getElementById('anthropic-key').value = '';
    document.getElementById('serpapi-key').value = '';
    document.getElementById('customsearch-key').value = '';
}

async function loadAPIKeys() {
    try {
        const res = await fetch(`${API_BASE}/api/load-keys/`);
        const data = await res.json();
        if (data.status === "ok") {
            const keys = data.keys;
            if (keys.gemini) document.getElementById('gemini-key').value = keys.gemini;
            if (keys.openai) document.getElementById('openai-key').value = keys.openai;
            if (keys.anthropic) document.getElementById('anthropic-key').value = keys.anthropic;
            if (keys.serpapi) document.getElementById('serpapi-key').value = keys.serpapi;
            if (keys.customsearch) document.getElementById('customsearch-key').value = keys.customsearch;
        }
    } catch (err) {
        console.error("Failed to load keys:", err);
    }
}

// Load existing keys when page starts
document.addEventListener("DOMContentLoaded", loadAPIKeys);



        // ------------------------------- Profile Management ------------------------------- // 


        // Profile Management
// 🔹 Sauvegarde du profil
async function saveProfile() {
    const cvFile = document.getElementById('cv-file-name').textContent || '';
    const description = document.getElementById('personal-description').value;

    const formData = new FormData();
    formData.append('cvFile', cvFile);
    formData.append('description', description);

    try {
        const res = await fetch(`${API_BASE}/api/save-profile/`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (data.status === "ok") {
            showToast('✅ Profil sauvegardé avec succès !', 'success');
        } else {
            showToast('⚠️ Erreur lors de la sauvegarde du profil', 'error');
        }
    } catch (err) {
        console.error(err);
        showToast('❌ Erreur réseau lors de la sauvegarde', 'error');
    }
}



// 🔹 Chargement du profil
async function loadProfile() {
    try {
        const res = await fetch(`${API_BASE}/api/load-profile/`);
        const data = await res.json();
        if (data.status === "ok" && data.profile) {
            const profile = data.profile;
            document.getElementById('cv-file-name').textContent = profile.cvFile || '';
            document.getElementById('personal-description').value = profile.description || '';
            updateCharCount();
        }
    } catch (err) {
        console.error("Failed to load profile:", err);
    }
}

        // Ajoutez cette fonction après updateCharCount()
function updateJobCharCount() {
    const textarea = document.getElementById('job-file-name');
    const count = document.getElementById('job-char-count');
    count.textContent = `${textarea.value.length} / ${textarea.maxLength} caractères`;
}
        function updateCharCount() {
            const textarea = document.getElementById('personal-description');
            const count = document.getElementById('char-count');
            count.textContent = `${textarea.value.length} / ${textarea.maxLength} caractères`;
        }


// 🔹 Upload du CV
async function handleCVUpload(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch(`${API_BASE}/api/upload-cv/`, {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.status === "ok") {
                document.getElementById('cv-file-name').textContent = `✅ ${data.fileName}`;
                showToast('📄 CV uploadé avec succès', 'success');
            } else {
                showToast('⚠️ Erreur upload CV', 'error');
            }
        } catch (err) {
            console.error(err);
            showToast('❌ Erreur réseau upload CV', 'error');
        }
    }
}



        // Drag and Drop
function setupDragAndDrop() {
    const dropZones = document.querySelectorAll('.drop-zone');
    
    dropZones.forEach(zone => {
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('drag-over');
        });
        
        zone.addEventListener('dragleave', () => {
            zone.classList.remove('drag-over');
        });
        
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                if (zone.id === 'cv-drop-zone') {
                    document.getElementById('cv-file').files = files;
                    handleCVUpload({files: files});
                }
                // Supprimez la partie job-drop-zone
                else if (zone.id === 'photo-drop-zone') {
                    document.getElementById('photo-file').files = files;
                    handlePhotoUpload({files: files});
                }
            }
        });
    });
}





//------------------------------- CV Generation ------------------------------- //



        // CV Generation
async function generateCV() {
    console.log("=== [DEBUG] generateCV() triggered ===");

    const jobDescription = document.getElementById('job-file-name').value.trim();
    const jobLink = document.getElementById('job-link').value.trim();
    const primaryColor = document.getElementById('primary-color').value;
    const secondaryColor = document.getElementById('secondary-color').value;
    const language = document.getElementById('cv-language').value;
    const font = document.getElementById('cv-font').value;
    const photoFile = document.getElementById('photo-file').files[0];

    console.log("[DEBUG] jobDescription:", jobDescription);
    console.log("[DEBUG] jobLink:", jobLink);
    console.log("[DEBUG] primaryColor:", primaryColor, "secondaryColor:", secondaryColor);
    console.log("[DEBUG] language:", language, "font:", font);
    console.log("[DEBUG] photoFile:", photoFile ? photoFile.name : "no photo");

    if (!jobDescription && !jobLink) {
        showToast('⚠️ Veuillez fournir une description de poste ou un lien', 'error');
        return;
    }

    // --- Progress simulation
    const progressContainer = document.getElementById('progress-container');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');

    progressContainer.style.display = 'block';
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        progressFill.style.width = progress + '%';
        progressText.textContent = `Génération en cours... ${Math.round(progress)}%`;
    }, 500);

    // --- Étape 1 : upload photo si présente
    let hasPhoto = false;
    if (photoFile) {
        console.log("[DEBUG] Uploading photo...");
        const photoForm = new FormData();
        photoForm.append("file", photoFile);

        try {
            const uploadRes = await fetch(`${API_BASE}/api/upload-photo/`, {
                method: "POST",
                body: photoForm,
            });
            const uploadData = await uploadRes.json();
            console.log("[DEBUG] upload-photo response:", uploadData);
            hasPhoto = uploadData.status === "ok";
        } catch (err) {
            console.error("[ERROR] Photo upload failed:", err);
        }
    }

    // --- Étape 2 : sauvegarde job info
    const formData = new FormData();
    formData.append("jobDescription", jobDescription);
    formData.append("jobLink", jobLink);
    formData.append("primaryColor", primaryColor);
    formData.append("secondaryColor", secondaryColor);
    formData.append("language", language);
    formData.append("font", font);
    formData.append("hasPhoto", hasPhoto);

    try {
        console.log("[DEBUG] Sending job info...");
        const res = await fetch(`${API_BASE}/api/save-job-info/`, {
            method: "POST",
            body: formData,
        });
        const data = await res.json();
        console.log("[DEBUG] save-job-info response:", data);
    } catch (err) {
        console.error("[ERROR] Error saving job info:", err);
    }

    // --- Fin de la simulation
    setTimeout(() => {
        clearInterval(interval);
        progressFill.style.width = '100%';
        progressText.textContent = 'Génération terminée !';

        setTimeout(() => {
            progressContainer.style.display = 'none';
            progressFill.style.width = '0%';
            showToast('✅ CV généré et sauvegardé !', 'success');
            document.getElementById('job-file-name').value = '';
            document.getElementById('job-link').value = '';
            updateJobCharCount();
        }, 1000);
    }, 3000);
}




        // ------------------------------- Results Management ------------------------------- //


// ------------------------------- Load Generated CVs from Backend ------------------------------- //

async function loadResults() {
    const cvGrid = document.getElementById('cv-grid');
    const emptyState = document.getElementById('empty-state');

    try {
        const res = await fetch(`${API_BASE}/api/list-cvs/`);
        const data = await res.json();

        if (data.status !== "ok" || !data.cvs || data.cvs.length === 0) {
            cvGrid.innerHTML = '';
            emptyState.style.display = 'block';
            return;
        }

        emptyState.style.display = 'none';

        // Store locally (optional, reuse your previous logic)
        localStorage.setItem(STORAGE_KEYS.cvs, JSON.stringify(data.cvs));

        cvGrid.innerHTML = data.cvs.map(cv => {
            const formattedDate = new Date(cv.timestamp * 1000).toLocaleString();
            return `
                <div class="cv-card" data-title="${cv.title}" data-company="${cv.title}" data-timestamp="${cv.timestamp}">
                    <div class="cv-card-header">
                        <h3 class="cv-card-title">${cv.title}</h3>
                        <p class="cv-card-company">🧾 Fichier: ${cv.filename}</p>
                        <span class="cv-card-date">📅 ${formattedDate}</span>
                    </div>
                    <div style="display: flex; gap: 10px; margin-top: 15px;">
                        <a class="btn btn-primary" style="flex: 1; padding: 10px;" href="${API_BASE}${cv.url}" target="_blank" download>
                            📥 Télécharger
                        </a>
                        <button class="btn btn-danger" style="padding: 10px;" onclick="deleteStoredCV('${cv.filename}')">
                            🗑️
                        </button>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        console.error("❌ Failed to load CVs:", err);
        emptyState.style.display = 'block';
    }
}

// ------------------------------- Delete CV (optional) ------------------------------- //
async function deleteStoredCV(filename) {
    if (confirm('⚠️ Êtes-vous sûr de vouloir supprimer ce CV ?')) {
        try {
            const res = await fetch(`${API_BASE}/api/delete-cv/${filename}`, { method: "DELETE" });
            const data = await res.json();
            if (data.status === "ok") {
                showToast('🗑️ CV supprimé avec succès', 'success');
                loadResults();
            } else {
                showToast('❌ Erreur lors de la suppression du CV', 'error');
            }
        } catch (err) {
            console.error("Error deleting CV:", err);
            showToast('❌ Erreur serveur', 'error');
        }
    }
}






// Photo Upload Handler (ajoutez après handleJobUpload)
function handlePhotoUpload(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];
        const fileName = file.name;
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const photoName = document.getElementById('photo-file-name');
            photoName.innerHTML = `
                <div class="photo-preview">
                    <img src="${e.target.result}" alt="Photo de profil">
                    <div style="margin-top: 10px;">✅ ${fileName}</div>
                </div>
            `;
        };
        
        reader.readAsDataURL(file);
    }
}













        // Toast Notification
        function showToast(message, type = 'success') {
            const toast = document.getElementById('toast');
            const toastMessage = document.getElementById('toast-message');
            
            toastMessage.textContent = message;
            toast.className = `toast ${type} active`;
            
            setTimeout(() => {
                toast.classList.remove('active');
            }, 3000);
        }