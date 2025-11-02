#!/usr/bin/env python3
"""
APEX DIRECTOR Web Interface

Simple web-based GUI for the APEX DIRECTOR music video generation system.
This provides an easy way to launch and use the system through a web browser.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

# FastAPI and web components
try:
    from fastapi import FastAPI, UploadFile, File, Form, Request
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from pydantic import BaseModel
    import uvicorn
    WEB_AVAILABLE = True
except ImportError:
    print("⚠️  Web interface requires FastAPI. Installing...")
    os.system("pip install fastapi uvicorn python-multipart jinja2")
    try:
        from fastapi import FastAPI, UploadFile, File, Form, Request
        from fastapi.responses import HTMLResponse, FileResponse
        from fastapi.staticfiles import StaticFiles
        from fastapi.templating import Jinja2Templates
        from pydantic import BaseModel
        import uvicorn
        WEB_AVAILABLE = True
    except ImportError:
        print("❌ Could not install web dependencies. Falling back to command-line interface.")
        WEB_AVAILABLE = False

# Add apex_director to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)

# Data models for the web interface
class MusicVideoRequestModel(BaseModel):
    """Data model for a music video generation request.

    Attributes:
        project_name: The name of the project.
        genre: The genre of the music.
        concept: The creative concept for the video.
        director_style: The director style to emulate.
        quality_preset: The quality preset to use for the output.
    """
    project_name: str
    genre: str
    concept: str
    director_style: str
    quality_preset: str

# Store for demo purposes
active_projects = {}

# Create FastAPI app
app = FastAPI(title="APEX DIRECTOR", description="Music Video Generation System")

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APEX DIRECTOR - Music Video Generation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        }

        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .form-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #4ecdc4;
        }

        input[type="text"], select, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        input[type="text"]:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #4ecdc4;
            box-shadow: 0 0 15px rgba(78, 205, 196, 0.3);
        }

        textarea {
            height: 100px;
            resize: vertical;
        }

        input[type="file"] {
            padding: 10px;
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.05);
            color: white;
            width: 100%;
        }

        .submit-btn {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 50px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 20px;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }

        .progress-section {
            display: none;
            margin-top: 30px;
        }

        .progress-bar {
            width: 100%;
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 15px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            width: 0%;
            transition: width 0.5s ease;
            border-radius: 10px;
        }

        .project-status {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .status-item:last-child {
            border-bottom: none;
        }

        .status-label {
            font-weight: 600;
            color: #4ecdc4;
        }

        .status-value {
            color: white;
        }

        .alert {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            background: rgba(78, 205, 196, 0.2);
            border: 1px solid rgba(78, 205, 196, 0.5);
            color: #4ecdc4;
        }

        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }

        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid #4ecdc4;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .output-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 30px;
            margin-top: 30px;
            display: none;
        }

        .video-preview {
            max-width: 100%;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .download-btn {
            background: #ff6b6b;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            font-weight: bold;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s ease;
        }

        .download-btn:hover {
            background: #ff5252;
            transform: translateY(-1px);
        }

        .active-projects {
            margin-top: 30px;
        }

        .project-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .project-title {
            font-size: 18px;
            font-weight: bold;
            color: #4ecdc4;
            margin-bottom: 10px;
        }

        .project-meta {
            font-size: 14px;
            color: rgba(255, 255, 255, 0.7);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 APEX DIRECTOR</h1>
        <p style="text-align: center; margin-bottom: 40px; font-size: 18px;">
            Professional Music Video Generation System
        </p>

        <!-- Main Form -->
        <div id="main-form" class="form-section">
            <h2 style="margin-bottom: 25px; color: #4ecdc4;">🎵 Create New Music Video Project</h2>
            
            <form id="project-form" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="project_name">Project Name</label>
                    <input type="text" id="project_name" name="project_name" required placeholder="Enter your project name">
                </div>

                <div class="form-group">
                    <label for="audio_file">Audio File</label>
                    <input type="file" id="audio_file" name="audio_file" accept=".mp3,.wav,.m4a,.flac" required>
                    <small style="color: rgba(255,255,255,0.7);">Supported formats: MP3, WAV, M4A, FLAC</small>
                </div>

                <div class="form-group">
                    <label for="genre">Genre</label>
                    <select id="genre" name="genre" required>
                        <option value="">Select Genre</option>
                        <option value="electronic">Electronic</option>
                        <option value="rock">Rock</option>
                        <option value="pop">Pop</option>
                        <option value="hip-hop">Hip-Hop</option>
                        <option value="jazz">Jazz</option>
                        <option value="classical">Classical</option>
                        <option value="indie">Indie</option>
                        <option value="r&b">R&B</option>
                        <option value="country">Country</option>
                        <option value="reggae">Reggae</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="concept">Creative Concept</label>
                    <textarea id="concept" name="concept" required placeholder="Describe your vision for the music video..."></textarea>
                </div>

                <div class="form-group">
                    <label for="director_style">Director Style</label>
                    <select id="director_style" name="director_style" required>
                        <option value="">Select Director Style</option>
                        <option value="christopher_nolan">Christopher Nolan</option>
                        <option value="quentin_tarantino">Quentin Tarantino</option>
                        <option value="denis_villeneuve">Denis Villeneuve</option>
                        <option value="spike_jonze">Spike Jonze</option>
                        <option value="michel_gondry">Michel Gondry</option>
                        <option value="david_fincher">David Fincher</option>
                        <option value="wes_anderson">Wes Anderson</option>
                        <option value="ridley_scott">Ridley Scott</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="quality_preset">Quality Preset</label>
                    <select id="quality_preset" name="quality_preset" required>
                        <option value="">Select Quality</option>
                        <option value="draft">Draft (Fast)</option>
                        <option value="web">Web (Good)</option>
                        <option value="broadcast" selected>Broadcast (Professional)</option>
                        <option value="cinema">Cinema (Highest)</option>
                    </select>
                </div>

                <button type="submit" class="submit-btn">🚀 Generate Music Video</button>
            </form>
        </div>

        <!-- Loading Section -->
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p>🎬 Generating your music video...</p>
        </div>

        <!-- Progress Section -->
        <div id="progress-section" class="progress-section">
            <h3 style="margin-bottom: 20px; color: #4ecdc4;">📊 Generation Progress</h3>
            <div class="progress-bar">
                <div id="progress-fill" class="progress-fill"></div>
            </div>
            <div id="progress-text" style="text-align: center; margin-bottom: 20px;">Preparing...</div>
            
            <div class="project-status">
                <h4 style="margin-bottom: 15px; color: #4ecdc4;">Current Status</h4>
                <div class="status-item">
                    <span class="status-label">Stage:</span>
                    <span id="current-stage" class="status-value">Initializing</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Progress:</span>
                    <span id="progress-percentage" class="status-value">0%</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Time Elapsed:</span>
                    <span id="time-elapsed" class="status-value">0s</span>
                </div>
                <div class="status-item">
                    <span class="status-label">ETA:</span>
                    <span id="eta" class="status-value">Calculating...</span>
                </div>
            </div>
        </div>

        <!-- Output Section -->
        <div id="output-section" class="output-section">
            <h3 style="margin-bottom: 20px; color: #4ecdc4;">🎉 Video Generated Successfully!</h3>
            <div class="alert">
                Your music video has been generated and is ready for download.
            </div>
            <div id="video-details"></div>
            <a id="download-btn" class="download-btn" href="#" download>📥 Download Video</a>
        </div>

        <!-- Active Projects -->
        <div id="active-projects" class="active-projects">
            <h3 style="margin-bottom: 20px; color: #4ecdc4;">📋 Active Projects</h3>
            <div id="projects-list">
                <p style="color: rgba(255,255,255,0.7);">No active projects.</p>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let currentProjectId = null;
        let progressInterval = null;
        let startTime = null;

        // Form submission handler
        document.getElementById('project-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const submitBtn = document.querySelector('.submit-btn');
            const loading = document.getElementById('loading');
            const mainForm = document.getElementById('main-form');
            
            // Show loading state
            submitBtn.style.display = 'none';
            loading.style.display = 'block';
            
            try {
                const response = await fetch('/api/project/create', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    currentProjectId = result.project_id;
                    startTime = Date.now();
                    showProgress();
                    pollProgress();
                } else {
                    alert('Error creating project: ' + result.error);
                    resetUI();
                }
            } catch (error) {
                alert('Error: ' + error.message);
                resetUI();
            }
        });

        function showProgress() {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('progress-section').style.display = 'block';
        }

        function resetUI() {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('progress-section').style.display = 'none';
            document.querySelector('.submit-btn').style.display = 'block';
        }

        async function pollProgress() {
            if (!currentProjectId) return;
            
            try {
                const response = await fetch(`/api/project/${currentProjectId}/status`);
                const data = await response.json();
                
                if (data.success) {
                    updateProgress(data);
                    
                    if (data.status === 'completed') {
                        showOutput(data.result);
                        clearInterval(progressInterval);
                    } else if (data.status === 'failed') {
                        alert('Generation failed: ' + data.error);
                        resetUI();
                        clearInterval(progressInterval);
                    } else {
                        // Continue polling
                        setTimeout(pollProgress, 2000);
                    }
                }
            } catch (error) {
                console.error('Error checking progress:', error);
                setTimeout(pollProgress, 2000);
            }
        }

        function updateProgress(data) {
            document.getElementById('progress-fill').style.width = data.progress + '%';
            document.getElementById('progress-percentage').textContent = data.progress + '%';
            document.getElementById('current-stage').textContent = data.current_stage || 'Processing...';
            document.getElementById('progress-text').textContent = data.status_message || 'Processing...';
            
            if (startTime) {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                document.getElementById('time-elapsed').textContent = elapsed + 's';
            }
        }

        function showOutput(data) {
            document.getElementById('progress-section').style.display = 'none';
            document.getElementById('output-section').style.display = 'block';
            
            document.getElementById('download-btn').href = data.output_path;
            document.getElementById('video-details').innerHTML = `
                <div class="status-item">
                    <span class="status-label">Quality Score:</span>
                    <span class="status-value">${data.quality_score}/100</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Processing Time:</span>
                    <span class="status-value">${data.processing_time}s</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Resolution:</span>
                    <span class="status-value">${data.resolution}</span>
                </div>
            `;
        }

        // Load active projects on page load
        window.addEventListener('load', function() {
            loadActiveProjects();
        });

        async function loadActiveProjects() {
            try {
                const response = await fetch('/api/projects/active');
                const data = await response.json();
                
                if (data.success && data.projects.length > 0) {
                    const projectsList = document.getElementById('projects-list');
                    projectsList.innerHTML = data.projects.map(project => `
                        <div class="project-card">
                            <div class="project-title">${project.name}</div>
                            <div class="project-meta">
                                Genre: ${project.genre} | 
                                Quality: ${project.quality} | 
                                Status: ${project.status}
                            </div>
                        </div>
                    `).join('');
                }
            } catch (error) {
                console.error('Error loading projects:', error);
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    """Main page with the music video generation form"""
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)

@app.post("/api/project/create")
async def create_project(
    project_name: str = Form(...),
    genre: str = Form(...),
    concept: str = Form(...),
    director_style: str = Form(...),
    quality_preset: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """Creates a new music video generation project.

    Args:
        project_name: The name of the project.
        genre: The genre of the music.
        concept: The creative concept for the video.
        director_style: The director style to emulate.
        quality_preset: The quality preset for the output.
        audio_file: The uploaded audio file.

    Returns:
        A dictionary with the project ID and a success message, or an error message.
    """
    
    try:
        # Generate project ID
        project_id = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save uploaded audio file
        audio_path = f"uploads/{project_id}_{audio_file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        with open(audio_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Store project info
        active_projects[project_id] = {
            "id": project_id,
            "name": project_name,
            "genre": genre,
            "concept": concept,
            "director_style": director_style,
            "quality_preset": quality_preset,
            "audio_path": audio_path,
            "status": "created",
            "progress": 0,
            "current_stage": "Project created",
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "project_id": project_id,
            "message": "Project created successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/project/{project_id}/status")
async def get_project_status(project_id: str):
    """Get the status of a specific project"""
    
    if project_id not in active_projects:
        return {
            "success": False,
            "error": "Project not found"
        }
    
    project = active_projects[project_id]
    
    # Simulate progress for demo purposes
    if project["status"] == "created":
        project["status"] = "processing"
        project["progress"] = 10
        project["current_stage"] = "Audio analysis"
    elif project["progress"] < 50:
        project["progress"] += 10
        if project["progress"] == 20:
            project["current_stage"] = "Beat detection"
        elif project["progress"] == 30:
            project["current_stage"] = "Cinematography planning"
        elif project["progress"] == 40:
            project["current_stage"] = "Image generation"
        elif project["progress"] >= 50:
            project["current_stage"] = "Video assembly"
    elif project["progress"] < 90:
        project["progress"] += 15
        if project["progress"] >= 90:
            project["current_stage"] = "Final rendering"
    elif project["progress"] < 100:
        project["progress"] = 100
        project["status"] = "completed"
        project["current_stage"] = "Complete"
        # Simulate completion
        project["result"] = {
            "output_path": f"/api/project/{project_id}/download",
            "quality_score": 92,
            "processing_time": 45,
            "resolution": "1920x1080 (Broadcast)"
        }
    
    return {
        "success": True,
        "status": project["status"],
        "progress": project["progress"],
        "current_stage": project["current_stage"],
        "status_message": f"Processing {project['current_stage']}...",
        **project.get("result", {})
    }

@app.get("/api/project/{project_id}/download")
async def download_video(project_id: str):
    """Download the generated video"""
    
    if project_id not in active_projects:
        return {"error": "Project not found"}
    
    # For demo purposes, return a placeholder
    return {
        "message": "Video would be downloaded here in production",
        "project_id": project_id
    }

@app.get("/api/projects/active")
async def get_active_projects():
    """Get all active projects"""
    
    projects = []
    for project in active_projects.values():
        projects.append({
            "id": project["id"],
            "name": project["name"],
            "genre": project["genre"],
            "quality": project["quality_preset"],
            "status": project["status"],
            "progress": project["progress"]
        })
    
    return {
        "success": True,
        "projects": projects
    }

def main():
    """Main function to run the web server"""
    
    if not WEB_AVAILABLE:
        print("❌ Web interface not available. Please install required dependencies.")
        return
    
    print("🎬 APEX DIRECTOR Web Interface Starting...")
    print("🌐 Open your browser to: http://localhost:8000")
    print("🎯 Ready to generate professional music videos!")
    print("=" * 50)
    
    # Create uploads directory
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
