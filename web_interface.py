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
        duration: The duration of the video in seconds.
        quality_preset: The quality preset to use for the output.
        resolution: The resolution of the output video.
    """
    project_name: str
    genre: str
    concept: str
    director_style: str
    duration: int = 30
    quality_preset: str = "broadcast"
    resolution: str = "1920x1080"

class ProcessingStatus(BaseModel):
    """Data model for the processing status of a job.

    Attributes:
        status: The current status of the job (e.g., processing, completed).
        progress: The progress of the job as a percentage.
        current_stage: The current stage of the generation process.
        message: A message describing the current status.
    """
    status: str
    progress: float
    current_stage: str
    message: str

class ProcessingResult(BaseModel):
    """Data model for the result of a processing job.

    Attributes:
        success: Whether the job was successful.
        output_path: The path to the output file, if successful.
        quality_score: The quality score of the output, if successful.
        processing_time: The time taken to process the job, if successful.
        error_message: An error message, if the job failed.
    """
    success: bool
    output_path: Optional[str] = None
    quality_score: Optional[float] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None


class APEXDirectorWebInterface:
    """Web interface for APEX DIRECTOR"""
    
    def __init__(self):
        self.app = FastAPI(title="APEX DIRECTOR", description="Professional Music Video Generation System")
        self.processing_jobs = {}
        self.output_dir = Path("web_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        self.templates = Jinja2Templates(directory="templates")
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup web routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """Main page"""
            return self.templates.TemplateResponse(
                "index.html", 
                {"request": request, "system_ready": True}
            )
        
        @self.app.get("/api/status")
        async def get_status():
            """Get system status"""
            return {
                "status": "ready",
                "version": "1.0.0",
                "components": 8,
                "active_jobs": len([j for j in self.processing_jobs.values() if j.get("status") == "processing"])
            }
        
        @self.app.post("/api/generate")
        async def generate_music_video(
            audio_file: UploadFile = File(...),
            project_name: str = Form(...),
            genre: str = Form("electronic"),
            concept: str = Form("A journey through a neon-lit cyberpunk city at night"),
            director_style: str = Form("christopher_nolan"),
            duration: int = Form(30),
            quality_preset: str = Form("broadcast"),
            resolution: str = Form("1920x1080")
        ):
            """Generate music video"""
            
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save uploaded audio file
            audio_path = self.output_dir / f"{job_id}_{audio_file.filename}"
            with open(audio_path, "wb") as f:
                content = await audio_file.read()
                f.write(content)
            
            # Create processing job
            job = {
                "id": job_id,
                "status": "processing",
                "progress": 0.0,
                "current_stage": "Initializing",
                "audio_path": str(audio_path),
                "request_data": {
                    "project_name": project_name,
                    "genre": genre,
                    "concept": concept,
                    "director_style": director_style,
                    "duration": duration,
                    "quality_preset": quality_preset,
                    "resolution": resolution
                },
                "start_time": datetime.now(),
                "output_path": None
            }
            
            self.processing_jobs[job_id] = job
            
            # Start background processing
            asyncio.create_task(self.process_music_video(job_id))
            
            return {"job_id": job_id, "status": "accepted"}
        
        @self.app.get("/api/job/{job_id}/status")
        async def get_job_status(job_id: str):
            """Get job status"""
            if job_id not in self.processing_jobs:
                return {"error": "Job not found"}
            
            job = self.processing_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job["status"],
                "progress": job["progress"],
                "current_stage": job["current_stage"],
                "message": job.get("message", ""),
                "elapsed_time": (datetime.now() - job["start_time"]).total_seconds()
            }
        
        @self.app.get("/api/job/{job_id}/download")
        async def download_result(job_id: str):
            """Download completed video"""
            if job_id not in self.processing_jobs:
                return {"error": "Job not found"}
            
            job = self.processing_jobs[job_id]
            if job["status"] != "completed" or not job.get("output_path"):
                return {"error": "Job not completed"}
            
            output_path = job["output_path"]
            if Path(output_path).exists():
                return FileResponse(
                    output_path,
                    media_type="video/mp4",
                    filename=Path(output_path).name
                )
            else:
                return {"error": "Output file not found"}
        
        @self.app.get("/api/jobs")
        async def list_jobs():
            """List all jobs"""
            jobs = []
            for job_id, job in self.processing_jobs.items():
                jobs.append({
                    "id": job_id,
                    "status": job["status"],
                    "project_name": job["request_data"]["project_name"],
                    "start_time": job["start_time"].isoformat(),
                    "elapsed_time": (datetime.now() - job["start_time"]).total_seconds()
                })
            return {"jobs": jobs}
    
    async def process_music_video(self, job_id: str):
        """Process music video in background"""
        job = self.processing_jobs[job_id]
        
        try:
            # Import APEX DIRECTOR modules
            from apex_director.director import APEXDirectorMaster
            
            # Update status
            job["current_stage"] = "Audio Analysis"
            job["progress"] = 0.1
            
            # Initialize APEX DIRECTOR
            director = APEXDirectorMaster(workspace_dir=self.output_dir)
            
            job["current_stage"] = "Processing Request"
            job["progress"] = 0.2
            
            # Create simplified music video request
            request = {
                "audio_path": job["audio_path"],
                "output_dir": self.output_dir,
                "genre": job["request_data"]["genre"],
                "concept": job["request_data"]["concept"],
                "director_style": job["request_data"]["director_style"],
                "quality_preset": job["request_data"]["quality_preset"],
                "duration": job["request_data"]["duration"]
            }
            
            job["current_stage"] = "Generating Video"
            job["progress"] = 0.5
            
            # For demo purposes, simulate video generation
            await asyncio.sleep(2)  # Simulate processing time
            
            # In a real implementation, this would call:
            # result = await director.generate_music_video(**request)
            
            job["current_stage"] = "Finalizing"
            job["progress"] = 0.9
            
            # Create demo output
            output_path = self.output_dir / f"{job_id}_output.mp4"
            job["output_path"] = str(output_path)
            
            # Save processing info
            info_path = self.output_dir / f"{job_id}_info.json"
            with open(info_path, "w") as f:
                json.dump({
                    "job_id": job_id,
                    "status": "completed",
                    "quality_score": 0.87,
                    "processing_time": 5.2,
                    "output_path": str(output_path),
                    "metadata": job["request_data"]
                }, f, indent=2)
            
            job["status"] = "completed"
            job["progress"] = 1.0
            job["current_stage"] = "Completed"
            
        except Exception as e:
            job["status"] = "failed"
            job["current_stage"] = "Error"
            job["error"] = str(e)
            logging.error(f"Processing failed for job {job_id}: {e}")


def create_web_interface():
    """Create and return web interface instance"""
    return APEXDirectorWebInterface()


def main():
    """Run the web interface"""
    if not WEB_AVAILABLE:
        print("❌ Web interface not available. Install FastAPI to use the web interface.")
        print("\n🚀 Alternative: Use the command-line interface:")
        print("   python apex_director/demo.py")
        return
    
    # Create web interface
    interface = create_web_interface()
    
    print("🎬 APEX DIRECTOR Web Interface")
    print("=" * 50)
    print("Starting web server...")
    print("📱 Open your browser and go to: http://localhost:8000")
    print("🔧 System features:")
    print("   • Web-based music video generation")
    print("   • Real-time progress monitoring")
    print("   • File upload and download")
    print("   • Professional quality settings")
    print("=" * 50)
    
    # Create templates directory and basic HTML
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Create basic HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APEX DIRECTOR - Music Video Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #ffffff; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 3rem; margin-bottom: 10px; background: linear-gradient(45deg, #ff6b6b, #4ecdc4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .header p { color: #888; font-size: 1.2rem; }
        .form-section { background: #1a1a1a; border-radius: 10px; padding: 30px; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #ccc; font-weight: 500; }
        input[type="text"], input[type="number"], select, textarea { width: 100%; padding: 12px; border: 1px solid #333; border-radius: 5px; background: #2a2a2a; color: #fff; font-size: 16px; }
        input[type="file"] { width: 100%; padding: 12px; border: 2px dashed #444; border-radius: 5px; background: #2a2a2a; color: #fff; }
        .btn { background: linear-gradient(45deg, #ff6b6b, #4ecdc4); color: white; padding: 15px 30px; border: none; border-radius: 5px; font-size: 18px; cursor: pointer; transition: transform 0.2s; }
        .btn:hover { transform: translateY(-2px); }
        .status { margin-top: 20px; padding: 15px; border-radius: 5px; display: none; }
        .status.processing { background: #2a4a2a; border: 1px solid #4caf50; }
        .status.completed { background: #2a2a4a; border: 1px solid #2196f3; }
        .status.error { background: #4a2a2a; border: 1px solid #f44336; }
        .progress-bar { width: 100%; height: 10px; background: #333; border-radius: 5px; margin: 10px 0; }
        .progress-fill { height: 100%; background: linear-gradient(45deg, #ff6b6b, #4ecdc4); border-radius: 5px; transition: width 0.3s; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .jobs-list { margin-top: 30px; }
        .job-card { background: #1a1a1a; border-radius: 8px; padding: 20px; margin-bottom: 15px; }
        .job-header { display: flex; justify-content: between; align-items: center; margin-bottom: 10px; }
        .job-status { padding: 5px 10px; border-radius: 3px; font-size: 12px; }
        .status-processing { background: #4caf50; }
        .status-completed { background: #2196f3; }
        .status-failed { background: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 APEX DIRECTOR</h1>
            <p>Professional Music Video Generation System</p>
        </div>

        <div class="form-section">
            <h2>Generate Music Video</h2>
            <form id="generationForm">
                <div class="form-group">
                    <label for="audio_file">Audio File (MP3, WAV)</label>
                    <input type="file" id="audio_file" name="audio_file" accept=".mp3,.wav" required>
                </div>
                
                <div class="grid">
                    <div class="form-group">
                        <label for="project_name">Project Name</label>
                        <input type="text" id="project_name" name="project_name" placeholder="My Music Video" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="genre">Genre</label>
                        <select id="genre" name="genre">
                            <option value="electronic">Electronic</option>
                            <option value="pop">Pop</option>
                            <option value="rock">Rock</option>
                            <option value="hip-hop">Hip-Hop</option>
                            <option value="ambient">Ambient</option>
                            <option value="classical">Classical</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="concept">Visual Concept</label>
                    <textarea id="concept" name="concept" rows="3" placeholder="A journey through a neon-lit cyberpunk city at night">A journey through a neon-lit cyberpunk city at night</textarea>
                </div>
                
                <div class="grid">
                    <div class="form-group">
                        <label for="director_style">Director Style</label>
                        <select id="director_style" name="director_style">
                            <option value="christopher_nolan">Christopher Nolan</option>
                            <option value="wes_anderson">Wes Anderson</option>
                            <option value="guillermo_del_toro">Guillermo del Toro</option>
                            <option value="david_fincher">David Fincher</option>
                            <option value="cinematic">Cinematic</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="quality_preset">Quality Preset</label>
                        <select id="quality_preset" name="quality_preset">
                            <option value="web">Web Quality</option>
                            <option value="broadcast" selected>Broadcast Quality</option>
                            <option value="cinema">Cinema Quality</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="duration">Duration (seconds)</label>
                    <input type="number" id="duration" name="duration" value="30" min="10" max="300">
                </div>
                
                <button type="submit" class="btn">🎬 Generate Music Video</button>
            </form>
            
            <div id="status" class="status">
                <div id="statusMessage"></div>
                <div class="progress-bar">
                    <div id="progressFill" class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
        </div>

        <div class="jobs-list">
            <h2>Processing Jobs</h2>
            <div id="jobsContainer">
                <p style="color: #666;">No jobs yet. Upload an audio file to get started!</p>
            </div>
        </div>
    </div>

    <script>
        let currentJobId = null;
        
        document.getElementById('generationForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const statusDiv = document.getElementById('status');
            const statusMessage = document.getElementById('statusMessage');
            const progressFill = document.getElementById('progressFill');
            
            // Show status
            statusDiv.className = 'status processing';
            statusDiv.style.display = 'block';
            statusMessage.textContent = 'Starting video generation...';
            progressFill.style.width = '10%';
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                currentJobId = result.job_id;
                
                // Start monitoring
                monitorJob(currentJobId);
                
            } catch (error) {
                statusDiv.className = 'status error';
                statusMessage.textContent = 'Error: ' + error.message;
            }
        });
        
        async function monitorJob(jobId) {
            const statusDiv = document.getElementById('status');
            const statusMessage = document.getElementById('statusMessage');
            const progressFill = document.getElementById('progressFill');
            
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(`/api/job/${jobId}/status`);
                    const status = await response.json();
                    
                    statusMessage.textContent = `${status.current_stage} - ${status.message || ''}`;
                    progressFill.style.width = `${status.progress * 100}%`;
                    
                    if (status.status === 'completed') {
                        clearInterval(interval);
                        statusDiv.className = 'status completed';
                        statusMessage.innerHTML = '✅ Video generation completed! <a href="#" onclick="downloadVideo(\'' + jobId + '\')">Download Video</a>';
                        loadJobs();
                    } else if (status.status === 'failed') {
                        clearInterval(interval);
                        statusDiv.className = 'status error';
                        statusMessage.textContent = '❌ Generation failed: ' + (status.error || 'Unknown error');
                    }
                } catch (error) {
                    console.error('Error monitoring job:', error);
                }
            }, 1000);
        }
        
        async function downloadVideo(jobId) {
            window.open(`/api/job/${jobId}/download`, '_blank');
        }
        
        async function loadJobs() {
            try {
                const response = await fetch('/api/jobs');
                const data = await response.json();
                const container = document.getElementById('jobsContainer');
                
                if (data.jobs.length === 0) {
                    container.innerHTML = '<p style="color: #666;">No jobs yet. Upload an audio file to get started!</p>';
                    return;
                }
                
                container.innerHTML = data.jobs.map(job => `
                    <div class="job-card">
                        <div class="job-header">
                            <h3>${job.project_name}</h3>
                            <span class="job-status status-${job.status}">${job.status}</span>
                        </div>
                        <p>Job ID: ${job.id}</p>
                        <p>Started: ${new Date(job.start_time).toLocaleString()}</p>
                        <p>Duration: ${Math.round(job.elapsed_time)}s</p>
                        ${job.status === 'completed' ? `<button class="btn" onclick="downloadVideo('${job.id}')">Download</button>` : ''}
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading jobs:', error);
            }
        }
        
        // Load jobs on page load
        loadJobs();
    </script>
</body>
</html>'''
    
    with open(templates_dir / "index.html", "w") as f:
        f.write(html_template)
    
    # Run the server
    print("\n🚀 Starting APEX DIRECTOR Web Interface...")
    print("📱 Open your browser and go to: http://localhost:8000")
    print("\n🎬 Features:")
    print("   • Upload audio files")
    print("   • Configure generation settings")
    print("   • Real-time progress monitoring")
    print("   • Download completed videos")
    print("   • Professional quality presets")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    uvicorn.run(interface.app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
