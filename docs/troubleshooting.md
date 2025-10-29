# APEX DIRECTOR Troubleshooting Guide

Comprehensive troubleshooting guide for resolving common issues and optimizing performance.

## Table of Contents

- [Getting Help](#getting-help)
- [Common Issues](#common-issues)
- [System Diagnostics](#system-diagnostics)
- [Performance Issues](#performance-issues)
- [Quality Problems](#quality-problems)
- [Configuration Issues](#configuration-issues)
- [Network and Backend Problems](#network-and-backend-problems)
- [Recovery Procedures](#recovery-procedures)
- [Prevention Tips](#prevention-tips)

---

## Getting Help

### Self-Help Resources

Before contacting support, try these self-help options:

1. **Check the logs**: Most issues are documented in log files
2. **Run diagnostics**: Use built-in diagnostic tools
3. **Review documentation**: Check relevant guides and API reference
4. **Search community**: Look for similar issues in forums or GitHub

### Diagnostic Commands

```bash
# Check system status
apex_director system status

# Run comprehensive diagnostics
apex_director diagnose --full

# Check configuration
apex_director config validate

# Test backend connectivity
apex_director backend test all

# Monitor system resources
apex_director monitor --real-time
```

### Support Channels

#### Community Support
- **GitHub Issues**: Bug reports and feature requests
- **Discussion Forum**: Community Q&A and discussions
- **Discord Server**: Real-time help and chat
- **Documentation**: Comprehensive guides and references

#### Professional Support
- **Email Support**: support@apex-director.com
- **Priority Support**: Available for enterprise customers
- **Phone Support**: For critical production issues
- **Training Services**: Custom training and consulting

### When to Contact Support

Contact support when you experience:
- System crashes or freezes
- Data corruption or loss
- Security incidents
- Performance degradation after configuration changes
- Issues not resolved by troubleshooting steps

---

## Common Issues

### Installation and Setup

#### Issue: "Python version not supported"

**Symptoms:**
```
ERROR: Python 3.8 or higher is required
Current version: 3.7.16
```

**Solutions:**
1. **Upgrade Python**:
   ```bash
   # Install Python 3.10 (Ubuntu/Debian)
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3.10-dev
   
   # Install Python 3.10 (CentOS/RHEL)
   sudo yum install python310 python3-devel
   ```

2. **Use pyenv**:
   ```bash
   # Install pyenv
   curl https://pyenv.run | bash
   
   # Install Python 3.10
   pyenv install 3.10.12
   pyenv global 3.10.12
   ```

#### Issue: "Permission denied during installation"

**Symptoms:**
```
ERROR: Permission denied when writing to /usr/local/lib/python3.10/site-packages/
```

**Solutions:**
1. **Use virtual environment** (recommended):
   ```bash
   python3 -m venv apex_env
   source apex_env/bin/activate
   pip install apex-director
   ```

2. **Fix permissions** (if using system installation):
   ```bash
   sudo chown -R $USER:$USER ~/.local/lib/python3.10/site-packages/
   pip install --user apex-director
   ```

#### Issue: "Missing system dependencies"

**Symptoms:**
```
ERROR: Microsoft Visual C++ 14.0 is required
ERROR: pkg-config not found
```

**Solutions:**

1. **Install Visual C++ Build Tools** (Windows):
   - Download and install Microsoft C++ Build Tools
   - Or install Visual Studio Community

2. **Install system packages** (Linux):
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install build-essential pkg-config libjpeg-dev libpng-dev
   
   # CentOS/RHEL
   sudo yum groupinstall "Development Tools"
   sudo yum install libjpeg-devel libpng-devel
   ```

### Backend Connectivity

#### Issue: "Backend API key invalid"

**Symptoms:**
```
ERROR: Authentication failed for backend: imagen
Status: 401 Unauthorized
Message: Invalid API key
```

**Solutions:**
1. **Verify API key**:
   ```python
   from apex_director.core.backend_manager import BackendManager
   
   manager = BackendManager()
   health = await manager.check_backend_health("imagen")
   print(health)
   ```

2. **Update API key**:
   ```bash
   # Set environment variable
   export GOOGLE_API_KEY="your-actual-api-key"
   
   # Or update config file
   apex_director config set backends.imagen.api_key "your-actual-api-key"
   ```

3. **Check API key permissions**:
   - Verify the API key has access to the required services
   - Ensure billing is enabled for your Google Cloud project

#### Issue: "Backend service unavailable"

**Symptoms:**
```
ERROR: Backend service timeout: imagen
Connection failed after 30 seconds
```

**Solutions:**
1. **Check network connectivity**:
   ```bash
   # Test basic connectivity
   ping googleapis.com
   
   # Test HTTPS connectivity
   curl -I https://vision.googleapis.com/v1/images:annotate
   ```

2. **Check backend status**:
   ```bash
   # Check backend status
   apex_director backend status
   
   # Test specific backend
   apex_director backend test nano_banana
   ```

3. **Verify configuration**:
   ```json
   {
     "backends": {
       "imagen": {
         "enabled": true,
         "api_endpoint": "https://vision.googleapis.com/v1",
         "timeout_seconds": 60
       }
     }
   }
   ```

### Generation Issues

#### Issue: "No images generated"

**Symptoms:**
```
WARNING: Generation completed with 0 images
All backends failed or returned empty results
```

**Solutions:**
1. **Check prompt validity**:
   ```python
   # Use simple, clear prompts
   prompt = "A red rose in a garden"
   
   # Avoid problematic content
   # - Very long prompts (>500 characters)
   # - Inappropriate content
   # - Contradictory descriptions
   ```

2. **Verify backend availability**:
   ```bash
   # Test all backends
   apex_director backend test all --verbose
   ```

3. **Check system resources**:
   ```bash
   # Check disk space
   df -h
   
   # Check memory usage
   free -h
   
   # Check GPU availability (if using)
   nvidia-smi
   ```

#### Issue: "Poor image quality"

**Symptoms:**
```
Quality Score: 0.45 (Poor)
Issues: Blurry, low resolution, artifacts
```

**Solutions:**
1. **Use higher quality preset**:
   ```python
   await submit_music_video_job(
       quality_preset="broadcast",  # Instead of "web"
       enable_upscaling=True,
       steps=75  # More generation steps
   )
   ```

2. **Improve prompts**:
   ```python
   # Instead of: "A person"
   # Use: "Close-up portrait of a professional headshot, 
   #       studio lighting, shallow depth of field, 
   #       85mm lens, f/2.8, high resolution"
   ```

3. **Enable upscaling**:
   ```python
   await submit_music_video_job(
       enable_upscaling=True,
       upscale_preset="broadcast_quality"
   )
   ```

---

## System Diagnostics

### Built-in Diagnostics

```python
from apex_director import run_diagnostics

# Run comprehensive diagnostics
diagnostics = run_diagnostics()

print("System Information:")
print(f"OS: {diagnostics['system']['os']}")
print(f"Python: {diagnostics['system']['python_version']}")
print(f"Architecture: {diagnostics['system']['architecture']}")

print("\nDependencies:")
for dep, version in diagnostics['dependencies'].items():
    status = "✓" if version else "✗"
    print(f"{status} {dep}: {version}")

print("\nConfiguration:")
for config, status in diagnostics['configuration'].items():
    print(f"{status} {config}")

print("\nBackends:")
for backend, status in diagnostics['backends'].items():
    print(f"{status} {backend}")
```

### Health Check Script

```bash
#!/bin/bash
# health_check.sh

echo "=== APEX DIRECTOR Health Check ==="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python: $python_version"

# Check disk space
disk_usage=$(df -h / | awk 'NR==2 {print $5}')
echo "Disk Usage: $disk_usage"

# Check memory
memory_info=$(free -h | awk 'NR==2{printf "Used: %s/%s (%.1f%%)\n", $3,$2,$3*100/$2}')
echo "Memory: $memory_info"

# Check logs for errors
error_count=$(tail -1000 /var/log/apex_director/app.log | grep -c "ERROR")
echo "Recent Errors: $error_count"

# Check backend connectivity
echo "Testing backends..."
for backend in nano_banana imagen minimax sdxl; do
    if apex_director backend test $backend --quiet; then
        echo "✓ $backend: OK"
    else
        echo "✗ $backend: Failed"
    fi
done

# Check configuration
if apex_director config validate --quiet; then
    echo "✓ Configuration: Valid"
else
    echo "✗ Configuration: Invalid"
fi

echo "=== Health Check Complete ==="
```

### Log Analysis

#### Common Log Patterns

```bash
# Find errors in last 24 hours
grep "ERROR" /var/log/apex_director/app.log | grep "$(date -d '24 hours ago' +'%Y-%m-%d')"

# Find failed jobs
grep "Job failed" /var/log/apex_director/app.log | tail -10

# Find backend errors
grep "Backend error" /var/log/apex_director/app.log | tail -20

# Performance issues
grep "Performance warning" /var/log/apex_director/app.log | tail -10

# Memory issues
grep -i "memory\|out of\|OOM" /var/log/apex_director/app.log | tail -5
```

#### Log Analysis Script

```python
#!/usr/bin/env python3
import re
from datetime import datetime, timedelta
from collections import defaultdict

def analyze_logs(log_file_path):
    """Analyze log file for patterns and issues."""
    
    issues = defaultdict(list)
    stats = {
        'total_lines': 0,
        'errors': 0,
        'warnings': 0,
        'jobs_completed': 0,
        'jobs_failed': 0,
        'backends_used': defaultdict(int)
    }
    
    # Define patterns
    error_pattern = re.compile(r'ERROR.*')
    warning_pattern = re.compile(r'WARNING.*')
    job_completed_pattern = re.compile(r'Job completed.*job_id=(\w+)')
    job_failed_pattern = re.compile(r'Job failed.*job_id=(\w+)')
    backend_pattern = re.compile(r'Backend.*(\w+).*')
    performance_pattern = re.compile(r'Performance.*(\w+\.?\w*)\s*=\s*(\d+\.?\d*)')
    
    with open(log_file_path, 'r') as f:
        for line in f:
            stats['total_lines'] += 1
            
            if error_pattern.search(line):
                stats['errors'] += 1
                issues['errors'].append(line.strip())
                
            elif warning_pattern.search(line):
                stats['warnings'] += 1
                issues['warnings'].append(line.strip())
                
            elif job_completed_pattern.search(line):
                stats['jobs_completed'] += 1
                
            elif job_failed_pattern.search(line):
                stats['jobs_failed'] += 1
                
            elif backend_pattern.search(line):
                backend = backend_pattern.search(line).group(1)
                stats['backends_used'][backend] += 1
    
    return stats, issues

# Run analysis
stats, issues = analyze_logs('/var/log/apex_director/app.log')

print("=== Log Analysis Report ===")
print(f"Total log lines: {stats['total_lines']}")
print(f"Errors: {stats['errors']}")
print(f"Warnings: {stats['warnings']}")
print(f"Jobs completed: {stats['jobs_completed']}")
print(f"Jobs failed: {stats['jobs_failed']}")

print("\nBackend Usage:")
for backend, count in stats['backends_used'].items():
    print(f"  {backend}: {count} times")

print("\nRecent Errors:")
for error in issues['errors'][-5:]:
    print(f"  {error}")
```

---

## Performance Issues

### Slow Generation

#### Symptoms
- Generation taking much longer than expected
- Timeouts occurring frequently
- System appears unresponsive

#### Diagnosis Steps

1. **Check system resources**:
   ```bash
   # CPU usage
   top -p $(pgrep -f apex_director)
   
   # Memory usage
   ps aux | grep apex_director
   
   # Disk I/O
   iostat -x 1 5
   ```

2. **Monitor generation pipeline**:
   ```python
   from apex_director import get_orchestrator
   
   orchestrator = get_orchestrator()
   stats = orchestrator.get_system_stats()
   
   print(f"Active jobs: {stats['jobs']['active_jobs']}")
   print(f"Queue size: {stats['jobs']['queued_jobs']}")
   print(f"Average processing time: {stats['performance']['average_processing_time']:.2f}s")
   ```

3. **Check backend performance**:
   ```bash
   # Test individual backends
   time apex_director backend test imagen --generate-test
   ```

#### Solutions

1. **Optimize concurrency**:
   ```json
   {
     "orchestrator": {
       "max_concurrent_jobs": 2,  // Reduce from 5
       "job_timeout_seconds": 1800  // Increase timeout
     }
   }
   ```

2. **Use faster presets**:
   ```python
   await submit_music_video_job(
       quality_preset="web",  # Instead of "broadcast"
       max_shots=10  // Reduce from 20
   )
   ```

3. **Enable caching**:
   ```json
   {
     "caching": {
       "image_cache": {
         "enabled": true,
         "max_size_mb": 2048
       }
     }
   }
   ```

### High Memory Usage

#### Symptoms
- System running out of memory
- OOM (Out of Memory) errors
- System becoming unresponsive

#### Diagnosis

```python
import psutil
import os

def check_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"Virtual memory: {memory_info.vms / 1024 / 1024:.2f} MB")
    
    # Check system memory
    system_memory = psutil.virtual_memory()
    print(f"System memory: {system_memory.percent}% used")
    
    # Check for memory leaks
    if memory_info.rss > 2 * 1024 * 1024 * 1024:  # 2GB
        print("WARNING: High memory usage detected")

check_memory_usage()
```

#### Solutions

1. **Reduce batch sizes**:
   ```python
   # Process in smaller batches
   batch_size = 5  # Instead of 20
   for i in range(0, total_items, batch_size):
       batch = items[i:i+batch_size]
       await process_batch(batch)
   ```

2. **Enable memory monitoring**:
   ```json
   {
     "performance": {
       "memory_limit_mb": 4096,
       "memory_monitoring": true,
       "gc_threshold": 700
     }
   }
   ```

3. **Use streaming for large files**:
   ```python
   # Stream processing instead of loading everything
   async def process_large_video(video_path):
       async with aiofiles.open(video_path, 'rb') as f:
           while True:
               chunk = await f.read(1024 * 1024)  # 1MB chunks
               if not chunk:
                   break
               await process_chunk(chunk)
   ```

### Network Latency

#### Symptoms
- Backend requests timing out
- Slow response times
- Intermittent connectivity issues

#### Diagnosis

```bash
# Test network latency to backend services
ping googleapis.com
ping api.nanobanana.com

# Test HTTPS connectivity
curl -w "@curl-format.txt" -o /dev/null -s https://vision.googleapis.com/v1

# Check DNS resolution
nslookup vision.googleapis.com
```

#### Solutions

1. **Increase timeouts**:
   ```json
   {
     "backends": {
       "imagen": {
         "timeout_seconds": 120,  // Increase from 60
         "retry_attempts": 5
       }
     }
   }
   ```

2. **Use closer backend**:
   ```json
   {
     "backends": {
       "imagen": {
         "region": "us-west2",  // Use region closer to your location
         "endpoint": "https://us-west2-aiplatform.googleapis.com"
       }
     }
   }
   ```

3. **Enable connection pooling**:
   ```json
   {
     "performance": {
       "network": {
         "connection_pool_size": 20,
         "keep_alive_seconds": 60,
         "retry_attempts": 3
       }
     }
   }
   ```

---

## Quality Problems

### Inconsistent Quality

#### Symptoms
- Quality varies significantly between images
- Some images fail quality checks
- Style consistency issues

#### Solutions

1. **Improve style bible**:
   ```json
   {
     "style_bible": {
       "color_palette": {
         "primary_colors": ["#FF6B35", "#004E89", "#1A936F"],
         "tolerance": 0.1  // Reduce color variation
       },
       "lighting_consistency": {
         "enabled": true,
         "max_deviation": 0.05
       }
     }
   }
   ```

2. **Use higher quality thresholds**:
   ```python
   await submit_music_video_job(
       quality_threshold=0.85,  // Increase from 0.7
       auto_regenerate_low_quality=True
   )
   ```

3. **Enable style monitoring**:
   ```json
   {
     "quality_assurance": {
       "style_monitoring": {
         "enabled": true,
         "drift_threshold": 0.1,
         "auto_correction": true
       }
     }
   }
   ```

### Character Inconsistency

#### Symptoms
- Character appearance changes between scenes
- Facial features don't match
- Clothing or accessories vary

#### Solutions

1. **Improve reference images**:
   ```python
   # Use more reference images
   character_manager.create_character_profile(
       name="character_name",
       reference_images=[
           "refs/front_view.jpg",
           "refs/profile_left.jpg",
           "refs/profile_right.jpg",
           "refs/close_up.jpg",
           "refs/full_body.jpg"
       ],
       description="Detailed character description"
   )
   ```

2. **Increase consistency threshold**:
   ```python
   await submit_music_video_job(
       character_consistency_threshold=0.9,  // Increase from 0.85
       enable_detailed_validation=True
   )
   ```

3. **Validate generated images**:
   ```python
   # Validate character consistency after generation
   for image_path in generated_images:
       is_consistent, confidence = await character_manager.validate_consistency(
           image_path, character_id
       )
       if not is_consistent:
           print(f"Character inconsistency detected: {confidence}")
           # Regenerate or adjust
   ```

### Audio Sync Issues

#### Symptoms
- Visual cuts don't align with beats
- Sections don't match musical structure
- Timing appears off

#### Solutions

1. **Improve audio analysis**:
   ```python
   # Analyze audio with higher precision
   audio_analysis = await audio_analyzer.analyze_audio(
       audio_path,
       sensitivity=0.8,  // Increase from default 0.5
       frame_accuracy=True
   )
   ```

2. **Manual beat correction**:
   ```python
   # Override automatic beat detection
   manual_beats = [0.5, 1.2, 1.8, 2.4]  # Manually defined beat times
   
   await submit_music_video_job(
       audio_path=audio_path,
       manual_beat_markers=manual_beats,
       frame_accurate_editing=True
   )
   ```

3. **Adjust cut tolerance**:
   ```json
   {
     "audio_sync": {
       "frame_accuracy": true,
       "max_timing_error_frames": 0.5,  // Allow 0.5 frame tolerance
       "beat_lock_cutting": true
     }
   }
   ```

---

## Configuration Issues

### Invalid Configuration

#### Symptoms
- Configuration validation errors
- System failing to start
- Default settings being used instead

#### Solutions

1. **Validate configuration**:
   ```bash
   apex_director config validate --verbose
   
   # Check specific section
   apex_director config validate --section backends
   ```

2. **Reset to defaults**:
   ```bash
   # Backup current config
   cp ~/.config/apex_director/config.json ~/.config/apex_director/config.json.backup
   
   # Reset to default
   apex_director config reset --confirm
   ```

3. **Fix specific issues**:
   ```bash
   # Common fixes
   apex_director config set system.log_level "INFO"
   apex_director config set orchestrator.max_concurrent_jobs 5
   apex_director config set backends.nano_banana.enabled true
   ```

### Permission Problems

#### Symptoms
- "Permission denied" errors
- Cannot write to configuration files
- Log files not being created

#### Solutions

1. **Fix file permissions**:
   ```bash
   # Fix configuration directory
   chmod 755 ~/.config/apex_director
   chmod 644 ~/.config/apex_director/*.json
   
   # Fix log directory
   sudo chown -R $USER:$USER /var/log/apex_director
   chmod 755 /var/log/apex_director
   
   # Fix data directory
   sudo chown -R $USER:$USER /var/lib/apex_director
   chmod 755 /var/lib/apex_director
   ```

2. **Run with appropriate privileges**:
   ```bash
   # For system-wide installation
   sudo apex_director [command]
   
   # For user installation
   apex_director [command]
   ```

### Environment Variables

#### Symptoms
- API keys not being read
- Configuration not loading
- Missing environment-dependent settings

#### Solutions

1. **Check environment variables**:
   ```bash
   # List all relevant env vars
   env | grep -E "(API_KEY|SECRET|CONFIG)"
   
   # Check specific variable
   echo $GOOGLE_API_KEY
   ```

2. **Set missing variables**:
   ```bash
   # Add to shell profile
   echo 'export GOOGLE_API_KEY="your-key"' >> ~/.bashrc
   echo 'export NANO_BANANA_API_KEY="your-key"' >> ~/.bashrc
   
   # Reload profile
   source ~/.bashrc
   ```

3. **Use dotenv file**:
   ```bash
   # Create .env file in project directory
   cat > .env << EOF
   GOOGLE_API_KEY=your-google-key
   NANO_BANANA_API_KEY=your-nano-banana-key
   MINIMAX_API_KEY=your-minimax-key
   JWT_SECRET_KEY=your-jwt-secret
   EOF
   
   # Load in Python
   from dotenv import load_dotenv
   load_dotenv()
   ```

---

## Network and Backend Problems

### DNS Issues

#### Symptoms
- Cannot resolve backend hostnames
- Intermittent connection failures
- "Name or service not known" errors

#### Solutions

1. **Test DNS resolution**:
   ```bash
   # Test DNS for specific hosts
   nslookup vision.googleapis.com
   dig api.nanobanana.com
   
   # Check system DNS
   cat /etc/resolv.conf
   ```

2. **Switch DNS servers**:
   ```bash
   # Use Google DNS
   echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
   
   # Or use Cloudflare DNS
   echo "nameserver 1.1.1.1" | sudo tee /etc/resolv.conf
   ```

3. **Update hosts file** (if necessary):
   ```bash
   # Add entries for backend services
   sudo tee -a /etc/hosts << EOF
   172.217.14.142 vision.googleapis.com
   104.16.219.84 api.nanobanana.com
   EOF
   ```

### Firewall Issues

#### Symptoms
- Connection timeouts
- "Connection refused" errors
- Backend requests failing

#### Solutions

1. **Check firewall status**:
   ```bash
   # Check iptables
   sudo iptables -L
   
   # Check ufw (Ubuntu)
   sudo ufw status
   
   # Check firewalld (CentOS)
   sudo firewall-cmd --list-all
   ```

2. **Allow necessary ports**:
   ```bash
   # Allow HTTPS outbound (port 443)
   sudo ufw allow out 443/tcp
   
   # Allow specific backend ports
   sudo ufw allow out 8080/tcp  # Example backend port
   ```

3. **Test connectivity**:
   ```bash
   # Test specific port
   telnet vision.googleapis.com 443
   
   # Use nc (netcat)
   nc -zv vision.googleapis.com 443
   ```

### Proxy Configuration

#### Symptoms
- Backend requests failing in corporate networks
- HTTPS certificate errors
- Authentication failures

#### Solutions

1. **Configure proxy settings**:
   ```bash
   # Set proxy environment variables
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   export NO_PROXY=localhost,127.0.0.1,.local
   ```

2. **Update APEX DIRECTOR configuration**:
   ```json
   {
     "network": {
       "proxy": {
         "http": "http://proxy.company.com:8080",
         "https": "http://proxy.company.com:8080",
         "bypass": ["localhost", "127.0.0.1"]
       }
     }
   }
   ```

3. **Handle SSL certificates**:
   ```bash
   # Add corporate CA certificate
   sudo cp corporate-ca.crt /usr/local/share/ca-certificates/
   sudo update-ca-certificates
   ```

---

## Recovery Procedures

### System Recovery

#### Partial System Failure

1. **Stop the system**:
   ```bash
   apex_director stop --force
   ```

2. **Check system state**:
   ```bash
   apex_director diagnose --recovery
   ```

3. **Restore from checkpoint**:
   ```bash
   apex_director restore --checkpoint-id <checkpoint-id>
   ```

4. **Restart system**:
   ```bash
   apex_director start
   ```

#### Complete System Failure

1. **Backup current state**:
   ```bash
   apex_director backup --full
   ```

2. **Clean installation**:
   ```bash
   # Remove installation
   pip uninstall apex-director
   
   # Clean configuration
   rm -rf ~/.config/apex_director
   rm -rf ~/.cache/apex_director
   
   # Reinstall
   pip install apex-director
   ```

3. **Restore configuration and data**:
   ```bash
   apex_director restore --from-backup backup-20231029.tar.gz
   ```

### Data Recovery

#### Asset Recovery

```python
from apex_director.core.asset_manager import AssetManager

def recover_lost_assets():
    """Attempt to recover lost or corrupted assets."""
    asset_manager = AssetManager()
    
    # Find orphaned files
    orphaned_files = asset_manager.find_orphaned_files()
    
    # Attempt to restore metadata
    for file_path in orphaned_files:
        try:
            restored = asset_manager.restore_asset_metadata(file_path)
            if restored:
                print(f"Recovered: {file_path}")
        except Exception as e:
            print(f"Failed to recover {file_path}: {e}")

# Run recovery
recover_lost_assets()
```

#### Job Recovery

```python
from apex_director.core.checkpoint import CheckpointManager

def recover_failed_jobs():
    """Recover and restart failed jobs."""
    checkpoint_manager = CheckpointManager()
    
    # Find jobs with partial completion
    partial_jobs = checkpoint_manager.find_partial_jobs()
    
    for job_state in partial_jobs:
        try:
            # Restore job state
            restored_job = checkpoint_manager.restore_job_state(job_state)
            
            # Restart job
            orchestrator = get_orchestrator()
            await orchestrator.restart_job(restored_job)
            
            print(f"Recovered job: {job_state['job_id']}")
        except Exception as e:
            print(f"Failed to recover job {job_state['job_id']}: {e}")

# Run recovery
await recover_failed_jobs()
```

### Database Recovery

```bash
# Backup database
apex_director database backup

# Restore database from backup
apex_director database restore backup-20231029.sql

# Check database integrity
apex_director database check

# Rebuild database indexes
apex_director database rebuild-indexes
```

---

## Prevention Tips

### Regular Maintenance

#### Daily Tasks
```bash
#!/bin/bash
# daily_maintenance.sh

echo "Running daily maintenance..."

# Check system health
apex_director system health-check

# Clean temporary files
apex_director cleanup --temp-files --older-than 24h

# Rotate logs
apex_director logs rotate

# Check disk space
df -h /var/lib/apex_director | awk 'NR==2 {if ($5+0 > 80) print "WARNING: Disk space low"}'

# Backup configuration
apex_director config backup

echo "Daily maintenance complete"
```

#### Weekly Tasks
```bash
#!/bin/bash
# weekly_maintenance.sh

echo "Running weekly maintenance..."

# Full system backup
apex_director backup --full

# Update configuration validation
apex_director config validate --full

# Clean old checkpoints
apex_director cleanup --checkpoints --older-than 7d

# Performance analysis
apex_director performance report --last-week

# Security audit
apex_director security audit

echo "Weekly maintenance complete"
```

### Monitoring and Alerting

#### Health Monitoring Script

```python
#!/usr/bin/env python3
import smtplib
from email.mime.text import MimeText
from datetime import datetime

def send_alert(subject, message):
    """Send alert email."""
    msg = MimeText(message)
    msg['Subject'] = subject
    msg['From'] = 'alerts@apex-director.com'
    msg['To'] = 'admin@company.com'
    
    with smtplib.SMTP('localhost') as server:
        server.send_message(msg)

def check_system_health():
    """Check system health and send alerts if needed."""
    from apex_director import run_diagnostics
    
    diagnostics = run_diagnostics()
    
    # Check for critical issues
    issues = []
    
    if diagnostics['system']['disk_space_percent'] > 90:
        issues.append("Disk space critical")
    
    if diagnostics['system']['memory_usage_percent'] > 95:
        issues.append("Memory usage critical")
    
    if diagnostics['backends']['failed_count'] > 2:
        issues.append("Multiple backends failing")
    
    if diagnostics['jobs']['failure_rate'] > 0.1:
        issues.append("High job failure rate")
    
    if issues:
        send_alert(
            "APEX DIRECTOR Alert",
            f"System health issues detected:\n" + "\n".join(f"- {issue}" for issue in issues)
        )
    
    return len(issues) == 0

if __name__ == "__main__":
    check_system_health()
```

#### Performance Monitoring

```python
def setup_performance_monitoring():
    """Setup performance monitoring and alerting."""
    
    # Monitor key metrics
    metrics = [
        'job_completion_rate',
        'average_generation_time',
        'memory_usage',
        'disk_usage',
        'backend_response_time'
    ]
    
    # Set up alerting thresholds
    thresholds = {
        'job_completion_rate': {'min': 0.95},
        'average_generation_time': {'max': 300},  # 5 minutes
        'memory_usage': {'max': 0.85},
        'disk_usage': {'max': 0.80},
        'backend_response_time': {'max': 30}
    }
    
    # Implementation would depend on your monitoring system
    # (Prometheus, Grafana, CloudWatch, etc.)
    
    print("Performance monitoring configured")

setup_performance_monitoring()
```

### Best Practices Summary

1. **Regular Backups**: Always backup before major changes
2. **Monitoring**: Set up comprehensive monitoring and alerting
3. **Testing**: Test changes in development environment first
4. **Documentation**: Document any custom configurations or modifications
5. **Updates**: Keep APEX DIRECTOR and dependencies updated
6. **Security**: Regularly audit security settings and access logs
7. **Performance**: Monitor and optimize performance regularly
8. **Capacity Planning**: Monitor resource usage trends for capacity planning

This troubleshooting guide covers the most common issues you might encounter. For issues not covered here, consult the [documentation](README.md#documentation) or [contact support](#getting-help).

*Remember: When in doubt, check the logs first - they often contain the exact information needed to resolve issues.*
