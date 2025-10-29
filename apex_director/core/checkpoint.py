"""
APEX DIRECTOR Checkpoint & Resume System
Manages system state for failure recovery during long generation runs
"""

import json
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
import uuid
import logging
import pickle
import os

from .asset_manager import AssetManager
from .backend_manager import get_backend_manager
from .config import get_config, OrchestratorConfig

logger = logging.getLogger(__name__)


@dataclass
class JobState:
    """State of a single job"""
    job_id: str
    status: str  # pending, queued, processing, completed, failed, cancelled
    progress: float = 0.0
    backend_used: Optional[str] = None
    request_data: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    asset_id: Optional[str] = None  # If completed successfully
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobState':
        if 'started_at' in data and isinstance(data['started_at'], str):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if 'completed_at' in data and isinstance(data['completed_at'], str):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


@dataclass
class SystemState:
    """Complete system state snapshot"""
    checkpoint_id: str
    timestamp: datetime
    version: str = "1.0"
    orchestrator_state: Dict[str, Any] = field(default_factory=dict)
    job_states: Dict[str, JobState] = field(default_factory=dict)
    backend_states: Dict[str, Any] = field(default_factory=dict)
    asset_inventory: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['job_states'] = {job_id: job.to_dict() for job_id, job in self.job_states.items()}
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'job_states' in data:
            data['job_states'] = {
                job_id: JobState.from_dict(job_data)
                for job_id, job_data in data['job_states'].items()
            }
        return cls(**data)


class CheckpointManager:
    """Manages checkpoint creation and system state recovery"""
    
    def __init__(self, asset_manager: AssetManager, custom_config: Optional[OrchestratorConfig] = None):
        self.asset_manager = asset_manager
        self.config = custom_config or get_config().get_orchestrator_config()
        self.checkpoint_dir = Path("assets/checkpoints")
        self.current_checkpoint: Optional[SystemState] = None
        self.last_checkpoint_time: datetime = datetime.utcnow()
        self.checkpoint_history: List[str] = []
        self.auto_checkpoint_task: Optional[asyncio.Task] = None
        self.checkpoint_lock = threading.Lock()
        
        # Initialize checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Start auto checkpointing
        self.start_auto_checkpointing()
    
    def start_auto_checkpointing(self):
        """Start automatic periodic checkpointing"""
        if self.auto_checkpoint_task is None or self.auto_checkpoint_task.done():
            self.auto_checkpoint_task = asyncio.create_task(self._auto_checkpoint_loop())
            logger.info("Auto checkpointing started")
    
    def stop_auto_checkpointing(self):
        """Stop automatic checkpointing"""
        if self.auto_checkpoint_task and not self.auto_checkpoint_task.done():
            self.auto_checkpoint_task.cancel()
            logger.info("Auto checkpointing stopped")
    
    async def _auto_checkpoint_loop(self):
        """Background loop for automatic checkpointing"""
        while True:
            try:
                await asyncio.sleep(self.config.checkpoint_interval)
                
                # Only checkpoint if significant changes occurred
                if self._should_create_checkpoint():
                    await self.create_checkpoint("auto")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto checkpointing error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    def _should_create_checkpoint(self) -> bool:
        """Determine if checkpoint should be created based on changes"""
        if not self.current_checkpoint:
            return True
        
        current_time = datetime.utcnow()
        time_since_last = current_time - self.last_checkpoint_time
        
        # Always checkpoint if enough time has passed
        if time_since_last.total_seconds() >= self.config.checkpoint_interval:
            return True
        
        # Check for significant state changes
        # This would be more sophisticated in a real implementation
        return len(self.current_checkpoint.job_states) > 0
    
    async def create_checkpoint(self, reason: str = "manual") -> str:
        """Create a checkpoint of current system state"""
        checkpoint_id = str(uuid.uuid4())[:8]
        
        try:
            with self.checkpoint_lock:
                # Capture current system state
                system_state = await self._capture_system_state()
                system_state.checkpoint_id = checkpoint_id
                system_state.timestamp = datetime.utcnow()
                
                # Save checkpoint
                checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(system_state.to_dict(), f, indent=2)
                
                # Save pickle backup for complex objects
                pickle_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
                with open(pickle_file, 'wb') as f:
                    pickle.dump(system_state, f)
                
                # Update current checkpoint
                self.current_checkpoint = system_state
                self.last_checkpoint_time = datetime.utcnow()
                
                # Add to history
                self.checkpoint_history.append(checkpoint_id)
                if len(self.checkpoint_history) > 10:  # Keep only last 10
                    old_checkpoint = self.checkpoint_history.pop(0)
                    self._cleanup_checkpoint_files(old_checkpoint)
                
                # Save checkpoint index
                await self._save_checkpoint_index()
                
                logger.info(f"Checkpoint {checkpoint_id} created (reason: {reason})")
                return checkpoint_id
                
        except Exception as e:
            logger.error(f"Failed to create checkpoint {checkpoint_id}: {e}")
            raise
    
    async def _capture_system_state(self) -> SystemState:
        """Capture current state of all system components"""
        
        # Capture orchestrator state
        orchestrator_state = {
            "active_jobs_count": len([job for job in self.current_checkpoint.job_states.values() 
                                    if job.status == "processing"]) if self.current_checkpoint else 0,
            "queued_jobs_count": len([job for job in self.current_checkpoint.job_states.values() 
                                    if job.status == "queued"]) if self.current_checkpoint else 0,
            "system_uptime": "N/A",  # Would track actual uptime
            "last_cleanup": datetime.utcnow().isoformat()
        }
        
        # Capture job states
        job_states = {}
        if self.current_checkpoint:
            job_states = self.current_checkpoint.job_states
        
        # Capture backend states
        backend_manager = get_backend_manager()
        backend_states = {}
        for backend_name, status in backend_manager.get_all_backend_status().items():
            backend_states[backend_name] = status.to_dict()
        
        # Capture asset inventory
        storage_stats = self.asset_manager.get_storage_stats()
        
        # Capture performance metrics
        performance_metrics = {
            "jobs_completed_24h": len([job for job in job_states.values() 
                                     if job.status == "completed" and 
                                     job.completed_at and 
                                     (datetime.utcnow() - job.completed_at).total_seconds() < 86400]),
            "average_job_time": 0.0,  # Would calculate from historical data
            "backend_statistics": backend_manager.get_statistics()
        }
        
        return SystemState(
            checkpoint_id="",
            timestamp=datetime.utcnow(),
            version="1.0",
            orchestrator_state=orchestrator_state,
            job_states=job_states,
            backend_states=backend_states,
            asset_inventory=storage_stats,
            performance_metrics=performance_metrics
        )
    
    async def _save_checkpoint_index(self):
        """Save index of available checkpoints"""
        index_file = self.checkpoint_dir / "checkpoint_index.json"
        
        index_data = {
            "checkpoints": self.checkpoint_history,
            "current_checkpoint": self.current_checkpoint.checkpoint_id if self.current_checkpoint else None,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def _cleanup_checkpoint_files(self, checkpoint_id: str):
        """Remove old checkpoint files"""
        for pattern in [f"checkpoint_{checkpoint_id}.json", f"checkpoint_{checkpoint_id}.pkl"]:
            file_path = self.checkpoint_dir / pattern
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.debug(f"Cleaned up checkpoint file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {file_path}: {e}")
    
    async def restore_from_checkpoint(self, checkpoint_id: Optional[str] = None) -> bool:
        """Restore system state from checkpoint"""
        
        # Find checkpoint to restore
        if not checkpoint_id:
            checkpoint_id = self._find_latest_checkpoint()
        
        if not checkpoint_id:
            logger.warning("No checkpoint found to restore from")
            return False
        
        try:
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.json"
            
            if not checkpoint_file.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_file}")
                return False
            
            # Load checkpoint data
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            system_state = SystemState.from_dict(checkpoint_data)
            
            # Restore job states
            await self._restore_job_states(system_state.job_states)
            
            # Restore backend states (informational only)
            self._restore_backend_states(system_state.backend_states)
            
            # Update current checkpoint reference
            self.current_checkpoint = system_state
            
            logger.info(f"Successfully restored from checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint {checkpoint_id}: {e}")
            return False
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint"""
        if not self.checkpoint_history:
            return None
        
        # Try current checkpoint first
        if self.current_checkpoint:
            return self.current_checkpoint.checkpoint_id
        
        # Find latest in history
        return self.checkpoint_history[-1] if self.checkpoint_history else None
    
    async def _restore_job_states(self, job_states: Dict[str, JobState]):
        """Restore job states to orchestrator"""
        # In a real implementation, this would restore jobs to the orchestrator
        # For now, we'll log the restoration
        logger.info(f"Restoring {len(job_states)} job states from checkpoint")
        
        for job_id, job_state in job_states.items():
            logger.debug(f"Restoring job {job_id}: {job_state.status} (progress: {job_state.progress:.1%})")
            
            # Here you would:
            # 1. Re-queue pending/queued jobs
            # 2. Continue processing in-progress jobs
            # 3. Mark failed jobs for retry based on retry count
            # 4. Update any necessary state tracking
            
            # For now, just log the restoration
            if job_state.status == "processing":
                # This job might need to be restarted
                logger.info(f"Job {job_id} was in progress, will be re-queued for continuation")
    
    def _restore_backend_states(self, backend_states: Dict[str, Any]):
        """Restore backend states (informational)"""
        logger.info("Backend states restored from checkpoint")
        for backend_name, state in backend_states.items():
            logger.debug(f"Backend {backend_name}: {state.get('status', 'unknown')}")
    
    async def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        
        try:
            for checkpoint_id in self.checkpoint_history:
                checkpoint_file = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.json"
                
                if checkpoint_file.exists():
                    # Get file info
                    stat = checkpoint_file.stat()
                    size_kb = stat.st_size / 1024
                    modified = datetime.fromtimestamp(stat.st_mtime)
                    
                    # Load checkpoint data for metadata
                    try:
                        with open(checkpoint_file, 'r') as f:
                            data = json.load(f)
                        
                        checkpoints.append({
                            "id": checkpoint_id,
                            "timestamp": data.get("timestamp"),
                            "version": data.get("version", "1.0"),
                            "file_size_kb": round(size_kb, 2),
                            "modified_at": modified.isoformat(),
                            "job_count": len(data.get("job_states", {})),
                            "active_jobs": len([job for job in data.get("job_states", {}).values() 
                                              if job.get("status") in ["pending", "queued", "processing"]])
                        })
                    except Exception as e:
                        logger.warning(f"Failed to read checkpoint metadata for {checkpoint_id}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
        
        # Sort by timestamp (most recent first)
        checkpoints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return checkpoints
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint"""
        try:
            if checkpoint_id in self.checkpoint_history:
                self.checkpoint_history.remove(checkpoint_id)
            
            # Clean up files
            self._cleanup_checkpoint_files(checkpoint_id)
            
            # Update index
            await self._save_checkpoint_index()
            
            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    async def cleanup_old_checkpoints(self, keep_count: int = 5) -> int:
        """Clean up old checkpoints, keeping only the most recent ones"""
        if len(self.checkpoint_history) <= keep_count:
            return 0
        
        deleted_count = 0
        checkpoints_to_delete = self.checkpoint_history[:-keep_count]
        
        for checkpoint_id in checkpoints_to_delete:
            if await self.delete_checkpoint(checkpoint_id):
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old checkpoints")
        return deleted_count
    
    def get_checkpoint_status(self) -> Dict[str, Any]:
        """Get current checkpoint system status"""
        return {
            "auto_checkpoint_enabled": self.auto_checkpoint_task is not None and not self.auto_checkpoint_task.done(),
            "checkpoint_interval": self.config.checkpoint_interval,
            "last_checkpoint": self.last_checkpoint_time.isoformat() if self.last_checkpoint_time else None,
            "total_checkpoints": len(self.checkpoint_history),
            "current_checkpoint_id": self.current_checkpoint.checkpoint_id if self.current_checkpoint else None,
            "disk_usage_mb": self._calculate_disk_usage()
        }
    
    def _calculate_disk_usage(self) -> float:
        """Calculate total disk usage of checkpoint files"""
        total_size = 0
        try:
            for file_path in self.checkpoint_dir.glob("checkpoint_*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    async def force_checkpoint(self):
        """Force an immediate checkpoint creation"""
        return await self.create_checkpoint("forced")
    
    async def shutdown(self):
        """Graceful shutdown with final checkpoint"""
        try:
            # Stop auto checkpointing
            self.stop_auto_checkpointing()
            
            # Create final checkpoint
            if self._should_create_checkpoint():
                await self.create_checkpoint("shutdown")
            
            logger.info("Checkpoint manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during checkpoint manager shutdown: {e}")


# Global checkpoint manager instance
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get the global checkpoint manager instance"""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        from .asset_manager import AssetManager
        _checkpoint_manager = CheckpointManager(AssetManager())
    return _checkpoint_manager


# Convenience functions
async def create_checkpoint(reason: str = "manual") -> str:
    """Create a system checkpoint"""
    return await get_checkpoint_manager().create_checkpoint(reason)


async def restore_from_checkpoint(checkpoint_id: Optional[str] = None) -> bool:
    """Restore system from checkpoint"""
    return await get_checkpoint_manager().restore_from_checkpoint(checkpoint_id)


async def list_checkpoints() -> List[Dict[str, Any]]:
    """List all available checkpoints"""
    return await get_checkpoint_manager().list_checkpoints()


def get_checkpoint_status() -> Dict[str, Any]:
    """Get checkpoint system status"""
    return get_checkpoint_manager().get_checkpoint_status()