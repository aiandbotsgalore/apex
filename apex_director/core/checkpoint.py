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
    """Represents the state of a single job at a point in time.

    Attributes:
        job_id: The unique identifier for the job.
        status: The current status of the job (e.g., "pending", "completed").
        progress: The progress of the job as a float between 0.0 and 1.0.
        backend_used: The name of the backend used for the job.
        request_data: A dictionary containing the original request data.
        started_at: The timestamp when the job started processing.
        completed_at: The timestamp when the job was completed or failed.
        error_message: The last error message, if the job failed.
        retry_count: The number of times the job has been retried.
        asset_id: The ID of the generated asset, if the job was successful.
    """
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
        """Converts the JobState to a dictionary.

        Returns:
            A dictionary representation of the job state.
        """
        data = asdict(self)
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobState':
        """Creates a JobState instance from a dictionary.

        Args:
            data: A dictionary containing job state data.

        Returns:
            An instance of JobState.
        """
        if 'started_at' in data and isinstance(data['started_at'], str):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if 'completed_at' in data and isinstance(data['completed_at'], str):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


@dataclass
class SystemState:
    """Represents a complete snapshot of the system's state.

    Attributes:
        checkpoint_id: The unique identifier for this checkpoint.
        timestamp: The timestamp when the checkpoint was created.
        version: The version of the system state schema.
        orchestrator_state: A dictionary containing the state of the
            orchestrator.
        job_states: A dictionary mapping job IDs to their JobState.
        backend_states: A dictionary containing the states of the backends.
        asset_inventory: A dictionary summarizing the asset inventory.
        performance_metrics: A dictionary of performance metrics at the time
            of the checkpoint.
    """
    checkpoint_id: str
    timestamp: datetime
    version: str = "1.0"
    orchestrator_state: Dict[str, Any] = field(default_factory=dict)
    job_states: Dict[str, JobState] = field(default_factory=dict)
    backend_states: Dict[str, Any] = field(default_factory=dict)
    asset_inventory: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the SystemState to a dictionary.

        Returns:
            A dictionary representation of the system state.
        """
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['job_states'] = {job_id: job.to_dict() for job_id, job in self.job_states.items()}
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        """Creates a SystemState instance from a dictionary.

        Args:
            data: A dictionary containing system state data.

        Returns:
            An instance of SystemState.
        """
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'job_states' in data:
            data['job_states'] = {
                job_id: JobState.from_dict(job_data)
                for job_id, job_data in data['job_states'].items()
            }
        return cls(**data)


class CheckpointManager:
    """Manages the creation and recovery of system state checkpoints.

    This class provides functionalities for automatic and manual checkpoint
    creation, as well as for restoring the system to a previous state.
    """
    
    def __init__(self, asset_manager: AssetManager, custom_config: Optional[OrchestratorConfig] = None):
        """Initializes the CheckpointManager.

        Args:
            asset_manager: An instance of the AssetManager.
            custom_config: An optional OrchestratorConfig to override the
                default configuration.
        """
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
        """Starts the automatic periodic checkpointing task."""
        if self.auto_checkpoint_task is None or self.auto_checkpoint_task.done():
            self.auto_checkpoint_task = asyncio.create_task(self._auto_checkpoint_loop())
            logger.info("Auto checkpointing started")
    
    def stop_auto_checkpointing(self):
        """Stops the automatic checkpointing task."""
        if self.auto_checkpoint_task and not self.auto_checkpoint_task.done():
            self.auto_checkpoint_task.cancel()
            logger.info("Auto checkpointing stopped")
    
    async def _auto_checkpoint_loop(self):
        """The background loop for performing automatic checkpoints."""
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
        """Determines if a checkpoint should be created.

        A checkpoint is created if enough time has passed since the last one
        or if there have been significant changes to the system state.

        Returns:
            True if a checkpoint should be created, False otherwise.
        """
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
        """Creates a checkpoint of the current system state.

        Args:
            reason: The reason for creating the checkpoint (e.g., "manual",
                "auto").

        Returns:
            The ID of the newly created checkpoint.
        """
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
        """Captures the current state of all system components.

        Returns:
            A SystemState object representing the current state of the system.
        """
        
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
        """Saves the index of available checkpoints to a file."""
        index_file = self.checkpoint_dir / "checkpoint_index.json"
        
        index_data = {
            "checkpoints": self.checkpoint_history,
            "current_checkpoint": self.current_checkpoint.checkpoint_id if self.current_checkpoint else None,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def _cleanup_checkpoint_files(self, checkpoint_id: str):
        """Removes the files associated with a specific checkpoint.

        Args:
            checkpoint_id: The ID of the checkpoint to clean up.
        """
        for pattern in [f"checkpoint_{checkpoint_id}.json", f"checkpoint_{checkpoint_id}.pkl"]:
            file_path = self.checkpoint_dir / pattern
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.debug(f"Cleaned up checkpoint file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {file_path}: {e}")
    
    async def restore_from_checkpoint(self, checkpoint_id: Optional[str] = None) -> bool:
        """Restores the system state from a specified checkpoint.

        If no checkpoint ID is provided, it restores from the latest
        available checkpoint.

        Args:
            checkpoint_id: The ID of the checkpoint to restore from.

        Returns:
            True if the restoration was successful, False otherwise.
        """
        
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
        """Finds the ID of the most recent checkpoint.

        Returns:
            The ID of the latest checkpoint, or None if no checkpoints exist.
        """
        if not self.checkpoint_history:
            return None
        
        # Try current checkpoint first
        if self.current_checkpoint:
            return self.current_checkpoint.checkpoint_id
        
        # Find latest in history
        return self.checkpoint_history[-1] if self.checkpoint_history else None
    
    async def _restore_job_states(self, job_states: Dict[str, JobState]):
        """Restores the states of jobs to the orchestrator.

        In a real implementation, this would re-queue jobs and update the
        orchestrator's state.

        Args:
            job_states: A dictionary of job states to restore.
        """
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
        """Restores backend states from a checkpoint.

        Note:
            This is currently for informational purposes only.

        Args:
            backend_states: A dictionary of backend states to restore.
        """
        logger.info("Backend states restored from checkpoint")
        for backend_name, state in backend_states.items():
            logger.debug(f"Backend {backend_name}: {state.get('status', 'unknown')}")
    
    async def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Lists all available checkpoints.

        Returns:
            A list of dictionaries, where each dictionary contains metadata
            about a checkpoint.
        """
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
        """Deletes a specific checkpoint.

        Args:
            checkpoint_id: The ID of the checkpoint to delete.

        Returns:
            True if the deletion was successful, False otherwise.
        """
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
        """Cleans up old checkpoints, keeping a specified number of recent ones.

        Args:
            keep_count: The number of recent checkpoints to keep.

        Returns:
            The number of checkpoints that were deleted.
        """
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
        """Gets the current status of the checkpoint system.

        Returns:
            A dictionary containing the status of the checkpoint system.
        """
        return {
            "auto_checkpoint_enabled": self.auto_checkpoint_task is not None and not self.auto_checkpoint_task.done(),
            "checkpoint_interval": self.config.checkpoint_interval,
            "last_checkpoint": self.last_checkpoint_time.isoformat() if self.last_checkpoint_time else None,
            "total_checkpoints": len(self.checkpoint_history),
            "current_checkpoint_id": self.current_checkpoint.checkpoint_id if self.current_checkpoint else None,
            "disk_usage_mb": self._calculate_disk_usage()
        }
    
    def _calculate_disk_usage(self) -> float:
        """Calculates the total disk usage of all checkpoint files.

        Returns:
            The total disk usage in megabytes.
        """
        total_size = 0
        try:
            for file_path in self.checkpoint_dir.glob("checkpoint_*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    async def force_checkpoint(self):
        """Forces an immediate checkpoint creation."""
        return await self.create_checkpoint("forced")
    
    async def shutdown(self):
        """Performs a graceful shutdown of the checkpoint manager.

        This includes stopping the auto-checkpointing task and creating a
        final checkpoint.
        """
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
    """Gets the global instance of the CheckpointManager.

    This function implements a singleton pattern to ensure that only one
    instance of the checkpoint manager exists.

    Returns:
        The global CheckpointManager instance.
    """
    global _checkpoint_manager
    if _checkpoint_manager is None:
        from .asset_manager import AssetManager
        _checkpoint_manager = CheckpointManager(AssetManager())
    return _checkpoint_manager


# Convenience functions
async def create_checkpoint(reason: str = "manual") -> str:
    """A convenience function to create a system checkpoint.

    Args:
        reason: The reason for creating the checkpoint.

    Returns:
        The ID of the newly created checkpoint.
    """
    return await get_checkpoint_manager().create_checkpoint(reason)


async def restore_from_checkpoint(checkpoint_id: Optional[str] = None) -> bool:
    """A convenience function to restore the system from a checkpoint.

    Args:
        checkpoint_id: The ID of the checkpoint to restore from. If None,
            restores from the latest checkpoint.

    Returns:
        True if the restoration was successful, False otherwise.
    """
    return await get_checkpoint_manager().restore_from_checkpoint(checkpoint_id)


async def list_checkpoints() -> List[Dict[str, Any]]:
    """A convenience function to list all available checkpoints.

    Returns:
        A list of dictionaries, where each dictionary contains metadata
        about a checkpoint.
    """
    return await get_checkpoint_manager().list_checkpoints()


def get_checkpoint_status() -> Dict[str, Any]:
    """A convenience function to get the status of the checkpoint system.

    Returns:
        A dictionary containing the status of the checkpoint system.
    """
    return get_checkpoint_manager().get_checkpoint_status()