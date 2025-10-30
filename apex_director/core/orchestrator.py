"""
APEX DIRECTOR Main Orchestrator
Central coordinator that manages the entire image generation pipeline
"""

import asyncio
import uuid
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import json
from concurrent.futures import ThreadPoolExecutor
import threading

from .config import get_config, OrchestratorConfig
from .backend_manager import get_backend_manager, GenerationRequest, GenerationResponse
from .asset_manager import AssetManager
from .estimator import get_estimator, HistoricalRecord, CostEstimate
from .checkpoint import get_checkpoint_manager, JobState

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationJob:
    """Represents a single image generation job and its state.

    Attributes:
        id: Unique identifier for the job.
        prompt: The text prompt for image generation.
        negative_prompt: An optional text prompt to guide the generation away
            from certain concepts.
        priority: Job priority (1-5), with 5 being the highest.
        status: The current status of the job.
        created_at: Timestamp when the job was created.
        started_at: Timestamp when the job processing started.
        completed_at: Timestamp when the job was completed or failed.
        backend_preference: The preferred generation backend.
        estimated_cost: The estimated cost to run the job.
        estimated_time: The estimated time in seconds to run the job.
        actual_cost: The actual cost after the job is completed.
        actual_time: The actual time in seconds it took to run the job.
        retry_count: The number of times the job has been retried.
        error_message: The last error message if the job failed.
        asset_id: The ID of the generated asset after completion.
        metadata: A dictionary for arbitrary metadata.
        generation_params: A dictionary of parameters for the generation
            backend.
    """
    id: str
    prompt: str
    negative_prompt: Optional[str] = None
    priority: int = 3  # 1-5, higher number = higher priority
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    backend_preference: Optional[str] = None
    estimated_cost: float = 0.0
    estimated_time: float = 0.0
    actual_cost: float = 0.0
    actual_time: float = 0.0
    retry_count: int = 0
    error_message: Optional[str] = None
    asset_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the GenerationJob to a dictionary.

        Returns:
            A dictionary representation of the job.
        """
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationJob':
        """Creates a GenerationJob from a dictionary.

        Args:
            data: A dictionary containing job data.

        Returns:
            A GenerationJob instance.
        """
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'started_at' in data and isinstance(data['started_at'], str):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if 'completed_at' in data and isinstance(data['completed_at'], str):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        if 'status' in data:
            data['status'] = JobStatus(data['status'])
        return cls(**data)


class JobQueue:
    """A thread-safe, priority-based queue for managing GenerationJobs.

    This queue manages jobs based on a priority level from 1 to 5, where 5 is
    the highest priority.
    """
    
    def __init__(self):
        """Initializes the priority-based job queue."""
        self._queues: Dict[int, List[GenerationJob]] = {i: [] for i in range(1, 6)}
        self._lock = threading.Lock()
    
    def add_job(self, job: GenerationJob):
        """Adds a job to the appropriate priority queue.

        Args:
            job: The GenerationJob to add to the queue.
        """
        with self._lock:
            priority = max(1, min(5, job.priority))
            self._queues[priority].append(job)
            logger.debug(f"Added job {job.id} to priority queue {priority}")
    
    def get_next_job(self) -> Optional[GenerationJob]:
        """Retrieves the next job from the highest-priority non-empty queue.

        Jobs are retrieved in a FIFO manner from the highest priority queue
        that contains jobs.

        Returns:
            The next GenerationJob to be processed, or None if the queue is
            empty.
        """
        with self._lock:
            for priority in range(5, 0, -1):  # 5 to 1 (highest to lowest)
                if self._queues[priority]:
                    return self._queues[priority].pop(0)
            return None
    
    def remove_job(self, job_id: str) -> bool:
        """Removes a job from the queue.

        Args:
            job_id: The ID of the job to remove.

        Returns:
            True if the job was found and removed, False otherwise.
        """
        with self._lock:
            for priority_queue in self._queues.values():
                for i, job in enumerate(priority_queue):
                    if job.id == job_id:
                        priority_queue.pop(i)
                        return True
            return False
    
    def get_queue_size(self) -> int:
        """Gets the total number of jobs in the queue.

        Returns:
            The total number of jobs across all priority queues.
        """
        with self._lock:
            return sum(len(queue) for queue in self._queues.values())
    
    def get_priority_sizes(self) -> Dict[int, int]:
        """Gets the number of jobs in each priority queue.

        Returns:
            A dictionary mapping priority levels to the number of jobs in
            that queue.
        """
        with self._lock:
            return {priority: len(queue) for priority, queue in self._queues.items()}


class APEXOrchestrator:
    """The main orchestrator for the APEX DIRECTOR system.

    This class is responsible for managing the entire image generation
    pipeline, including job submission, queuing, processing, and state
    management. It coordinates all other components of the system.
    """
    
    def __init__(self, custom_config: Optional[OrchestratorConfig] = None):
        """Initializes the APEXOrchestrator.

        Args:
            custom_config: An optional OrchestratorConfig to override the
                default configuration.
        """
        self.config = custom_config or get_config().get_orchestrator_config()
        
        # Initialize components
        self.backend_manager = get_backend_manager()
        self.asset_manager = AssetManager()
        self.estimator = get_estimator()
        self.checkpoint_manager = get_checkpoint_manager()
        
        # Job management
        self.job_queue = JobQueue()
        self.active_jobs: Dict[str, GenerationJob] = {}
        self.completed_jobs: Dict[str, GenerationJob] = {}
        self.failed_jobs: Dict[str, GenerationJob] = {}
        
        # Processing control
        self.is_running = False
        self.processor_tasks: List[asyncio.Task] = []
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_jobs)
        
        # Statistics
        self.start_time = datetime.utcnow()
        self.total_jobs_processed = 0
        self.total_jobs_failed = 0
        self.total_cost_accumulated = 0.0
        
        # Event handlers
        self.job_completed_handlers: List[Callable] = []
        self.job_failed_handlers: List[Callable] = []
        
        logger.info("APEX Orchestrator initialized")
    
    async def start(self):
        """Starts the orchestrator and begins processing jobs.

        This method initializes and starts the job processor tasks and the
        health monitoring loop.
        """
        if self.is_running:
            logger.warning("Orchestrator already running")
            return
        
        self.is_running = True
        
        # Start processor tasks
        for i in range(self.config.max_concurrent_jobs):
            task = asyncio.create_task(self._job_processor(f"processor-{i}"))
            self.processor_tasks.append(task)
        
        # Start health monitoring
        health_task = asyncio.create_task(self._health_monitor_loop())
        self.processor_tasks.append(health_task)
        
        logger.info(f"Orchestrator started with {self.config.max_concurrent_jobs} processors")
    
    async def stop(self, wait_for_completion: bool = True):
        """Stops the orchestrator and its background tasks.

        Args:
            wait_for_completion: If True, waits for all processor tasks to
                finish their current jobs before stopping.
        """
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel processor tasks
        for task in self.processor_tasks:
            task.cancel()
        
        # Wait for tasks to complete if requested
        if wait_for_completion:
            await asyncio.gather(*self.processor_tasks, return_exceptions=True)
        
        logger.info("Orchestrator stopped")
    
    async def submit_job(self, 
                        prompt: str,
                        negative_prompt: Optional[str] = None,
                        priority: int = 3,
                        backend_preference: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        generation_params: Optional[Dict[str, Any]] = None) -> str:
        """Submits a new generation job to the orchestrator.

        Args:
            prompt: The text prompt for image generation.
            negative_prompt: An optional text prompt to guide the generation
                away from certain concepts.
            priority: Job priority (1-5), with 5 being the highest.
            backend_preference: The preferred generation backend.
            metadata: A dictionary for arbitrary metadata.
            generation_params: A dictionary of parameters for the generation
                backend.

        Returns:
            The unique ID of the submitted job.
        """
        job_id = str(uuid.uuid4())
        
        # Create job
        job = GenerationJob(
            id=job_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            priority=priority,
            backend_preference=backend_preference,
            metadata=metadata or {},
            generation_params=generation_params or {}
        )
        
        # Estimate cost and time
        try:
            estimate = await self._estimate_job_cost_time(job)
            job.estimated_cost = estimate.estimated_cost
            job.estimated_time = estimate.estimated_time_seconds
            
            logger.info(f"Job {job_id} estimated: ${job.estimated_cost:.4f} in {job.estimated_time:.1f}s")
            
        except Exception as e:
            logger.warning(f"Failed to estimate job {job_id}: {e}")
            # Use default estimates
            job.estimated_cost = 0.05
            job.estimated_time = 10.0
        
        # Add to queue
        self.job_queue.add_job(job)
        job.status = JobStatus.QUEUED
        
        # Save checkpoint
        await self.checkpoint_manager.create_checkpoint(f"job_submitted_{job_id}")
        
        logger.info(f"Job {job_id} submitted to queue (priority: {priority})")
        return job_id
    
    async def submit_batch(self, jobs: List[Dict[str, Any]]) -> List[str]:
        """Submits multiple jobs as a batch.

        Args:
            jobs: A list of dictionaries, where each dictionary contains the
                parameters for a single job.

        Returns:
            A list of the unique IDs of the submitted jobs.
        """
        job_ids = []
        
        for job_data in jobs:
            job_id = await self.submit_job(**job_data)
            job_ids.append(job_id)
        
        logger.info(f"Submitted batch of {len(job_ids)} jobs")
        return job_ids
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancels a job if it is still in the queue or being processed.

        Args:
            job_id: The ID of the job to cancel.

        Returns:
            True if the job was successfully cancelled, False otherwise.
        """
        # Try to remove from queue
        if self.job_queue.remove_job(job_id):
            job = next((job for job in self.active_jobs.values() if job.id == job_id), None)
            if job:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                self.failed_jobs[job_id] = job
                del self.active_jobs[job_id]
            
            logger.info(f"Job {job_id} cancelled")
            return True
        
        # Check if job is currently processing
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            self.failed_jobs[job_id] = job
            del self.active_jobs[job_id]
            
            logger.info(f"Job {job_id} cancelled during processing")
            return True
        
        logger.warning(f"Job {job_id} not found or cannot be cancelled")
        return False
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Gets the status of a specific job.

        Args:
            job_id: The ID of the job to check.

        Returns:
            A dictionary containing the job's status and details, or None if
            the job is not found.
        """
        # Check active jobs
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return job.to_dict()
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            return job.to_dict()
        
        # Check failed jobs
        if job_id in self.failed_jobs:
            job = self.failed_jobs[job_id]
            return job.to_dict()
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Gets the current status of the job queue.

        Returns:
            A dictionary with statistics about the job queue, including the
            number of queued, active, completed, and failed jobs.
        """
        return {
            "total_queued": self.job_queue.get_queue_size(),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "failed_jobs": len(self.failed_jobs),
            "priority_breakdown": self.job_queue.get_priority_sizes()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Gets a comprehensive status of the entire system.

        This includes the orchestrator status, job queue statistics,
        performance metrics, and backend and asset information.

        Returns:
            A dictionary containing a snapshot of the system's status.
        """
        uptime = datetime.utcnow() - self.start_time
        
        backend_stats = self.backend_manager.get_statistics()
        
        return {
            "orchestrator": {
                "status": "running" if self.is_running else "stopped",
                "uptime_seconds": uptime.total_seconds(),
                "active_processors": len(self.processor_tasks),
                "max_concurrent_jobs": self.config.max_concurrent_jobs
            },
            "jobs": self.get_queue_status(),
            "performance": {
                "total_jobs_processed": self.total_jobs_processed,
                "total_jobs_failed": self.total_jobs_failed,
                "success_rate": self.total_jobs_processed / (self.total_jobs_processed + self.total_jobs_failed) if (self.total_jobs_processed + self.total_jobs_failed) > 0 else 0,
                "total_cost_accumulated": self.total_cost_accumulated
            },
            "backends": backend_stats,
            "assets": self.asset_manager.get_storage_stats()
        }
    
    async def _job_processor(self, processor_id: str):
        """The main loop for a single job processor task.

        This method continuously fetches jobs from the queue and processes them.

        Args:
            processor_id: A unique identifier for the processor.
        """
        logger.info(f"Job processor {processor_id} started")
        
        while self.is_running:
            try:
                # Get next job from queue
                job = self.job_queue.get_next_job()
                if not job:
                    await asyncio.sleep(1)
                    continue
                
                # Process the job
                await self._process_job(job, processor_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Job processor {processor_id} error: {e}")
                await asyncio.sleep(5)
        
        logger.info(f"Job processor {processor_id} stopped")
    
    async def _process_job(self, job: GenerationJob, processor_id: str):
        """Processes a single generation job.

        This method handles the entire lifecycle of a job, from sending it to
        the backend to saving the asset and recording the history.

        Args:
            job: The GenerationJob to process.
            processor_id: The ID of the processor handling the job.
        """
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()
        self.active_jobs[job.id] = job
        
        logger.info(f"Processor {processor_id} processing job {job.id}")
        
        try:
            # Create generation request
            request = GenerationRequest(
                job_id=job.id,
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                width=job.generation_params.get('width', 512),
                height=job.generation_params.get('height', 512),
                steps=job.generation_params.get('steps', 20),
                guidance_scale=job.generation_params.get('guidance_scale', 7.5),
                seed=job.generation_params.get('seed'),
                quality_level=job.generation_params.get('quality_level', 3)
            )
            
            # Select backend if preference specified
            if job.backend_preference:
                # Override normal backend selection
                pass  # Would implement backend-specific logic
            
            # Generate image with fallback
            response = await self.backend_manager.generate_with_fallback(request)
            
            if response.success:
                # Save asset
                asset_id = await self._save_generated_asset(job, response)
                job.asset_id = asset_id
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.actual_cost = response.cost
                job.actual_time = response.generation_time
                
                # Update statistics
                self.total_jobs_processed += 1
                self.total_cost_accumulated += response.cost
                
                # Move to completed jobs
                self.completed_jobs[job.id] = job
                del self.active_jobs[job.id]
                
                # Log completion
                logger.info(f"Job {job.id} completed successfully: {job.actual_time:.1f}s, ${job.actual_cost:.4f}")
                
                # Notify handlers
                for handler in self.job_completed_handlers:
                    try:
                        await handler(job)
                    except Exception as e:
                        logger.error(f"Job completed handler error: {e}")
                
                # Add to historical data for estimation
                await self._record_generation_history(job, response)
                
            else:
                # Handle failure
                await self._handle_job_failure(job, response)
        
        except Exception as e:
            logger.error(f"Job {job.id} processing error: {e}")
            await self._handle_job_failure(job, None, str(e))
        
        finally:
            # Save checkpoint after job completion
            await self.checkpoint_manager.create_checkpoint(f"job_completed_{job.id}")
    
    async def _estimate_job_cost_time(self, job: GenerationJob) -> CostEstimate:
        """Estimates the cost and time for a given job.

        Args:
            job: The GenerationJob to estimate.

        Returns:
            A CostEstimate object with the estimated cost and time.
        """
        # Prepare job parameters for estimation
        job_params = {
            "width": job.generation_params.get('width', 512),
            "height": job.generation_params.get('height', 512),
            "steps": job.generation_params.get('steps', 20),
            "quality_level": job.generation_params.get('quality_level', 3)
        }
        
        # Get estimate
        estimate = self.estimator.estimate_generation(job_params, job.prompt)
        return estimate
    
    async def _save_generated_asset(self, job: GenerationJob, response: GenerationResponse) -> str:
        """Saves the generated asset to storage.

        This method creates a placeholder for the generated image and
        registers it with the AssetManager.

        Args:
            job: The job that generated the asset.
            response: The response from the generation backend.

        Returns:
            The ID of the newly registered asset.
        """
        try:
            # In a real implementation, this would save the actual image data
            # For now, we'll create a placeholder
            
            # Create mock image data
            mock_image_data = f"Mock image for job {job.id} using {response.backend_used}"
            
            # Save to temporary file
            from pathlib import Path
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            image_file = temp_dir / f"job_{job.id}.png"
            with open(image_file, 'w') as f:
                f.write(mock_image_data)
            
            # Register with asset manager
            asset_id = self.asset_manager.register_asset(
                file_path=str(image_file),
                job_id=job.id,
                prompt=job.prompt,
                backend_used=response.backend_used,
                generation_params=job.generation_params
            )
            
            return asset_id
            
        except Exception as e:
            logger.error(f"Failed to save asset for job {job.id}: {e}")
            raise
    
    async def _handle_job_failure(self, job: GenerationJob, response: Optional[GenerationResponse], error_msg: Optional[str] = None):
        """Handles a job failure, with potential for retries.

        If the job has not exceeded its retry limit, it is re-queued.
        Otherwise, it is marked as failed.

        Args:
            job: The job that failed.
            response: The backend response, if available.
            error_msg: An optional error message.
        """
        error_message = error_msg or response.error if response else "Unknown error"
        
        # Check if we should retry
        job.retry_count += 1
        if (self.config.auto_retry and 
            job.retry_count <= self.config.failure_threshold and
            job.status != JobStatus.CANCELLED):
            
            logger.info(f"Job {job.id} failed, retrying ({job.retry_count}/{self.config.failure_threshold})")
            
            # Wait before retry
            await asyncio.sleep(self.config.retry_delay)
            
            # Re-queue job
            job.status = JobStatus.QUEUED
            job.started_at = None
            job.error_message = error_message
            self.job_queue.add_job(job)
            
            # Remove from active jobs
            if job.id in self.active_jobs:
                del self.active_jobs[job.id]
            
        else:
            # Final failure
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = error_message
            
            self.failed_jobs[job.id] = job
            if job.id in self.active_jobs:
                del self.active_jobs[job.id]
            
            self.total_jobs_failed += 1
            
            logger.error(f"Job {job.id} failed permanently: {error_message}")
            
            # Notify handlers
            for handler in self.job_failed_handlers:
                try:
                    await handler(job)
                except Exception as e:
                    logger.error(f"Job failed handler error: {e}")
    
    async def _record_generation_history(self, job: GenerationJob, response: GenerationResponse):
        """Records the details of a completed job for future estimations.

        Args:
            job: The completed job.
            response: The successful response from the backend.
        """
        try:
            record = HistoricalRecord(
                timestamp=job.completed_at or datetime.utcnow(),
                backend=response.backend_used,
                width=job.generation_params.get('width', 512),
                height=job.generation_params.get('height', 512),
                steps=job.generation_params.get('steps', 20),
                quality_level=job.generation_params.get('quality_level', 3),
                actual_time=job.actual_time,
                actual_cost=job.actual_cost,
                success=True,
                prompt_length=len(job.prompt),
                complexity_score=0.0  # Would calculate actual complexity
            )
            
            self.estimator.add_historical_record(record)
            
        except Exception as e:
            logger.warning(f"Failed to record generation history: {e}")
    
    async def _health_monitor_loop(self):
        """A background task that periodically monitors the system's health.

        This method checks backend statuses, system resources, and triggers
        periodic checkpoints.
        """
        logger.info("Health monitor started")
        
        while self.is_running:
            try:
                # Check backend health
                backend_statuses = self.backend_manager.get_all_backend_status()
                
                # Log any backend issues
                for backend_name, status in backend_statuses.items():
                    if status.status != "online":
                        logger.warning(f"Backend {backend_name} status: {status.status}")
                
                # Check system resources (simplified)
                if len(self.active_jobs) > self.config.max_concurrent_jobs:
                    logger.warning("Too many active jobs")
                
                # Periodic checkpoint
                await self.checkpoint_manager.create_checkpoint("health_check")
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Health monitor stopped")
    
    def add_job_completed_handler(self, handler: Callable):
        """Adds a callback handler for job completion events.

        Args:
            handler: A callable that will be invoked when a job is
                successfully completed.
        """
        self.job_completed_handlers.append(handler)
    
    def add_job_failed_handler(self, handler: Callable):
        """Adds a callback handler for job failure events.

        Args:
            handler: A callable that will be invoked when a job fails.
        """
        self.job_failed_handlers.append(handler)
    
    async def force_checkpoint(self):
        """Forces the CheckpointManager to create a checkpoint immediately."""
        await self.checkpoint_manager.force_checkpoint()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Gets detailed statistics about the orchestrator's performance.

        Returns:
            A dictionary containing performance metrics such as jobs per hour,
            costs, success rate, and backend usage.
        """
        # Calculate rates
        uptime_hours = (datetime.utcnow() - self.start_time).total_seconds() / 3600
        jobs_per_hour = self.total_jobs_processed / uptime_hours if uptime_hours > 0 else 0
        
        # Cost statistics
        avg_cost_per_job = self.total_cost_accumulated / self.total_jobs_processed if self.total_jobs_processed > 0 else 0
        
        # Backend usage
        backend_usage = {}
        for job in list(self.completed_jobs.values()) + list(self.failed_jobs.values()):
            if hasattr(job, 'backend_used') and job.backend_used:
                backend_usage[job.backend_used] = backend_usage.get(job.backend_used, 0) + 1
        
        return {
            "uptime_hours": uptime_hours,
            "jobs_per_hour": jobs_per_hour,
            "total_cost": self.total_cost_accumulated,
            "average_cost_per_job": avg_cost_per_job,
            "success_rate": self.total_jobs_processed / (self.total_jobs_processed + self.total_jobs_failed) if (self.total_jobs_processed + self.total_jobs_failed) > 0 else 0,
            "backend_usage": backend_usage,
            "queue_status": self.get_queue_status()
        }


# Global orchestrator instance
_orchestrator: Optional[APEXOrchestrator] = None


def get_orchestrator() -> APEXOrchestrator:
    """Gets the global instance of the APEXOrchestrator.

    This function implements a singleton pattern to ensure that only one
    instance of the orchestrator exists.

    Returns:
        The global APEXOrchestrator instance.
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = APEXOrchestrator()
    return _orchestrator


# Convenience functions
async def start_orchestrator():
    """Starts the global APEXOrchestrator instance."""
    orchestrator = get_orchestrator()
    await orchestrator.start()


async def stop_orchestrator():
    """Stops the global APEXOrchestrator instance."""
    orchestrator = get_orchestrator()
    await orchestrator.stop()


async def submit_generation_job(*args, **kwargs) -> str:
    """Convenience function to submit a job to the global orchestrator.

    Args:
        *args: Positional arguments to be passed to the
            APEXOrchestrator.submit_job method.
        **kwargs: Keyword arguments to be passed to the
            APEXOrchestrator.submit_job method.

    Returns:
        The unique ID of the submitted job.
    """
    orchestrator = get_orchestrator()
    return await orchestrator.submit_job(*args, **kwargs)


def get_system_status() -> Dict[str, Any]:
    """Get system status"""
    orchestrator = get_orchestrator()
    return orchestrator.get_system_status()