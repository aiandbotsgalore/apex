"""
APEX DIRECTOR Progress Monitoring System
Tracks and monitors progress through the music video generation workflow
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import threading
from collections import defaultdict

from ..core.orchestrator import JobStatus, GenerationJob
from ..core.backend_manager import GenerationRequest, GenerationResponse

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Stages of the music video generation workflow"""
    INPUT_VALIDATION = "input_validation"
    TREATMENT_GENERATION = "treatment_generation"
    STORYBOARD_CREATION = "storyboard_creation"
    IMAGE_GENERATION = "image_generation"
    VIDEO_ASSEMBLY = "video_assembly"
    POST_PROCESSING = "post_processing"
    DELIVERY = "delivery"
    COMPLETE = "complete"


class TaskStatus(Enum):
    """Status of individual tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class Priority(Enum):
    """Task priorities"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


@dataclass
class WorkflowTask:
    """Represents an individual task within a workflow.

    Attributes:
        id: The unique identifier for the task.
        stage: The workflow stage this task belongs to.
        name: The name of the task.
        description: A description of the task.
        status: The current status of the task.
        priority: The priority of the task.
        created_at: The timestamp when the task was created.
        started_at: The timestamp when the task was started.
        completed_at: The timestamp when the task was completed.
        estimated_duration: The estimated duration of the task in seconds.
        actual_duration: The actual duration of the task in seconds.
        progress: The progress of the task, from 0.0 to 1.0.
        dependencies: A list of task IDs that this task depends on.
        result: A dictionary of results from the task.
        error_message: An error message if the task failed.
        metadata: A dictionary of additional metadata.
        callbacks: A list of callback functions to be called when the task is completed.
    """
    id: str
    stage: WorkflowStage
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: float = 0.0  # in seconds
    actual_duration: float = 0.0
    progress: float = 0.0  # 0.0 to 1.0
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    callbacks: List[Callable] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the WorkflowTask to a dictionary.

        Returns:
            A dictionary representation of the WorkflowTask.
        """
        data = asdict(self)
        data['stage'] = self.stage.value
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data
    
    @property
    def duration_so_far(self) -> float:
        """Gets the current duration of the task.

        Returns:
            The current duration of the task in seconds.
        """
        if self.started_at:
            end_time = self.completed_at or datetime.utcnow()
            return (end_time - self.started_at).total_seconds()
        return 0.0
    
    @property
    def is_completed(self) -> bool:
        """Checks if the task is completed.

        Returns:
            True if the task is completed, False otherwise.
        """
        return self.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
    
    @property
    def is_failed(self) -> bool:
        """Checks if the task has failed.

        Returns:
            True if the task has failed, False otherwise.
        """
        return self.status == TaskStatus.FAILED


@dataclass
class WorkflowStageProgress:
    """Represents the progress of a workflow stage.

    Attributes:
        stage: The workflow stage.
        total_tasks: The total number of tasks in the stage.
        completed_tasks: The number of completed tasks in the stage.
        failed_tasks: The number of failed tasks in the stage.
        current_task_id: The ID of the currently running task in the stage.
        stage_progress: The progress of the stage, from 0.0 to 1.0.
        estimated_remaining: The estimated remaining time for the stage in seconds.
        started_at: The timestamp when the stage was started.
        completed_at: The timestamp when the stage was completed.
    """
    stage: WorkflowStage
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    current_task_id: Optional[str] = None
    stage_progress: float = 0.0
    estimated_remaining: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def percentage_complete(self) -> float:
        """Gets the percentage of the stage that is complete.

        Returns:
            The percentage of the stage that is complete.
        """
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100.0


@dataclass
class WorkflowProgress:
    """Represents the complete progress of a workflow.

    Attributes:
        workflow_id: The unique identifier for the workflow.
        project_name: The name of the project.
        current_stage: The current stage of the workflow.
        stage_progress: A dictionary of progress for each stage in the workflow.
        total_tasks: The total number of tasks in the workflow.
        completed_tasks: The number of completed tasks in the workflow.
        failed_tasks: The number of failed tasks in the workflow.
        overall_progress: The overall progress of the workflow, from 0.0 to 1.0.
        estimated_completion: The estimated completion time of the workflow.
        started_at: The timestamp when the workflow was started.
        last_updated: The timestamp when the workflow was last updated.
        status: The current status of the workflow.
    """
    workflow_id: str
    project_name: str
    current_stage: WorkflowStage
    stage_progress: Dict[str, WorkflowStageProgress]
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    overall_progress: float = 0.0
    estimated_completion: Optional[datetime] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    status: str = "running"  # running, completed, failed, cancelled
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the WorkflowProgress to a dictionary.

        Returns:
            A dictionary representation of the WorkflowProgress.
        """
        data = asdict(self)
        data['current_stage'] = self.current_stage.value
        data['stage_progress'] = {k: asdict(v) for k, v in self.stage_progress.items()}
        data['started_at'] = self.started_at.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        if self.estimated_completion:
            data['estimated_completion'] = self.estimated_completion.isoformat()
        return data


class ProgressMonitor:
    """A comprehensive system for monitoring the progress of workflows.

    This class provides functionality for:
    - Creating and managing workflows and tasks
    - Starting, stopping, and updating the progress of tasks
    - Monitoring the overall progress of workflows
    - Exporting progress reports
    - Triggering event handlers for workflow events
    """
    
    def __init__(self):
        """Initializes the ProgressMonitor."""
        self.active_workflows: Dict[str, WorkflowProgress] = {}
        self.tasks: Dict[str, WorkflowTask] = {}
        self.stage_handlers: Dict[WorkflowStage, Callable] = {}
        self.event_handlers: Dict[str, List[Callable]] = {
            'task_started': [],
            'task_completed': [],
            'task_failed': [],
            'stage_changed': [],
            'workflow_completed': []
        }
        self._lock = threading.Lock()
        self._update_thread: Optional[threading.Thread] = None
        self._is_monitoring = False
        
        logger.info("Progress Monitor initialized")
    
    def start_monitoring(self):
        """Starts the progress monitoring system."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        logger.info("Progress monitoring started")
    
    def stop_monitoring(self):
        """Stops the progress monitoring system."""
        self._is_monitoring = False
        if self._update_thread:
            self._update_thread.join(timeout=5)
        logger.info("Progress monitoring stopped")
    
    def _update_loop(self):
        """Background update loop"""
        while self._is_monitoring:
            try:
                self._update_all_workflows()
                time.sleep(1)  # Update every second
            except Exception as e:
                logger.error(f"Error in progress update loop: {e}")
                time.sleep(5)
    
    def _update_all_workflows(self):
        """Update progress for all active workflows"""
        with self._lock:
            for workflow in list(self.active_workflows.values()):
                self._update_workflow_progress(workflow)
    
    def _update_workflow_progress(self, workflow: WorkflowProgress):
        """Update progress for a specific workflow"""
        try:
            # Update stage progress
            for stage_name, stage_progress in workflow.stage_progress.items():
                self._update_stage_progress(stage_progress)
            
            # Calculate overall progress
            self._calculate_overall_progress(workflow)
            
            # Update timestamp
            workflow.last_updated = datetime.utcnow()
            
            # Check for completion
            if workflow.completed_tasks + workflow.failed_tasks >= workflow.total_tasks:
                workflow.status = "completed" if workflow.failed_tasks == 0 else "failed"
                self._trigger_event('workflow_completed', workflow)
            
        except Exception as e:
            logger.error(f"Error updating workflow {workflow.workflow_id}: {e}")
    
    def _update_stage_progress(self, stage_progress: WorkflowStageProgress):
        """Update progress for a workflow stage"""
        # This would normally query task statuses and update progress
        # For now, we'll update based on known task completion
        pass
    
    def _calculate_overall_progress(self, workflow: WorkflowProgress):
        """Calculate overall workflow progress"""
        total_tasks = workflow.total_tasks
        if total_tasks == 0:
            workflow.overall_progress = 0.0
            return
        
        # Weight progress by stage and task importance
        weighted_progress = 0.0
        total_weight = 0.0
        
        for stage_name, stage_progress in workflow.stage_progress.items():
            stage_weight = stage_progress.total_tasks  # Simple weight by task count
            weighted_progress += stage_progress.stage_progress * stage_weight
            total_weight += stage_weight
        
        if total_weight > 0:
            workflow.overall_progress = (weighted_progress / total_weight) * 100.0
        else:
            workflow.overall_progress = 0.0
    
    def create_workflow(self, 
                       project_name: str,
                       workflow_config: Optional[Dict[str, Any]] = None) -> str:
        """Creates a new workflow for progress tracking.

        Args:
            project_name: The name of the project.
            workflow_config: Optional workflow configuration.

        Returns:
            The ID of the newly created workflow.
        """
        workflow_id = str(uuid.uuid4())
        
        # Initialize stage progress tracking
        stage_progress = {}
        for stage in WorkflowStage:
            if stage != WorkflowStage.COMPLETE:  # Don't track the final completion stage
                stage_progress[stage.value] = WorkflowStageProgress(
                    stage=stage,
                    total_tasks=0,
                    completed_tasks=0,
                    failed_tasks=0
                )
        
        workflow = WorkflowProgress(
            workflow_id=workflow_id,
            project_name=project_name,
            current_stage=WorkflowStage.INPUT_VALIDATION,
            stage_progress=stage_progress,
            total_tasks=0,
            completed_tasks=0,
            failed_tasks=0
        )
        
        with self._lock:
            self.active_workflows[workflow_id] = workflow
        
        logger.info(f"Created workflow {workflow_id} for project: {project_name}")
        return workflow_id
    
    def add_task(self,
                workflow_id: str,
                stage: WorkflowStage,
                name: str,
                description: str,
                priority: Priority = Priority.NORMAL,
                estimated_duration: float = 0.0,
                dependencies: Optional[List[str]] = None) -> str:
        """Adds a task to a workflow.

        Args:
            workflow_id: The ID of the workflow to add the task to.
            stage: The workflow stage this task belongs to.
            name: The name of the task.
            description: A description of the task.
            priority: The priority of the task.
            estimated_duration: The estimated duration of the task in seconds.
            dependencies: A list of task IDs that this task depends on.

        Returns:
            The ID of the newly created task.
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        task_id = str(uuid.uuid4())
        
        task = WorkflowTask(
            id=task_id,
            stage=stage,
            name=name,
            description=description,
            priority=priority,
            estimated_duration=estimated_duration,
            dependencies=dependencies or []
        )
        
        with self._lock:
            self.tasks[task_id] = task
            
            # Update workflow
            workflow = self.active_workflows[workflow_id]
            workflow.total_tasks += 1
            
            # Update stage progress
            stage_progress = workflow.stage_progress[stage.value]
            stage_progress.total_tasks += 1
        
        logger.info(f"Added task {task_id} to workflow {workflow_id}")
        return task_id
    
    def start_task(self, task_id: str) -> bool:
        """Marks a task as started.

        Args:
            task_id: The ID of the task to start.

        Returns:
            True if the task was successfully started, False otherwise.
        """
        with self._lock:
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found")
                return False
            
            task = self.tasks[task_id]
            if task.status != TaskStatus.PENDING:
                logger.warning(f"Task {task_id} already started with status: {task.status}")
                return False
            
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            task.progress = 0.0
            
            # Update workflow stage progress
            self._update_task_in_workflow(task)
            
        logger.info(f"Started task {task_id}")
        self._trigger_event('task_started', task)
        return True
    
    def update_task_progress(self, task_id: str, progress: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Updates the progress of a task.

        Args:
            task_id: The ID of the task to update.
            progress: The new progress of the task, from 0.0 to 1.0.
            metadata: Optional metadata to add to the task.

        Returns:
            True if the task was successfully updated, False otherwise.
        """
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1
            
            if metadata:
                task.metadata.update(metadata)
            
            # Update workflow progress
            self._update_task_in_workflow(task)
        
        return True
    
    def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """Marks a task as completed.

        Args:
            task_id: The ID of the task to complete.
            result: Optional results from the task.

        Returns:
            True if the task was successfully completed, False otherwise.
        """
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = datetime.utcnow()
            task.result = result
            
            if task.started_at:
                task.actual_duration = (task.completed_at - task.started_at).total_seconds()
            
            # Update workflow progress
            self._update_task_in_workflow(task)
        
        logger.info(f"Completed task {task_id}")
        self._trigger_event('task_completed', task)
        return True
    
    def fail_task(self, task_id: str, error_message: str) -> bool:
        """Marks a task as failed.

        Args:
            task_id: The ID of the task to mark as failed.
            error_message: The error message associated with the failure.

        Returns:
            True if the task was successfully marked as failed, False otherwise.
        """
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.status = TaskStatus.FAILED
            task.error_message = error_message
            task.completed_at = datetime.utcnow()
            
            if task.started_at:
                task.actual_duration = (task.completed_at - task.started_at).total_seconds()
            
            # Update workflow progress
            self._update_task_in_workflow(task)
        
        logger.error(f"Failed task {task_id}: {error_message}")
        self._trigger_event('task_failed', task)
        return True
    
    def _update_task_in_workflow(self, task: WorkflowTask):
        """Update workflow progress based on task status"""
        # Find workflows containing this task
        for workflow in self.active_workflows.values():
            stage_progress = workflow.stage_progress.get(task.stage.value)
            if stage_progress:
                # Recalculate stage progress
                stage_tasks = [t for t in self.tasks.values() if t.stage == task.stage]
                
                completed = len([t for t in stage_tasks if t.status == TaskStatus.COMPLETED])
                failed = len([t for t in stage_tasks if t.status == TaskStatus.FAILED])
                
                stage_progress.completed_tasks = completed
                stage_progress.failed_tasks = failed
                
                if stage_tasks:
                    # Calculate stage progress as weighted average
                    stage_progress.stage_progress = sum(t.progress for t in stage_tasks) / len(stage_tasks) * 100.0
                    
                    # Update current task
                    current_task = next((t for t in stage_tasks if t.status == TaskStatus.RUNNING), None)
                    if current_task:
                        stage_progress.current_task_id = current_task.id
                
                # Update overall workflow progress
                workflow.completed_tasks = sum(sp.completed_tasks for sp in workflow.stage_progress.values())
                workflow.failed_tasks = sum(sp.failed_tasks for sp in workflow.stage_progress.values())
    
    def get_workflow_progress(self, workflow_id: str) -> Optional[WorkflowProgress]:
        """Gets the current progress for a workflow.

        Args:
            workflow_id: The ID of the workflow.

        Returns:
            A WorkflowProgress object, or None if the workflow is not found.
        """
        return self.active_workflows.get(workflow_id)
    
    def get_task_progress(self, task_id: str) -> Optional[WorkflowTask]:
        """Gets the current progress for a task.

        Args:
            task_id: The ID of the task.

        Returns:
            A WorkflowTask object, or None if the task is not found.
        """
        return self.tasks.get(task_id)
    
    def get_all_workflows(self) -> Dict[str, WorkflowProgress]:
        """Gets all active workflows.

        Returns:
            A dictionary of all active workflows.
        """
        return self.active_workflows.copy()
    
    def get_workflow_summary(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Gets a summary of the progress for a workflow.

        Args:
            workflow_id: The ID of the workflow.

        Returns:
            A dictionary containing a summary of the workflow's progress, or None
            if the workflow is not found.
        """
        workflow = self.get_workflow_progress(workflow_id)
        if not workflow:
            return None
        
        # Calculate time metrics
        elapsed_time = (datetime.utcnow() - workflow.started_at).total_seconds()
        
        # Estimate remaining time
        if workflow.overall_progress > 0:
            estimated_total_time = elapsed_time / (workflow.overall_progress / 100.0)
            estimated_remaining = max(0, estimated_total_time - elapsed_time)
        else:
            estimated_remaining = 0
        
        summary = {
            'workflow_id': workflow.workflow_id,
            'project_name': workflow.project_name,
            'current_stage': workflow.current_stage.value,
            'status': workflow.status,
            'overall_progress': f"{workflow.overall_progress:.1f}%",
            'total_tasks': workflow.total_tasks,
            'completed_tasks': workflow.completed_tasks,
            'failed_tasks': workflow.failed_tasks,
            'success_rate': f"{(workflow.completed_tasks / max(1, workflow.completed_tasks + workflow.failed_tasks)) * 100:.1f}%",
            'elapsed_time': f"{elapsed_time:.0f}s",
            'estimated_remaining': f"{estimated_remaining:.0f}s",
            'started_at': workflow.started_at.strftime('%Y-%m-%d %H:%M:%S'),
            'last_updated': workflow.last_updated.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add stage summaries
        stage_summaries = {}
        for stage_name, stage_progress in workflow.stage_progress.items():
            stage_summaries[stage_name] = {
                'percentage_complete': f"{stage_progress.percentage_complete:.1f}%",
                'tasks_completed': stage_progress.completed_tasks,
                'tasks_total': stage_progress.total_tasks,
                'current_task': stage_progress.current_task_id
            }
        summary['stages'] = stage_summaries
        
        return summary
    
    def export_progress_report(self, workflow_id: str, file_path: str) -> bool:
        """Exports a detailed progress report for a workflow to a file.

        Args:
            workflow_id: The ID of the workflow to export.
            file_path: The path to the file to export the report to.

        Returns:
            True if the report was successfully exported, False otherwise.
        """
        try:
            workflow = self.get_workflow_progress(workflow_id)
            if not workflow:
                return False
            
            # Get all tasks for this workflow
            workflow_tasks = []
            for task in self.tasks.values():
                # Find which workflow this task belongs to
                for wf in self.active_workflows.values():
                    if wf.workflow_id == workflow_id:
                        # Check if task belongs to one of the workflow's stages
                        if task.stage.value in wf.stage_progress:
                            workflow_tasks.append(task)
                        break
            
            # Create detailed report
            report = {
                'workflow_summary': self.get_workflow_summary(workflow_id),
                'workflow_details': workflow.to_dict(),
                'tasks': [task.to_dict() for task in workflow_tasks],
                'generated_at': datetime.utcnow().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Progress report exported to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export progress report: {e}")
            return False
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Adds an event handler for workflow events.

        Args:
            event_type: The type of event to handle.
            handler: The handler function.
        """
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
            logger.info(f"Added event handler for {event_type}")
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def _trigger_event(self, event_type: str, data: Any):
        """Triggers event handlers for a given event.

        Args:
            event_type: The type of event to trigger.
            data: The data to pass to the event handlers.
        """
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
    
    def register_stage_handler(self, stage: WorkflowStage, handler: Callable):
        """Registers a handler for a specific workflow stage completion.

        Args:
            stage: The workflow stage to register the handler for.
            handler: The handler function.
        """
        self.stage_handlers[stage] = handler
        logger.info(f"Registered stage handler for {stage.value}")
    
    def get_stage_statistics(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Gets detailed statistics for the stages of a workflow.

        Args:
            workflow_id: The ID of the workflow.

        Returns:
            A dictionary of stage statistics, or None if the workflow is not found.
        """
        workflow = self.get_workflow_progress(workflow_id)
        if not workflow:
            return None
        
        stats = {}
        for stage_name, stage_progress in workflow.stage_progress.items():
            # Get tasks for this stage
            stage_tasks = [t for t in self.tasks.values() if t.stage.value == stage_name]
            
            if stage_tasks:
                total_duration = sum(t.actual_duration for t in stage_tasks if t.actual_duration > 0)
                avg_duration = total_duration / len([t for t in stage_tasks if t.actual_duration > 0]) if any(t.actual_duration > 0 for t in stage_tasks) else 0
                
                stats[stage_name] = {
                    'total_tasks': len(stage_tasks),
                    'completed_tasks': len([t for t in stage_tasks if t.status == TaskStatus.COMPLETED]),
                    'failed_tasks': len([t for t in stage_tasks if t.status == TaskStatus.FAILED]),
                    'running_tasks': len([t for t in stage_tasks if t.status == TaskStatus.RUNNING]),
                    'pending_tasks': len([t for t in stage_tasks if t.status == TaskStatus.PENDING]),
                    'average_duration': f"{avg_duration:.1f}s",
                    'total_duration': f"{total_duration:.1f}s",
                    'success_rate': f"{(len([t for t in stage_tasks if t.status == TaskStatus.COMPLETED]) / len(stage_tasks)) * 100:.1f}%"
                }
        
        return stats
    
    def cleanup_workflow(self, workflow_id: str):
        """Cleans up a completed workflow.

        Args:
            workflow_id: The ID of the workflow to clean up.
        """
        with self._lock:
            if workflow_id in self.active_workflows:
                workflow = self.active_workflows[workflow_id]
                
                # Remove tasks belonging to this workflow
                tasks_to_remove = []
                for task_id, task in self.tasks.items():
                    if task.stage.value in workflow.stage_progress:
                        tasks_to_remove.append(task_id)
                
                for task_id in tasks_to_remove:
                    del self.tasks[task_id]
                
                # Remove workflow
                del self.active_workflows[workflow_id]
                
                logger.info(f"Cleaned up workflow {workflow_id}")
    
    def get_current_tasks(self, workflow_id: str, stage: Optional[WorkflowStage] = None) -> List[WorkflowTask]:
        """Gets the currently running tasks for a workflow.

        Args:
            workflow_id: The ID of the workflow.
            stage: An optional workflow stage to filter by.

        Returns:
            A list of currently running tasks.
        """
        workflow = self.get_workflow_progress(workflow_id)
        if not workflow:
            return []
        
        current_tasks = []
        for task in self.tasks.values():
            if task.stage == stage or stage is None:
                # Check if this task belongs to the workflow
                if task.stage.value in workflow.stage_progress:
                    current_tasks.append(task)
        
        return [t for t in current_tasks if t.status == TaskStatus.RUNNING]
