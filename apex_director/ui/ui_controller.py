"""
APEX DIRECTOR UI Controller
Central orchestrator for the UI and workflow management system
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
import logging
from pathlib import Path
from dataclasses import dataclass

from .input_validator import InputValidator, ProcessedInput, ValidationResult
from .treatment_generator import TreatmentGenerator, VisualTreatment
from .storyboard import StoryboardCreator, Storyboard
from .progress_monitor import ProgressMonitor, WorkflowTask, WorkflowStage, TaskStatus, Priority
from .approval_gates import ApprovalGateSystem, ApprovalWorkflow, ApprovalType
from .error_handler import ErrorHandler, ErrorContext, ErrorSeverity, ErrorCategory
from .deliverable_packager import DeliverablePackager, PackageTemplate

from ..core.orchestrator import get_orchestrator
from ..core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class ProjectSession:
    """Complete project session data"""
    session_id: str
    project_name: str
    created_at: datetime
    current_stage: WorkflowStage
    processed_input: Optional[ProcessedInput] = None
    treatment: Optional[VisualTreatment] = None
    storyboard: Optional[Storyboard] = None
    workflow_progress_id: Optional[str] = None
    approval_workflow_id: Optional[str] = None
    package_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    status: str = "active"  # active, paused, completed, cancelled
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UIController:
    """Main UI Controller coordinating all components"""
    
    def __init__(self):
        # Initialize core components
        self.input_validator = InputValidator()
        self.treatment_generator = TreatmentGenerator()
        self.storyboard_creator = StoryboardCreator()
        self.progress_monitor = ProgressMonitor()
        self.approval_system = ApprovalGateSystem()
        self.error_handler = ErrorHandler()
        self.deliverable_packager = DeliverablePackager()
        
        # Session management
        self.sessions: Dict[str, ProjectSession] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'session_created': [],
            'session_completed': [],
            'stage_changed': [],
            'approval_required': [],
            'error_occurred': [],
            'package_ready': []
        }
        
        # Start progress monitoring
        self.progress_monitor.start_monitoring()
        
        # Setup error handlers
        self._setup_error_handlers()
        
        logger.info("UI Controller initialized")
    
    def _setup_error_handlers(self):
        """Setup error handling for the UI system"""
        # Add error handlers for different components
        self.error_handler.add_error_handler(ErrorCategory.WORKFLOW, self._handle_workflow_error)
        self.error_handler.add_error_handler(ErrorCategory.VALIDATION, self._handle_validation_error)
        self.error_handler.add_error_handler(ErrorCategory.GENERATION, self._handle_generation_error)
        
        # Add recovery handlers
        self.error_handler.recovery_handlers[ErrorCategory.WORKFLOW] = self._recover_workflow_error
    
    def _handle_workflow_error(self, error_record, original_error):
        """Handle workflow-related errors"""
        logger.error(f"Workflow error detected: {error_record.message}")
        # Could pause workflow or notify administrators
    
    def _handle_validation_error(self, error_record, original_error):
        """Handle validation errors"""
        logger.warning(f"Validation error: {error_record.message}")
        # Could provide user feedback
    
    def _handle_generation_error(self, error_record, original_error):
        """Handle generation errors"""
        logger.error(f"Generation error: {error_record.message}")
        # Could trigger fallback generation
    
    def _recover_workflow_error(self, error_id: str):
        """Recover from workflow errors"""
        error_details = self.error_handler.get_error_details(error_id)
        if error_details:
            # Implementation for workflow recovery
            pass
    
    # Session Management
    
    async def create_project_session(self, project_name: str, user_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new project session
        
        Args:
            project_name: Name of the project
            user_metadata: Optional user metadata
            
        Returns:
            Session ID
        """
        try:
            session_id = str(uuid.uuid4())
            
            session = ProjectSession(
                session_id=session_id,
                project_name=project_name,
                created_at=datetime.utcnow(),
                current_stage=WorkflowStage.INPUT_VALIDATION,
                metadata=user_metadata or {}
            )
            
            self.sessions[session_id] = session
            
            # Create workflow progress tracking
            workflow_id = self.progress_monitor.create_workflow(project_name)
            session.workflow_progress_id = workflow_id
            
            # Setup initial tasks
            await self._setup_initial_tasks(session_id)
            
            logger.info(f"Created project session {session_id} for project: {project_name}")
            self._trigger_event('session_created', session)
            
            return session_id
            
        except Exception as e:
            error_context = ErrorContext(
                component="ui_controller",
                operation="create_project_session",
                user_id=user_metadata.get('user_id') if user_metadata else None
            )
            self.error_handler.handle_error(e, error_context, ErrorSeverity.CRITICAL)
            raise
    
    async def _setup_initial_tasks(self, session_id: str):
        """Setup initial workflow tasks"""
        session = self.sessions[session_id]
        workflow_id = session.workflow_progress_id
        
        if not workflow_id:
            return
        
        # Add initial tasks to workflow
        self.progress_monitor.add_task(
            workflow_id=workflow_id,
            stage=WorkflowStage.INPUT_VALIDATION,
            name="Validate Input",
            description="Validate and process user input",
            priority=Priority.HIGH,
            estimated_duration=30.0
        )
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a project session"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Get workflow progress
        workflow_progress = None
        if session.workflow_progress_id:
            workflow_progress = self.progress_monitor.get_workflow_summary(session.workflow_progress_id)
        
        # Get approval status
        approval_status = None
        if session.approval_workflow_id:
            approval_status = self.approval_system.get_workflow_status(session.approval_workflow_id)
        
        return {
            'session_id': session.session_id,
            'project_name': session.project_name,
            'current_stage': session.current_stage.value,
            'status': session.status,
            'created_at': session.created_at.isoformat(),
            'workflow_progress': workflow_progress,
            'approval_status': approval_status,
            'treatment_id': session.treatment.id if session.treatment else None,
            'storyboard_id': session.storyboard.id if session.storyboard else None,
            'package_id': session.package_id,
            'metadata': session.metadata
        }
    
    # Input Processing
    
    async def validate_and_process_input(self, session_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and process user input
        
        Args:
            session_id: Session ID
            input_data: Raw input data
            
        Returns:
            Validation results and processed input
        """
        try:
            session = self.sessions[session_id]
            
            # Validate input
            is_valid, validation_results = await self.input_validator.validate_project_input(input_data)
            validation_summary = self.input_validator.get_validation_summary(validation_results)
            
            if not is_valid:
                return {
                    'valid': False,
                    'validation_summary': validation_summary,
                    'suggested_fixes': self.input_validator.suggest_fixes(validation_results),
                    'processed_input': None
                }
            
            # Process input
            processed_input = await self.input_validator.process_validated_input(input_data)
            session.processed_input = processed_input
            
            # Mark task as completed
            if session.workflow_progress_id:
                # Find validation task and complete it
                workflow_progress = self.progress_monitor.get_workflow_progress(session.workflow_progress_id)
                if workflow_progress:
                    for stage_progress in workflow_progress.stage_progress.values():
                        if stage_progress.stage == WorkflowStage.INPUT_VALIDATION:
                            # This would need to be implemented to find and complete specific tasks
                            break
            
            logger.info(f"Input validated and processed for session {session_id}")
            return {
                'valid': True,
                'validation_summary': validation_summary,
                'processed_input': {
                    'project_name': processed_input.project_name,
                    'audio_file': processed_input.audio_file,
                    'duration_seconds': processed_input.duration_seconds,
                    'visual_style': processed_input.visual_style,
                    'output_resolution': processed_input.output_resolution
                }
            }
            
        except Exception as e:
            error_context = ErrorContext(
                component="ui_controller",
                operation="validate_and_process_input",
                session_id=session_id,
                additional_data={'input_keys': list(input_data.keys())}
            )
            error_id = self.error_handler.handle_error(
                e, error_context, ErrorSeverity.ERROR, ErrorCategory.VALIDATION
            )
            raise ValueError(f"Input validation failed: {str(e)}")
    
    # Treatment Generation
    
    async def generate_creative_treatment(self, 
                                        session_id: str,
                                        custom_requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate creative treatment for the project
        
        Args:
            session_id: Session ID
            custom_requirements: Optional custom requirements
            
        Returns:
            Treatment details
        """
        try:
            session = self.sessions[session_id]
            
            if not session.processed_input:
                raise ValueError("Input must be validated before generating treatment")
            
            # Add task to workflow
            task_id = None
            if session.workflow_progress_id:
                task_id = self.progress_monitor.add_task(
                    workflow_id=session.workflow_progress_id,
                    stage=WorkflowStage.TREATMENT_GENERATION,
                    name="Generate Treatment",
                    description="Create visual treatment based on input",
                    priority=Priority.HIGH,
                    estimated_duration=60.0
                )
                self.progress_monitor.start_task(task_id)
            
            # Generate treatment
            treatment = await self.treatment_generator.generate_treatment(
                session.processed_input, custom_requirements
            )
            
            session.treatment = treatment
            session.current_stage = WorkflowStage.TREATMENT_GENERATION
            
            # Complete task
            if task_id:
                self.progress_monitor.complete_task(task_id, {
                    'treatment_id': treatment.id,
                    'scene_count': len(treatment.scenes),
                    'treatment_type': treatment.treatment_type.value
                })
            
            # Update progress
            self._update_stage_progress(session_id, WorkflowStage.TREATMENT_GENERATION)
            
            logger.info(f"Treatment generated for session {session_id}: {len(treatment.scenes)} scenes")
            return {
                'treatment_id': treatment.id,
                'treatment_type': treatment.treatment_type.value,
                'visual_complexity': treatment.visual_complexity.value,
                'scene_count': len(treatment.scenes),
                'duration_seconds': treatment.audio_duration,
                'style_keywords': treatment.style_keywords,
                'color_scheme': treatment.color_scheme.get('scheme_name', 'Custom'),
                'overall_concept': treatment.overall_concept
            }
            
        except Exception as e:
            error_context = ErrorContext(
                component="ui_controller",
                operation="generate_creative_treatment",
                session_id=session_id,
                workflow_id=session.workflow_progress_id
            )
            error_id = self.error_handler.handle_error(
                e, error_context, ErrorSeverity.ERROR, ErrorCategory.GENERATION
            )
            
            # Mark task as failed if it exists
            if session.workflow_progress_id and 'task_id' in locals():
                self.progress_monitor.fail_task(task_id, str(e))
            
            raise ValueError(f"Treatment generation failed: {str(e)}")
    
    # Storyboard Creation
    
    async def create_storyboard(self, session_id: str, custom_shots: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Create storyboard based on treatment
        
        Args:
            session_id: Session ID
            custom_shots: Optional custom shot specifications
            
        Returns:
            Storyboard details
        """
        try:
            session = self.sessions[session_id]
            
            if not session.treatment:
                raise ValueError("Treatment must be generated before creating storyboard")
            
            # Add task to workflow
            task_id = None
            if session.workflow_progress_id:
                task_id = self.progress_monitor.add_task(
                    workflow_id=session.workflow_progress_id,
                    stage=WorkflowStage.STORYBOARD_CREATION,
                    name="Create Storyboard",
                    description="Generate detailed storyboard from treatment",
                    priority=Priority.HIGH,
                    estimated_duration=120.0
                )
                self.progress_monitor.start_task(task_id)
            
            # Create storyboard
            storyboard = await self.storyboard_creator.create_storyboard(session.treatment, custom_shots)
            
            session.storyboard = storyboard
            session.current_stage = WorkflowStage.STORYBOARD_CREATION
            
            # Complete task
            if task_id:
                total_shots = sum(len(scene.shots) for scene in storyboard.scenes)
                self.progress_monitor.complete_task(task_id, {
                    'storyboard_id': storyboard.id,
                    'scene_count': len(storyboard.scenes),
                    'total_shots': total_shots
                })
            
            # Update progress
            self._update_stage_progress(session_id, WorkflowStage.STORYBOARD_CREATION)
            
            logger.info(f"Storyboard created for session {session_id}: {len(storyboard.scenes)} scenes, {total_shots} shots")
            return {
                'storyboard_id': storyboard.id,
                'scene_count': len(storyboard.scenes),
                'total_shots': total_shots,
                'duration_seconds': storyboard.total_duration,
                'shot_type_distribution': self._get_shot_type_distribution(storyboard),
                'scene_type_distribution': self._get_scene_type_distribution(storyboard)
            }
            
        except Exception as e:
            error_context = ErrorContext(
                component="ui_controller",
                operation="create_storyboard",
                session_id=session_id,
                workflow_id=session.workflow_progress_id
            )
            error_id = self.error_handler.handle_error(
                e, error_context, ErrorSeverity.ERROR, ErrorCategory.GENERATION
            )
            
            if session.workflow_progress_id and 'task_id' in locals():
                self.progress_monitor.fail_task(task_id, str(e))
            
            raise ValueError(f"Storyboard creation failed: {str(e)}")
    
    def _get_shot_type_distribution(self, storyboard: Storyboard) -> Dict[str, int]:
        """Get distribution of shot types in storyboard"""
        distribution = {}
        for scene in storyboard.scenes:
            for shot in scene.shots:
                shot_type = shot.shot_type.value
                distribution[shot_type] = distribution.get(shot_type, 0) + 1
        return distribution
    
    def _get_scene_type_distribution(self, storyboard: Storyboard) -> Dict[str, int]:
        """Get distribution of scene types in storyboard"""
        distribution = {}
        for scene in storyboard.scenes:
            scene_type = scene.scene_type
            distribution[scene_type] = distribution.get(scene_type, 0) + 1
        return distribution
    
    # Approval Workflow
    
    async def setup_approval_workflow(self, session_id: str, custom_gates: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Setup approval workflow for the project
        
        Args:
            session_id: Session ID
            custom_gates: Optional custom gate configurations
            
        Returns:
            Approval workflow ID
        """
        try:
            session = self.sessions[session_id]
            
            if not session.treatment or not session.storyboard:
                raise ValueError("Treatment and storyboard must be created before setting up approval workflow")
            
            workflow_id = self.approval_system.create_approval_workflow(
                session.project_name, custom_gates
            )
            
            session.approval_workflow_id = workflow_id
            session.current_stage = WorkflowStage.IMAGE_GENERATION  # Wait for image generation
            
            logger.info(f"Approval workflow created for session {session_id}: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            error_context = ErrorContext(
                component="ui_controller",
                operation="setup_approval_workflow",
                session_id=session_id
            )
            self.error_handler.handle_error(e, error_context, ErrorSeverity.ERROR)
            raise ValueError(f"Failed to setup approval workflow: {str(e)}")
    
    async def get_approval_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get approval status for the session"""
        session = self.sessions.get(session_id)
        if not session or not session.approval_workflow_id:
            return None
        
        return self.approval_system.get_workflow_status(session.approval_workflow_id)
    
    # Progress Tracking
    
    def _update_stage_progress(self, session_id: str, stage: WorkflowStage):
        """Update progress for a specific stage"""
        session = self.sessions.get(session_id)
        if not session or not session.workflow_progress_id:
            return
        
        session.current_stage = stage
        self._trigger_event('stage_changed', {
            'session_id': session_id,
            'stage': stage.value
        })
    
    async def get_workflow_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow progress"""
        session = self.sessions.get(session_id)
        if not session or not session.workflow_progress_id:
            return None
        
        return self.progress_monitor.get_workflow_summary(session.workflow_progress_id)
    
    # Package Creation
    
    async def create_deliverable_package(self, session_id: str, template_name: str = 'client_delivery') -> str:
        """
        Create final deliverable package
        
        Args:
            session_id: Session ID
            template_name: Package template to use
            
        Returns:
            Package ID
        """
        try:
            session = self.sessions[session_id]
            
            if not session.treatment or not session.storyboard:
                raise ValueError("Treatment and storyboard must be completed before creating package")
            
            # Add final packaging task
            task_id = None
            if session.workflow_progress_id:
                task_id = self.progress_monitor.add_task(
                    workflow_id=session.workflow_progress_id,
                    stage=WorkflowStage.DELIVERY,
                    name="Create Package",
                    description="Package final deliverables",
                    priority=Priority.HIGH,
                    estimated_duration=60.0
                )
                self.progress_monitor.start_task(task_id)
            
            # Create package
            package_id = await self.deliverable_packager.create_package(
                session.project_name,
                session.treatment,
                session.storyboard,
                template_name
            )
            
            session.package_id = package_id
            session.current_stage = WorkflowStage.DELIVERY
            
            # Complete task
            if task_id:
                self.progress_monitor.complete_task(task_id, {
                    'package_id': package_id,
                    'template': template_name
                })
            
            # Update session status
            session.status = "completed"
            session.current_stage = WorkflowStage.COMPLETE
            
            logger.info(f"Deliverable package created for session {session_id}: {package_id}")
            self._trigger_event('package_ready', {'session_id': session_id, 'package_id': package_id})
            
            return package_id
            
        except Exception as e:
            error_context = ErrorContext(
                component="ui_controller",
                operation="create_deliverable_package",
                session_id=session_id,
                workflow_id=session.workflow_progress_id
            )
            self.error_handler.handle_error(e, error_context, ErrorSeverity.ERROR)
            
            if session.workflow_progress_id and 'task_id' in locals():
                self.progress_monitor.fail_task(task_id, str(e))
            
            raise ValueError(f"Package creation failed: {str(e)}")
    
    async def get_package_info(self, package_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a deliverable package"""
        return self.deliverable_packager.get_package_info(package_id)
    
    # Error Handling
    
    async def handle_user_error(self, session_id: str, error_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Handle user-reported errors"""
        error_context = ErrorContext(
            component="user_interface",
            operation="user_reported",
            session_id=session_id,
            additional_data=context or {}
        )
        
        error_id = self.error_handler.handle_error(
            ValueError(error_message), 
            error_context, 
            ErrorSeverity.WARNING,
            ErrorCategory.USER_INPUT
        )
        
        self._trigger_event('error_occurred', {
            'session_id': session_id,
            'error_id': error_id,
            'message': error_message
        })
        
        return error_id
    
    # Session Management
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause a project session"""
        session = self.sessions.get(session_id)
        if session:
            session.status = "paused"
            logger.info(f"Paused session {session_id}")
            return True
        return False
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused project session"""
        session = self.sessions.get(session_id)
        if session and session.status == "paused":
            session.status = "active"
            logger.info(f"Resumed session {session_id}")
            return True
        return False
    
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel a project session"""
        session = self.sessions.get(session_id)
        if session:
            session.status = "cancelled"
            
            # Cleanup associated resources
            if session.workflow_progress_id:
                self.progress_monitor.cleanup_workflow(session.workflow_progress_id)
            
            if session.package_id:
                self.deliverable_packager.cleanup_package(session.package_id)
            
            logger.info(f"Cancelled session {session_id}")
            return True
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all project sessions"""
        return [
            {
                'session_id': session.session_id,
                'project_name': session.project_name,
                'status': session.status,
                'current_stage': session.current_stage.value,
                'created_at': session.created_at.isoformat(),
                'has_treatment': session.treatment is not None,
                'has_storyboard': session.storyboard is not None,
                'has_package': session.package_id is not None
            }
            for session in self.sessions.values()
        ]
    
    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed session information"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        details = {
            'session_id': session.session_id,
            'project_name': session.project_name,
            'status': session.status,
            'current_stage': session.current_stage.value,
            'created_at': session.created_at.isoformat(),
            'metadata': session.metadata
        }
        
        # Add component-specific details
        if session.processed_input:
            details['processed_input'] = {
                'project_name': session.processed_input.project_name,
                'visual_style': session.processed_input.visual_style,
                'duration_seconds': session.processed_input.duration_seconds,
                'output_resolution': session.processed_input.output_resolution
            }
        
        if session.treatment:
            details['treatment'] = {
                'id': session.treatment.id,
                'type': session.treatment.treatment_type.value,
                'scenes': len(session.treatment.scenes)
            }
        
        if session.storyboard:
            details['storyboard'] = {
                'id': session.storyboard.id,
                'scenes': len(session.storyboard.scenes),
                'total_shots': sum(len(scene.shots) for scene in session.storyboard.scenes)
            }
        
        return details
    
    # Event Handling
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler for UI events"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
            logger.info(f"Added event handler for {event_type}")
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def _trigger_event(self, event_type: str, data: Any):
        """Trigger UI events"""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
    
    # System Status
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'active_sessions': len([s for s in self.sessions.values() if s.status == 'active']),
            'paused_sessions': len([s for s in self.sessions.values() if s.status == 'paused']),
            'completed_sessions': len([s for s in self.sessions.values() if s.status == 'completed']),
            'total_sessions': len(self.sessions),
            'error_statistics': self.error_handler.get_error_statistics(),
            'packages_created': len(self.deliverable_packager.packages),
            'progress_monitor_active': self.progress_monitor._is_monitoring
        }
    
    def export_session_report(self, session_id: str, file_path: str) -> bool:
        """Export comprehensive session report"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            report = {
                'session_info': self.get_session_details(session_id),
                'workflow_progress': self.get_workflow_progress(session_id),
                'approval_status': self.get_approval_status(session_id),
                'package_info': self.get_package_info(session.package_id) if session.package_id else None,
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Add error information if any
            if session.workflow_progress_id:
                workflow_errors = self.error_handler.get_workflow_errors(session.workflow_progress_id)
                if workflow_errors:
                    report['errors'] = workflow_errors
            
            with open(file_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Session report exported to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export session report: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop progress monitoring
            self.progress_monitor.stop_monitoring()
            
            # Cleanup packages
            for package_id in list(self.deliverable_packager.packages.keys()):
                self.deliverable_packager.cleanup_package(package_id)
            
            logger.info("UI Controller cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global UI Controller instance
_ui_controller: Optional[UIController] = None


def get_ui_controller() -> UIController:
    """Get the global UI Controller instance"""
    global _ui_controller
    if _ui_controller is None:
        _ui_controller = UIController()
    return _ui_controller


# Convenience functions
async def create_project_session(project_name: str, user_metadata: Optional[Dict[str, Any]] = None) -> str:
    """Create a new project session"""
    controller = get_ui_controller()
    return await controller.create_project_session(project_name, user_metadata)


async def validate_input(session_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and process input"""
    controller = get_ui_controller()
    return await controller.validate_and_process_input(session_id, input_data)


async def generate_treatment(session_id: str, custom_requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate creative treatment"""
    controller = get_ui_controller()
    return await controller.generate_creative_treatment(session_id, custom_requirements)


async def create_storyboard(session_id: str, custom_shots: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Create storyboard"""
    controller = get_ui_controller()
    return await controller.create_storyboard(session_id, custom_shots)


async def create_package(session_id: str, template_name: str = 'client_delivery') -> str:
    """Create deliverable package"""
    controller = get_ui_controller()
    return await controller.create_deliverable_package(session_id, template_name)


def get_system_status() -> Dict[str, Any]:
    """Get system status"""
    controller = get_ui_controller()
    return controller.get_system_status()
