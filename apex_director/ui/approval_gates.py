"""
APEX DIRECTOR Approval Gate System
Manages approval workflow and checkpoints throughout the music video generation process
"""

import uuid
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading

from .treatment_generator import VisualTreatment
from .storyboard import Storyboard

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Approval status options"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    SKIPPED = "skipped"


class ApprovalType(Enum):
    """Types of approvals"""
    INITIAL_CONCEPT = "initial_concept"
    TREATMENT_REVIEW = "treatment_review"
    STORYBOARD_APPROVAL = "storyboard_approval"
    PREVIEW_APPROVAL = "preview_approval"
    FINAL_APPROVAL = "final_approval"
    STYLE_BOARD = "style_board"
    BUDGET_APPROVAL = "budget_approval"
    SCHEDULE_APPROVAL = "schedule_approval"


class ApprovalPriority(Enum):
    """Approval priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


@dataclass
class ApprovalGate:
    """Represents an individual approval gate in a workflow.

    Attributes:
        id: The unique identifier for the approval gate.
        approval_type: The type of approval.
        name: The name of the approval gate.
        description: A description of the approval gate.
        status: The current status of the approval gate.
        priority: The priority of the approval gate.
        created_at: The timestamp when the approval gate was created.
        assigned_to: The person or team assigned to the approval gate.
        deadline: The deadline for the approval gate.
        submitted_at: The timestamp when the approval gate was submitted.
        reviewed_at: The timestamp when the approval gate was reviewed.
        reviewer_comments: Comments from the reviewer.
        submitter_comments: Comments from the submitter.
        attachments: A list of file paths for attachments.
        dependencies: A list of approval gate IDs that must be completed before this one.
        metadata: A dictionary of additional metadata.
    """
    id: str
    approval_type: ApprovalType
    name: str
    description: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    priority: ApprovalPriority = ApprovalPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    assigned_to: Optional[str] = None
    deadline: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    reviewer_comments: Optional[str] = None
    submitter_comments: Optional[str] = None
    attachments: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the ApprovalGate to a dictionary.

        Returns:
            A dictionary representation of the ApprovalGate.
        """
        data = asdict(self)
        data['approval_type'] = self.approval_type.value
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        if self.deadline:
            data['deadline'] = self.deadline.isoformat()
        if self.submitted_at:
            data['submitted_at'] = self.submitted_at.isoformat()
        if self.reviewed_at:
            data['reviewed_at'] = self.reviewed_at.isoformat()
        return data
    
    @property
    def is_approved(self) -> bool:
        """Checks if the approval gate is approved.

        Returns:
            True if the approval gate is approved, False otherwise.
        """
        return self.status == ApprovalStatus.APPROVED
    
    @property
    def is_rejected(self) -> bool:
        """Checks if the approval gate is rejected.

        Returns:
            True if the approval gate is rejected, False otherwise.
        """
        return self.status == ApprovalStatus.REJECTED
    
    @property
    def is_pending(self) -> bool:
        """Checks if the approval gate is pending.

        Returns:
            True if the approval gate is pending, False otherwise.
        """
        return self.status == ApprovalStatus.PENDING
    
    @property
    def is_overdue(self) -> bool:
        """Checks if the approval gate is overdue.

        Returns:
            True if the approval gate is overdue, False otherwise.
        """
        return (self.deadline and 
                self.status == ApprovalStatus.PENDING and 
                datetime.utcnow() > self.deadline)
    
    @property
    def days_until_deadline(self) -> Optional[int]:
        """Gets the number of days until the deadline.

        Returns:
            The number of days until the deadline, or None if there is no deadline.
        """
        if not self.deadline:
            return None
        delta = self.deadline - datetime.utcnow()
        return delta.days


@dataclass
class ApprovalWorkflow:
    """Represents a complete approval workflow for a project.

    Attributes:
        id: The unique identifier for the approval workflow.
        project_name: The name of the project.
        gates: A list of approval gates in the workflow.
        created_at: The timestamp when the workflow was created.
        started_at: The timestamp when the workflow was started.
        completed_at: The timestamp when the workflow was completed.
        overall_status: The overall status of the workflow.
        metadata: A dictionary of additional metadata.
    """
    id: str
    project_name: str
    gates: List[ApprovalGate]
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    overall_status: ApprovalStatus = ApprovalStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the ApprovalWorkflow to a dictionary.

        Returns:
            A dictionary representation of the ApprovalWorkflow.
        """
        data = asdict(self)
        data['overall_status'] = self.overall_status.value
        data['gates'] = [gate.to_dict() for gate in self.gates]
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data
    
    @property
    def approved_gates(self) -> List[ApprovalGate]:
        """Gets the approved gates in the workflow.

        Returns:
            A list of approved gates.
        """
        return [gate for gate in self.gates if gate.status == ApprovalStatus.APPROVED]
    
    @property
    def pending_gates(self) -> List[ApprovalGate]:
        """Gets the pending gates in the workflow.

        Returns:
            A list of pending gates.
        """
        return [gate for gate in self.gates if gate.status == ApprovalStatus.PENDING]
    
    @property
    def rejected_gates(self) -> List[ApprovalGate]:
        """Gets the rejected gates in the workflow.

        Returns:
            A list of rejected gates.
        """
        return [gate for gate in self.gates if gate.status == ApprovalStatus.REJECTED]
    
    @property
    def next_gate(self) -> Optional[ApprovalGate]:
        """Gets the next gate that needs approval.

        Returns:
            The next gate that needs approval, or None if there are no pending gates.
        """
        # Find first pending gate whose dependencies are all approved
        for gate in self.gates:
            if gate.status == ApprovalStatus.PENDING:
                if all(dep_id in [g.id for g in self.approved_gates] for dep_id in gate.dependencies):
                    return gate
        return None
    
    @property
    def is_completed(self) -> bool:
        """Checks if the workflow is completed.

        Returns:
            True if the workflow is completed, False otherwise.
        """
        return all(gate.status in [ApprovalStatus.APPROVED, ApprovalStatus.SKIPPED] for gate in self.gates)
    
    @property
    def progress_percentage(self) -> float:
        """Gets the progress percentage of the workflow.

        Returns:
            The progress percentage of the workflow.
        """
        if not self.gates:
            return 0.0
        approved_count = len(self.approved_gates)
        total_count = len([g for g in self.gates if g.status != ApprovalStatus.SKIPPED])
        return (approved_count / total_count) * 100.0 if total_count > 0 else 0.0


class ApprovalGateSystem:
    """A comprehensive system for managing approval gates and workflows.

    This class provides functionality for creating, managing, and tracking
    approval workflows for projects. It includes features such as:
    - Creating and managing approval gates and workflows
    - Submitting, approving, rejecting, and skipping gates
    - Requesting revisions for gates
    - Tracking the status of gates and workflows
    - Exporting workflow reports
    - Event handling for approval events
    - Assigning gates to reviewers and extending deadlines
    - Calculating workflow statistics
    """
    
    def __init__(self):
        """Initializes the ApprovalGateSystem."""
        self.workflows: Dict[str, ApprovalWorkflow] = {}
        self.gates: Dict[str, ApprovalGate] = {}
        self.event_handlers: Dict[str, List[Callable]] = {
            'gate_submitted': [],
            'gate_approved': [],
            'gate_rejected': [],
            'gate_revision_requested': [],
            'workflow_completed': []
        }
        self._lock = threading.Lock()
        
        # Auto-approval rules
        self.auto_approval_rules = self._initialize_auto_approval_rules()
        
        logger.info("Approval Gate System initialized")
    
    def _initialize_auto_approval_rules(self) -> Dict[ApprovalType, Dict[str, Any]]:
        """Initializes the auto-approval rules.

        Returns:
            A dictionary of auto-approval rules.
        """
        return {
            ApprovalType.INITIAL_CONCEPT: {
                'auto_approve': False,
                'timeout_hours': 48,
                'reminder_hours': [24, 4]
            },
            ApprovalType.TREATMENT_REVIEW: {
                'auto_approve': False,
                'timeout_hours': 72,
                'reminder_hours': [48, 24, 4]
            },
            ApprovalType.STORYBOARD_APPROVAL: {
                'auto_approve': False,
                'timeout_hours': 96,
                'reminder_hours': [72, 48, 24, 4]
            },
            ApprovalType.PREVIEW_APPROVAL: {
                'auto_approve': False,
                'timeout_hours': 48,
                'reminder_hours': [24, 4]
            },
            ApprovalType.FINAL_APPROVAL: {
                'auto_approve': False,
                'timeout_hours': 24,
                'reminder_hours': [12, 2]
            }
        }
    
    def create_approval_workflow(self, 
                               project_name: str,
                               gate_configs: Optional[List[Dict[str, Any]]] = None) -> str:
        """Creates a new approval workflow for a project.

        Args:
            project_name: The name of the project.
            gate_configs: Optional custom gate configurations.

        Returns:
            The ID of the newly created workflow.
        """
        workflow_id = str(uuid.uuid4())
        
        # Create default gates if none specified
        if gate_configs is None:
            gate_configs = self._get_default_gate_configs()
        
        gates = []
        for config in gate_configs:
            gate = ApprovalGate(
                id=str(uuid.uuid4()),
                approval_type=ApprovalType(config['type']),
                name=config['name'],
                description=config['description'],
                priority=ApprovalPriority(config.get('priority', 2)),
                deadline=self._calculate_deadline(ApprovalType(config['type'])),
                dependencies=config.get('dependencies', [])
            )
            gates.append(gate)
            self.gates[gate.id] = gate
        
        workflow = ApprovalWorkflow(
            id=workflow_id,
            project_name=project_name,
            gates=gates
        )
        
        with self._lock:
            self.workflows[workflow_id] = workflow
        
        logger.info(f"Created approval workflow {workflow_id} for project: {project_name}")
        return workflow_id
    
    def _get_default_gate_configs(self) -> List[Dict[str, Any]]:
        """Gets the default gate configurations.

        Returns:
            A list of default gate configurations.
        """
        return [
            {
                'type': ApprovalType.INITIAL_CONCEPT.value,
                'name': 'Initial Concept Approval',
                'description': 'Review and approve the initial creative concept',
                'priority': 3,
                'dependencies': []
            },
            {
                'type': ApprovalType.STYLE_BOARD.value,
                'name': 'Style Board Approval',
                'description': 'Approve visual style and aesthetic direction',
                'priority': 3,
                'dependencies': []
            },
            {
                'type': ApprovalType.TREATMENT_REVIEW.value,
                'name': 'Treatment Review',
                'description': 'Review and approve the creative treatment',
                'priority': 3,
                'dependencies': []
            },
            {
                'type': ApprovalType.STORYBOARD_APPROVAL.value,
                'name': 'Storyboard Approval',
                'description': 'Approve detailed storyboard and shot plan',
                'priority': 3,
                'dependencies': []
            },
            {
                'type': ApprovalType.PREVIEW_APPROVAL.value,
                'name': 'Preview Approval',
                'description': 'Approve preview/preliminary video version',
                'priority': 3,
                'dependencies': []
            },
            {
                'type': ApprovalType.FINAL_APPROVAL.value,
                'name': 'Final Approval',
                'description': 'Final approval for completed video',
                'priority': 4,
                'dependencies': []
            }
        ]
    
    def _calculate_deadline(self, approval_type: ApprovalType, hours: Optional[int] = None) -> datetime:
        """Calculates the deadline for an approval gate.

        Args:
            approval_type: The type of approval.
            hours: The number of hours until the deadline.

        Returns:
            The deadline for the approval gate.
        """
        if hours is None:
            rule = self.auto_approval_rules.get(approval_type, {})
            hours = rule.get('timeout_hours', 72)  # Default 3 days
        
        return datetime.utcnow() + timedelta(hours=hours)
    
    def submit_for_approval(self, 
                           gate_id: str,
                           submitter: str,
                           comments: Optional[str] = None,
                           attachments: Optional[List[str]] = None) -> bool:
        """Submits a gate for approval.

        Args:
            gate_id: The ID of the gate to submit.
            submitter: The person submitting the gate.
            comments: Optional submission comments.
            attachments: Optional list of attachment paths.

        Returns:
            True if the gate was successfully submitted, False otherwise.
        """
        with self._lock:
            if gate_id not in self.gates:
                logger.warning(f"Gate {gate_id} not found")
                return False
            
            gate = self.gates[gate_id]
            if gate.status != ApprovalStatus.PENDING:
                logger.warning(f"Gate {gate_id} already has status: {gate.status}")
                return False
            
            gate.status = ApprovalStatus.PENDING  # Still pending but now submitted
            gate.submitted_at = datetime.utcnow()
            gate.submitter_comments = comments
            gate.attachments = attachments or []
            
            logger.info(f"Submitted gate {gate_id} for approval by {submitter}")
            self._trigger_event('gate_submitted', gate)
            return True
    
    def approve_gate(self,
                    gate_id: str,
                    reviewer: str,
                    comments: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Approves a gate.

        Args:
            gate_id: The ID of the gate to approve.
            reviewer: The person approving the gate.
            comments: Optional approval comments.
            metadata: Optional additional metadata.

        Returns:
            True if the gate was successfully approved, False otherwise.
        """
        with self._lock:
            if gate_id not in self.gates:
                logger.warning(f"Gate {gate_id} not found")
                return False
            
            gate = self.gates[gate_id]
            gate.status = ApprovalStatus.APPROVED
            gate.reviewed_at = datetime.utcnow()
            gate.reviewer_comments = comments
            gate.metadata.update(metadata or {})
            
            # Update workflow status
            self._update_workflow_status(gate)
            
            logger.info(f"Approved gate {gate_id} by {reviewer}")
            self._trigger_event('gate_approved', gate)
            return True
    
    def reject_gate(self,
                   gate_id: str,
                   reviewer: str,
                   reason: str,
                   comments: Optional[str] = None) -> bool:
        """Rejects a gate.

        Args:
            gate_id: The ID of the gate to reject.
            reviewer: The person rejecting the gate.
            reason: The reason for rejection.
            comments: Optional additional comments.

        Returns:
            True if the gate was successfully rejected, False otherwise.
        """
        with self._lock:
            if gate_id not in self.gates:
                logger.warning(f"Gate {gate_id} not found")
                return False
            
            gate = self.gates[gate_id]
            gate.status = ApprovalStatus.REJECTED
            gate.reviewed_at = datetime.utcnow()
            gate.reviewer_comments = f"Rejected: {reason}\\n{comments or ''}"
            
            # Update workflow status
            self._update_workflow_status(gate)
            
            logger.info(f"Rejected gate {gate_id} by {reviewer}: {reason}")
            self._trigger_event('gate_rejected', gate)
            return True
    
    def request_revision(self,
                        gate_id: str,
                        reviewer: str,
                        revision_notes: str,
                        new_deadline: Optional[datetime] = None) -> bool:
        """Requests a revision for a gate.

        Args:
            gate_id: The ID of the gate to request a revision for.
            reviewer: The person requesting the revision.
            revision_notes: Notes about the required revisions.
            new_deadline: An optional new deadline for the revision.

        Returns:
            True if the revision was successfully requested, False otherwise.
        """
        with self._lock:
            if gate_id not in self.gates:
                logger.warning(f"Gate {gate_id} not found")
                return False
            
            gate = self.gates[gate_id]
            gate.status = ApprovalStatus.NEEDS_REVISION
            gate.reviewed_at = datetime.utcnow()
            gate.reviewer_comments = f"Revision requested: {revision_notes}"
            gate.metadata['revision_count'] = gate.metadata.get('revision_count', 0) + 1
            
            if new_deadline:
                gate.deadline = new_deadline
            else:
                # Extend deadline by 48 hours
                gate.deadline = datetime.utcnow() + timedelta(hours=48)
            
            logger.info(f"Requested revision for gate {gate_id} by {reviewer}")
            self._trigger_event('gate_revision_requested', gate)
            return True
    
    def skip_gate(self,
                 gate_id: str,
                 reason: str,
                 skipper: str) -> bool:
        """Skips a gate.

        This can be used for time constraints or testing purposes.

        Args:
            gate_id: The ID of the gate to skip.
            reason: The reason for skipping the gate.
            skipper: The person skipping the gate.

        Returns:
            True if the gate was successfully skipped, False otherwise.
        """
        with self._lock:
            if gate_id not in self.gates:
                logger.warning(f"Gate {gate_id} not found")
                return False
            
            gate = self.gates[gate_id]
            gate.status = ApprovalStatus.SKIPPED
            gate.reviewed_at = datetime.utcnow()
            gate.metadata['skipped_by'] = skipper
            gate.metadata['skip_reason'] = reason
            
            # Update workflow status
            self._update_workflow_status(gate)
            
            logger.info(f"Skipped gate {gate_id} by {skipper}: {reason}")
            return True
    
    def _update_workflow_status(self, gate: ApprovalGate):
        """Updates the workflow status based on gate changes.

        Args:
            gate: The gate that was changed.
        """
        # Find workflow containing this gate
        for workflow in self.workflows.values():
            if gate.id in [g.id for g in workflow.gates]:
                workflow.started_at = workflow.started_at or datetime.utcnow()
                
                # Check if workflow is completed
                if workflow.is_completed:
                    workflow.completed_at = datetime.utcnow()
                    workflow.overall_status = ApprovalStatus.APPROVED
                    self._trigger_event('workflow_completed', workflow)
                
                break
    
    def get_gate_status(self, gate_id: str) -> Optional[Dict[str, Any]]:
        """Gets the status of a specific gate.

        Args:
            gate_id: The ID of the gate.

        Returns:
            A dictionary with the gate's status, or None if the gate is not found.
        """
        gate = self.gates.get(gate_id)
        if not gate:
            return None
        
        return {
            'gate_id': gate.id,
            'type': gate.approval_type.value,
            'name': gate.name,
            'status': gate.status.value,
            'priority': gate.priority.value,
            'is_overdue': gate.is_overdue,
            'days_until_deadline': gate.days_until_deadline,
            'created_at': gate.created_at.isoformat(),
            'deadline': gate.deadline.isoformat() if gate.deadline else None,
            'submitted_at': gate.submitted_at.isoformat() if gate.submitted_at else None,
            'reviewed_at': gate.reviewed_at.isoformat() if gate.reviewed_at else None,
            'reviewer_comments': gate.reviewer_comments,
            'submitter_comments': gate.submitter_comments
        }
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Gets the status of a specific workflow.

        Args:
            workflow_id: The ID of the workflow.

        Returns:
            A dictionary with the workflow's status, or None if the workflow is not found.
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            'workflow_id': workflow.id,
            'project_name': workflow.project_name,
            'status': workflow.overall_status.value,
            'progress_percentage': f"{workflow.progress_percentage:.1f}%",
            'gates_total': len(workflow.gates),
            'gates_approved': len(workflow.approved_gates),
            'gates_pending': len(workflow.pending_gates),
            'gates_rejected': len(workflow.rejected_gates),
            'next_gate': workflow.next_gate.name if workflow.next_gate else None,
            'is_completed': workflow.is_completed,
            'created_at': workflow.created_at.isoformat(),
            'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
            'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None
        }
    
    def get_pending_gates(self, reviewer: Optional[str] = None) -> List[Dict[str, Any]]:
        """Gets all pending gates.

        Args:
            reviewer: An optional reviewer to filter by.

        Returns:
            A list of pending gates.
        """
        pending_gates = []
        
        for gate in self.gates.values():
            if gate.status == ApprovalStatus.PENDING:
                if reviewer is None or gate.assigned_to == reviewer:
                    pending_gates.append(self.get_gate_status(gate.id))
        
        return pending_gates
    
    def get_overdue_gates(self) -> List[Dict[str, Any]]:
        """Gets all overdue gates.

        Returns:
            A list of overdue gates.
        """
        overdue_gates = []
        
        for gate in self.gates.values():
            if gate.is_overdue:
                overdue_gates.append(self.get_gate_status(gate.id))
        
        return overdue_gates
    
    def export_workflow_report(self, workflow_id: str, file_path: str) -> bool:
        """Exports a detailed workflow report to a file.

        Args:
            workflow_id: The ID of the workflow to export.
            file_path: The path to the file to export the report to.

        Returns:
            True if the report was successfully exported, False otherwise.
        """
        try:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                return False
            
            # Get gate statuses
            gate_statuses = [self.get_gate_status(gate.id) for gate in workflow.gates]
            
            # Create report
            report = {
                'workflow_summary': self.get_workflow_status(workflow_id),
                'gates': gate_statuses,
                'exported_at': datetime.utcnow().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Workflow report exported to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export workflow report: {e}")
            return False
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Adds an event handler for approval events.

        Args:
            event_type: The type of event to handle.
            handler: The event handler function.
        """
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
            logger.info(f"Added event handler for {event_type}")
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def _trigger_event(self, event_type: str, data: Any):
        """Triggers event handlers for a given event type.

        Args:
            event_type: The type of event to trigger.
            data: The data to pass to the event handlers.
        """
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error in approval event handler for {event_type}: {e}")
    
    def set_gate_assignee(self, gate_id: str, assignee: str) -> bool:
        """Assigns a gate to a specific reviewer.

        Args:
            gate_id: The ID of the gate to assign.
            assignee: The person or team to assign the gate to.

        Returns:
            True if the gate was successfully assigned, False otherwise.
        """
        with self._lock:
            if gate_id not in self.gates:
                return False
            
            self.gates[gate_id].assigned_to = assignee
            logger.info(f"Assigned gate {gate_id} to {assignee}")
            return True
    
    def extend_deadline(self, gate_id: str, additional_hours: int) -> bool:
        """Extends the deadline for a gate.

        Args:
            gate_id: The ID of the gate to extend the deadline for.
            additional_hours: The number of additional hours to add to the deadline.

        Returns:
            True if the deadline was successfully extended, False otherwise.
        """
        with self._lock:
            if gate_id not in self.gates:
                return False
            
            gate = self.gates[gate_id]
            if gate.deadline:
                gate.deadline += timedelta(hours=additional_hours)
                logger.info(f"Extended deadline for gate {gate_id} by {additional_hours} hours")
                return True
            return False
    
    def get_workflow_statistics(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Gets detailed statistics for a workflow.

        Args:
            workflow_id: The ID of the workflow.

        Returns:
            A dictionary of workflow statistics, or None if the workflow is not found.
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None
        
        # Calculate time metrics
        total_time = 0
        if workflow.completed_at and workflow.started_at:
            total_time = (workflow.completed_at - workflow.started_at).total_seconds()
        
        # Average approval time
        approval_times = []
        for gate in workflow.approved_gates:
            if gate.submitted_at and gate.reviewed_at:
                approval_time = (gate.reviewed_at - gate.submitted_at).total_seconds()
                approval_times.append(approval_time)
        
        avg_approval_time = sum(approval_times) / len(approval_times) if approval_times else 0
        
        # Revision statistics
        revision_count = sum(1 for gate in workflow.gates if gate.metadata.get('revision_count', 0) > 0)
        
        return {
            'total_gates': len(workflow.gates),
            'approved_gates': len(workflow.approved_gates),
            'pending_gates': len(workflow.pending_gates),
            'rejected_gates': len(workflow.rejected_gates),
            'skipped_gates': len([g for g in workflow.gates if g.status == ApprovalStatus.SKIPPED]),
            'revision_count': revision_count,
            'average_approval_time_hours': f"{avg_approval_time / 3600:.1f}",
            'total_workflow_time_hours': f"{total_time / 3600:.1f}",
            'success_rate': f"{(len(workflow.approved_gates) / len(workflow.gates)) * 100:.1f}%",
            'completion_percentage': f"{workflow.progress_percentage:.1f}%"
        }
