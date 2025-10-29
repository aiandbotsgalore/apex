"""
APEX DIRECTOR Error Handling and Recovery System
Comprehensive error handling, logging, and recovery mechanisms
"""

import traceback
import uuid
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import sys
import os
from pathlib import Path

from .progress_monitor import WorkflowTask, TaskStatus
from .approval_gates import ApprovalGate

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Categories of errors"""
    VALIDATION = "validation"
    GENERATION = "generation"
    IO_ERROR = "io_error"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RESOURCE = "resource"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    WORKFLOW = "workflow"
    APPROVAL = "approval"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for errors"""
    RETRY = "retry"
    SKIP = "skip"
    PAUSE = "pause"
    ESCALATE = "escalate"
    MANUAL_INTERVENTION = "manual_intervention"
    FALLBACK = "fallback"
    ABORT = "abort"
    IGNORE = "ignore"


@dataclass
class ErrorContext:
    """Context information for an error"""
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    task_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """Complete error record"""
    id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: str
    context: ErrorContext
    stack_trace: Optional[str] = None
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    retry_count: int = 0
    max_retries: int = 3
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    related_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['severity'] = self.severity.value
        data['category'] = self.category.value
        data['recovery_strategy'] = self.recovery_strategy.value
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data
    
    @property
    def is_retryable(self) -> bool:
        """Check if error is retryable"""
        return (self.recovery_strategy in [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK] and
                self.retry_count < self.max_retries)
    
    @property
    def age_hours(self) -> float:
        """Get error age in hours"""
        return (datetime.utcnow() - self.timestamp).total_seconds() / 3600


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    id: str
    error_id: str
    strategy: RecoveryStrategy
    attempted_at: datetime
    success: bool
    duration_seconds: float
    details: str
    result_data: Optional[Dict[str, Any]] = None


class ErrorHandler:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.errors: Dict[str, ErrorRecord] = {}
        self.recovery_attempts: Dict[str, List[RecoveryAttempt]] = {}
        self.error_handlers: Dict[ErrorCategory, List[Callable]] = {}
        self.global_handlers: List[Callable] = []
        self.recovery_handlers: Dict[RecoveryStrategy, Callable] = {}
        self._lock = threading.Lock()
        
        # Error statistics
        self.error_counts: Dict[str, int] = {}
        self.last_error_by_category: Dict[str, datetime] = {}
        
        self._setup_logging()
        self._setup_default_handlers()
        
        logger.info("Error Handler initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default error handler configuration"""
        return {
            'max_errors_per_category': 100,
            'error_retention_days': 30,
            'auto_recovery_enabled': True,
            'retry_delay_seconds': 5,
            'escalation_threshold': 5,
            'notification_channels': ['log', 'console'],
            'log_to_file': True,
            'log_file_path': 'logs/errors.log',
            'backup_error_storage': True
        }
    
    def _setup_logging(self):
        """Setup error logging configuration"""
        # Create logs directory if it doesn't exist
        log_file = self.config.get('log_file_path', 'logs/errors.log')
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Configure error logger
        error_logger = logging.getLogger('apex_director.errors')
        error_logger.setLevel(logging.DEBUG)
        
        # File handler
        if self.config.get('log_to_file', True):
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            error_logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        error_logger.addHandler(console_handler)
        
        # Store error logger reference
        self.error_logger = error_logger
    
    def _setup_default_handlers(self):
        """Setup default error and recovery handlers"""
        # Error handlers by category
        self.error_handlers[ErrorCategory.VALIDATION] = [self._handle_validation_error]
        self.error_handlers[ErrorCategory.GENERATION] = [self._handle_generation_error]
        self.error_handlers[ErrorCategory.IO_ERROR] = [self._handle_io_error]
        self.error_handlers[ErrorCategory.NETWORK] = [self._handle_network_error]
        self.error_handlers[ErrorCategory.WORKFLOW] = [self._handle_workflow_error]
        self.error_handlers[ErrorCategory.APPROVAL] = [self._handle_approval_error]
        
        # Recovery handlers
        self.recovery_handlers[RecoveryStrategy.RETRY] = self._retry_recovery
        self.recovery_handlers[RecoveryStrategy.SKIP] = self._skip_recovery
        self.recovery_handlers[RecoveryStrategy.ESCALATE] = self._escalate_recovery
        self.recovery_handlers[RecoveryStrategy.FALLBACK] = self._fallback_recovery
        self.recovery_handlers[RecoveryStrategy.MANUAL_INTERVENTION] = self._manual_intervention_recovery
        self.recovery_handlers[RecoveryStrategy.ABORT] = self._abort_recovery
        self.recovery_handlers[RecoveryStrategy.IGNORE] = self._ignore_recovery
    
    def handle_error(self,
                    error: Exception,
                    context: ErrorContext,
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    category: Optional[ErrorCategory] = None,
                    recovery_strategy: Optional[RecoveryStrategy] = None,
                    custom_message: Optional[str] = None) -> str:
        """
        Handle an error with comprehensive logging and recovery
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            severity: Severity level of the error
            category: Category of the error
            recovery_strategy: Preferred recovery strategy
            custom_message: Custom error message
            
        Returns:
            Error ID for tracking
        """
        # Determine error category if not provided
        if category is None:
            category = self._determine_error_category(error, context)
        
        # Determine recovery strategy if not provided
        if recovery_strategy is None:
            recovery_strategy = self._determine_recovery_strategy(error, category)
        
        # Create error record
        error_id = str(uuid.uuid4())
        error_record = ErrorRecord(
            id=error_id,
            timestamp=datetime.utcnow(),
            severity=severity,
            category=category,
            message=custom_message or str(error),
            details=f"Error in {context.component}.{context.operation}: {str(error)}",
            context=context,
            stack_trace=traceback.format_exc() if isinstance(error, Exception) else None,
            recovery_strategy=recovery_strategy,
            max_retries=self._get_max_retries_for_category(category)
        )
        
        with self._lock:
            self.errors[error_id] = error_record
            
            # Update statistics
            self.error_counts[category.value] = self.error_counts.get(category.value, 0) + 1
            self.last_error_by_category[category.value] = datetime.utcnow()
            
            # Cleanup old errors
            self._cleanup_old_errors()
        
        # Log the error
        self._log_error(error_record)
        
        # Trigger error handlers
        self._trigger_error_handlers(error_record, error)
        
        # Attempt recovery if enabled
        if self.config.get('auto_recovery_enabled', True):
            self._attempt_recovery(error_id)
        
        logger.info(f"Handled error {error_id}: {category.value} - {custom_message or str(error)}")
        return error_id
    
    def _determine_error_category(self, error: Exception, context: ErrorContext) -> ErrorCategory:
        """Determine error category based on error type and context"""
        error_type = type(error).__name__.lower()
        
        if 'validation' in error_type or 'value' in error_type:
            return ErrorCategory.VALIDATION
        elif 'io' in error_type or 'file' in error_type:
            return ErrorCategory.IO_ERROR
        elif 'network' in error_type or 'connection' in error_type:
            return ErrorCategory.NETWORK
        elif 'auth' in error_type or 'permission' in error_type:
            return ErrorCategory.AUTHENTICATION
        elif 'resource' in error_type or 'memory' in error_type or 'disk' in error_type:
            return ErrorCategory.RESOURCE
        elif 'system' in error_type:
            return ErrorCategory.SYSTEM
        elif context.component in ['input_validator', 'treatment_generator', 'storyboard']:
            return ErrorCategory.GENERATION
        elif context.component in ['progress_monitor', 'workflow']:
            return ErrorCategory.WORKFLOW
        elif context.component in ['approval_gates']:
            return ErrorCategory.APPROVAL
        else:
            return ErrorCategory.UNKNOWN
    
    def _determine_recovery_strategy(self, error: Exception, category: ErrorCategory) -> RecoveryStrategy:
        """Determine appropriate recovery strategy"""
        error_type = type(error).__name__.lower()
        
        # Category-specific defaults
        category_strategies = {
            ErrorCategory.VALIDATION: RecoveryStrategy.MANUAL_INTERVENTION,
            ErrorCategory.GENERATION: RecoveryStrategy.RETRY,
            ErrorCategory.IO_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.NETWORK: RecoveryStrategy.RETRY,
            ErrorCategory.AUTHENTICATION: RecoveryStrategy.ESCALATE,
            ErrorCategory.RESOURCE: RecoveryStrategy.ESCALATE,
            ErrorCategory.SYSTEM: RecoveryStrategy.ABORT,
            ErrorCategory.WORKFLOW: RecoveryStrategy.RETRY,
            ErrorCategory.APPROVAL: RecoveryStrategy.MANUAL_INTERVENTION,
            ErrorCategory.UNKNOWN: RecoveryStrategy.RETRY
        }
        
        default_strategy = category_strategies.get(category, RecoveryStrategy.RETRY)
        
        # Check for specific error patterns
        if 'timeout' in error_type:
            return RecoveryStrategy.RETRY
        elif 'not found' in str(error).lower():
            return RecoveryStrategy.SKIP
        elif 'permission denied' in str(error).lower():
            return RecoveryStrategy.ESCALATE
        elif 'disk full' in str(error).lower():
            return RecoveryStrategy.ESCALATE
        else:
            return default_strategy
    
    def _get_max_retries_for_category(self, category: ErrorCategory) -> int:
        """Get max retries for a specific error category"""
        retry_limits = {
            ErrorCategory.VALIDATION: 0,
            ErrorCategory.GENERATION: 3,
            ErrorCategory.IO_ERROR: 2,
            ErrorCategory.NETWORK: 5,
            ErrorCategory.AUTHENTICATION: 1,
            ErrorCategory.RESOURCE: 1,
            ErrorCategory.SYSTEM: 0,
            ErrorCategory.WORKFLOW: 3,
            ErrorCategory.APPROVAL: 0,
            ErrorCategory.UNKNOWN: 2
        }
        return retry_limits.get(category, 2)
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error according to configured channels"""
        log_message = f"[{error_record.severity.value.upper()}] {error_record.category.value}: {error_record.message}"
        log_details = {
            'error_id': error_record.id,
            'component': error_record.context.component,
            'operation': error_record.context.operation,
            'timestamp': error_record.timestamp.isoformat(),
            'retry_count': error_record.retry_count
        }
        
        # Log to configured channels
        if 'log' in self.config.get('notification_channels', ['log']):
            if error_record.severity == ErrorSeverity.CRITICAL:
                self.error_logger.critical(f"{log_message} - {log_details}")
            elif error_record.severity == ErrorSeverity.ERROR:
                self.error_logger.error(f"{log_message} - {log_details}")
            elif error_record.severity == ErrorSeverity.WARNING:
                self.error_logger.warning(f"{log_message} - {log_details}")
            else:
                self.error_logger.info(f"{log_message} - {log_details}")
        
        if 'console' in self.config.get('notification_channels', ['log']):
            print(f"[{error_record.severity.value.upper()}] {log_message}")
    
    def _trigger_error_handlers(self, error_record: ErrorRecord, original_error: Exception):
        """Trigger registered error handlers"""
        # Category-specific handlers
        handlers = self.error_handlers.get(error_record.category, [])
        for handler in handlers:
            try:
                handler(error_record, original_error)
            except Exception as e:
                self.error_logger.error(f"Error in category handler: {e}")
        
        # Global handlers
        for handler in self.global_handlers:
            try:
                handler(error_record, original_error)
            except Exception as e:
                self.error_logger.error(f"Error in global handler: {e}")
    
    def _attempt_recovery(self, error_id: str):
        """Attempt to recover from an error"""
        error_record = self.errors.get(error_id)
        if not error_record or not error_record.is_retryable:
            return
        
        try:
            handler = self.recovery_handlers.get(error_record.recovery_strategy)
            if handler:
                handler(error_id)
        except Exception as e:
            self.error_logger.error(f"Recovery attempt failed for error {error_id}: {e}")
    
    # Recovery strategy implementations
    
    def _retry_recovery(self, error_id: str):
        """Implement retry recovery strategy"""
        error_record = self.errors[error_id]
        retry_delay = self.config.get('retry_delay_seconds', 5)
        
        # Record retry attempt
        self._record_recovery_attempt(error_id, RecoveryStrategy.RETRY, True, retry_delay, "Automatic retry")
        
        error_record.retry_count += 1
        
        if error_record.retry_count < error_record.max_retries:
            self.error_logger.info(f"Scheduling retry {error_record.retry_count}/{error_record.max_retries} for error {error_id}")
            # In a real implementation, this would schedule a retry task
            # For now, we'll just mark it for retry
            # asyncio.create_task(self._delayed_retry(error_id, retry_delay))
        else:
            error_record.recovery_strategy = RecoveryStrategy.ESCALATE
            self.error_logger.warning(f"Max retries reached for error {error_id}, escalating")
    
    def _skip_recovery(self, error_id: str):
        """Implement skip recovery strategy"""
        error_record = self.errors[error_id]
        self._record_recovery_attempt(error_id, RecoveryStrategy.SKIP, True, 0, "Skipping failed operation")
        self._mark_error_resolved(error_id, "Operation skipped")
    
    def _escalate_recovery(self, error_id: str):
        """Implement escalate recovery strategy"""
        error_record = self.errors[error_id]
        self._record_recovery_attempt(error_id, RecoveryStrategy.ESCALATE, True, 0, "Error escalated to higher level")
        # In a real implementation, this would notify administrators
        self.error_logger.critical(f"Error {error_id} escalated: {error_record.message}")
    
    def _fallback_recovery(self, error_id: str):
        """Implement fallback recovery strategy"""
        error_record = self.errors[error_id]
        self._record_recovery_attempt(error_id, RecoveryStrategy.FALLBACK, True, 0, "Using fallback mechanism")
        self._mark_error_resolved(error_id, "Fallback mechanism used")
    
    def _manual_intervention_recovery(self, error_id: str):
        """Implement manual intervention recovery strategy"""
        error_record = self.errors[error_id]
        self._record_recovery_attempt(error_id, RecoveryStrategy.MANUAL_INTERVENTION, False, 0, "Manual intervention required")
        self.error_logger.warning(f"Error {error_id} requires manual intervention: {error_record.message}")
    
    def _abort_recovery(self, error_id: str):
        """Implement abort recovery strategy"""
        error_record = self.errors[error_id]
        self._record_recovery_attempt(error_id, RecoveryStrategy.ABORT, True, 0, "Operation aborted")
        self._mark_error_resolved(error_id, "Operation aborted due to critical error")
    
    def _ignore_recovery(self, error_id: str):
        """Implement ignore recovery strategy"""
        error_record = self.errors[error_id]
        self._record_recovery_attempt(error_id, RecoveryStrategy.IGNORE, True, 0, "Error ignored")
        self._mark_error_resolved(error_id, "Error ignored")
    
    def _record_recovery_attempt(self, 
                               error_id: str,
                               strategy: RecoveryStrategy,
                               success: bool,
                               duration: float,
                               details: str):
        """Record a recovery attempt"""
        attempt = RecoveryAttempt(
            id=str(uuid.uuid4()),
            error_id=error_id,
            strategy=strategy,
            attempted_at=datetime.utcnow(),
            success=success,
            duration_seconds=duration,
            details=details
        )
        
        if error_id not in self.recovery_attempts:
            self.recovery_attempts[error_id] = []
        
        self.recovery_attempts[error_id].append(attempt)
    
    def _mark_error_resolved(self, error_id: str, resolution_notes: str):
        """Mark an error as resolved"""
        if error_id in self.errors:
            self.errors[error_id].resolved = True
            self.errors[error_id].resolved_at = datetime.utcnow()
            self.errors[error_id].resolution_notes = resolution_notes
    
    def _cleanup_old_errors(self):
        """Cleanup old error records"""
        retention_days = self.config.get('error_retention_days', 30)
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        errors_to_remove = []
        for error_id, error_record in self.errors.items():
            if error_record.resolved and error_record.resolved_at and error_record.resolved_at < cutoff_date:
                errors_to_remove.append(error_id)
        
        for error_id in errors_to_remove:
            del self.errors[error_id]
            if error_id in self.recovery_attempts:
                del self.recovery_attempts[error_id]
    
    # Category-specific error handlers
    
    def _handle_validation_error(self, error_record: ErrorRecord, original_error: Exception):
        """Handle validation errors"""
        # Validation errors usually require manual intervention
        self.error_logger.warning(f"Validation error in {error_record.context.component}: {error_record.message}")
    
    def _handle_generation_error(self, error_record: ErrorRecord, original_error: Exception):
        """Handle generation errors"""
        self.error_logger.error(f"Generation error in {error_record.context.component}: {error_record.message}")
        # Could trigger fallback generation strategies
    
    def _handle_io_error(self, error_record: ErrorRecord, original_error: Exception):
        """Handle I/O errors"""
        if 'disk' in str(original_error).lower() and 'full' in str(original_error).lower():
            self.error_logger.critical("Disk space issue detected - immediate attention required")
        else:
            self.error_logger.error(f"I/O error in {error_record.context.component}: {error_record.message}")
    
    def _handle_network_error(self, error_record: ErrorRecord, original_error: Exception):
        """Handle network errors"""
        self.error_logger.warning(f"Network error in {error_record.context.component}: {error_record.message}")
        # Could implement retry with exponential backoff
    
    def _handle_workflow_error(self, error_record: ErrorRecord, original_error: Exception):
        """Handle workflow errors"""
        self.error_logger.error(f"Workflow error in {error_record.context.component}: {error_record.message}")
        # Could pause workflow or skip problematic tasks
    
    def _handle_approval_error(self, error_record: ErrorRecord, original_error: Exception):
        """Handle approval system errors"""
        self.error_logger.error(f"Approval error in {error_record.context.component}: {error_record.message}")
        # Could notify approval stakeholders
    
    # Public interface methods
    
    def get_error_details(self, error_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an error"""
        error_record = self.errors.get(error_id)
        if not error_record:
            return None
        
        data = error_record.to_dict()
        data['recovery_attempts'] = [asdict(attempt) for attempt in self.recovery_attempts.get(error_id, [])]
        return data
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        total_errors = len(self.errors)
        resolved_errors = len([e for e in self.errors.values() if e.resolved])
        unresolved_errors = total_errors - resolved_errors
        
        # Count by severity
        severity_counts = {}
        for error in self.errors.values():
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by category
        category_counts = {}
        for error in self.errors.values():
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Average resolution time
        resolved_times = []
        for error in self.errors.values():
            if error.resolved and error.resolved_at:
                resolution_time = (error.resolved_at - error.timestamp).total_seconds()
                resolved_times.append(resolution_time)
        
        avg_resolution_time = sum(resolved_times) / len(resolved_times) if resolved_times else 0
        
        return {
            'total_errors': total_errors,
            'resolved_errors': resolved_errors,
            'unresolved_errors': unresolved_errors,
            'resolution_rate': f"{(resolved_errors / total_errors) * 100:.1f}%" if total_errors > 0 else "0%",
            'average_resolution_time_hours': f"{avg_resolution_time / 3600:.1f}",
            'errors_by_severity': severity_counts,
            'errors_by_category': category_counts,
            'recent_errors_24h': len([e for e in self.errors.values() if e.age_hours < 24])
        }
    
    def get_errors_by_category(self, category: ErrorCategory, include_resolved: bool = True) -> List[Dict[str, Any]]:
        """Get errors filtered by category"""
        filtered_errors = []
        for error in self.errors.values():
            if error.category == category:
                if include_resolved or not error.resolved:
                    filtered_errors.append(self.get_error_details(error.id))
        
        return filtered_errors
    
    def export_error_report(self, file_path: str, include_resolved: bool = True) -> bool:
        """Export comprehensive error report"""
        try:
            report = {
                'generated_at': datetime.utcnow().isoformat(),
                'statistics': self.get_error_statistics(),
                'errors': []
            }
            
            for error in self.errors.values():
                if include_resolved or not error.resolved:
                    error_data = self.get_error_details(error.id)
                    if error_data:
                        report['errors'].append(error_data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Error report exported to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export error report: {e}")
            return False
    
    def add_error_handler(self, category: ErrorCategory, handler: Callable):
        """Add custom error handler for a category"""
        if category not in self.error_handlers:
            self.error_handlers[category] = []
        self.error_handlers[category].append(handler)
        logger.info(f"Added error handler for {category.value}")
    
    def add_global_error_handler(self, handler: Callable):
        """Add global error handler for all errors"""
        self.global_handlers.append(handler)
        logger.info("Added global error handler")
    
    def resolve_error(self, error_id: str, resolution_notes: str) -> bool:
        """Manually resolve an error"""
        if error_id in self.errors:
            self._mark_error_resolved(error_id, resolution_notes)
            logger.info(f"Manually resolved error {error_id}")
            return True
        return False
    
    def retry_error(self, error_id: str) -> bool:
        """Manually retry an error"""
        if error_id in self.errors:
            error_record = self.errors[error_id]
            if error_record.is_retryable:
                error_record.retry_count = 0  # Reset retry count
                error_record.resolved = False
                self._attempt_recovery(error_id)
                logger.info(f"Manually retried error {error_id}")
                return True
        return False
    
    def get_critical_errors(self) -> List[Dict[str, Any]]:
        """Get all critical errors that need attention"""
        critical_errors = []
        for error in self.errors.values():
            if (error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL] and 
                not error.resolved):
                critical_errors.append(self.get_error_details(error.id))
        
        return critical_errors
    
    def get_workflow_errors(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get errors related to a specific workflow"""
        workflow_errors = []
        for error in self.errors.values():
            if error.context.workflow_id == workflow_id:
                workflow_errors.append(self.get_error_details(error.id))
        
        return workflow_errors
