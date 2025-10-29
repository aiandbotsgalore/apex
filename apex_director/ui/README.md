# APEX DIRECTOR UI System

A comprehensive user interface and workflow management system for APEX DIRECTOR music video generation.

## Overview

The APEX DIRECTOR UI system provides a complete workflow management solution that orchestrates all aspects of music video production from initial concept to final delivery. It integrates seamlessly with the existing APEX DIRECTOR components to provide a professional-grade production workflow.

## System Architecture

```
apex_director/ui/
├── __init__.py              # Package initialization and exports
├── input_validator.py       # Input validation and processing
├── treatment_generator.py   # Creative treatment generation
├── storyboard.py           # Detailed storyboard creation
├── progress_monitor.py     # Workflow progress tracking
├── approval_gates.py       # Approval workflow management
├── error_handler.py        # Error handling and recovery
├── deliverable_packager.py # Final deliverable packaging
├── ui_controller.py        # Central orchestration controller
└── demo.py                # Comprehensive system demonstration
```

## Core Components

### 1. Input Validator (`input_validator.py`)

Validates and processes user inputs for the music video generation workflow.

**Key Features:**
- Comprehensive input validation with severity levels
- Cross-field dependency checking
- Custom validation rules and constraints
- Detailed validation reports with suggested fixes
- Support for multiple file formats and data types

**Example Usage:**
```python
from apex_director.ui import InputValidator

validator = InputValidator()

# Validate input data
is_valid, results = await validator.validate_project_input(input_data)

if not is_valid:
    # Get suggested fixes
    fixes = validator.suggest_fixes(results)
    print("Please fix the following issues:", fixes)

# Process validated input
processed_input = await validator.process_validated_input(input_data)
```

### 2. Treatment Generator (`treatment_generator.py`)

Generates creative treatments for music videos based on validated inputs.

**Key Features:**
- Multiple treatment types (narrative, performance, conceptual, etc.)
- Style-specific templates and configurations
- Scene-by-scene breakdown with technical details
- Color palette and visual style management
- Automated creative generation with customization options

**Example Usage:**
```python
from apex_director.ui import TreatmentGenerator

generator = TreatmentGenerator()

# Generate treatment
treatment = await generator.generate_treatment(processed_input)

# Access treatment details
print(f"Treatment Type: {treatment.treatment_type}")
print(f"Scenes: {len(treatment.scenes)}")
print(f"Style: {treatment.style_keywords}")
```

### 3. Storyboard Creator (`storyboard.py`)

Creates detailed storyboards based on visual treatments.

**Key Features:**
- Shot-by-shot breakdown with camera specifications
- Multiple shot types and camera angles
- Transition management and audio sync points
- Production notes and technical specifications
- Export to multiple formats (JSON, detailed reports)

**Example Usage:**
```python
from apex_director.ui import StoryboardCreator

creator = StoryboardCreator()

# Create storyboard
storyboard = await creator.create_storyboard(treatment)

# Access storyboard details
print(f"Scenes: {storyboard.scene_count}")
print(f"Total Shots: {sum(len(scene.shots) for scene in storyboard.scenes)}")
```

### 4. Progress Monitor (`progress_monitor.py`)

Comprehensive progress tracking for the entire workflow.

**Key Features:**
- Real-time workflow progress tracking
- Task-level progress monitoring
- Stage-based workflow management
- Performance statistics and analytics
- Event-driven progress notifications

**Example Usage:**
```python
from apex_director.ui import ProgressMonitor

monitor = ProgressMonitor()

# Create workflow
workflow_id = monitor.create_workflow("My Project")

# Add and track tasks
task_id = monitor.add_task(workflow_id, stage, name, description)
monitor.start_task(task_id)
monitor.update_task_progress(task_id, 0.5)  # 50% complete
monitor.complete_task(task_id, result)

# Get progress
progress = monitor.get_workflow_summary(workflow_id)
print(f"Progress: {progress['overall_progress']}")
```

### 5. Approval Gate System (`approval_gates.py`)

Manages approval workflows and checkpoints throughout the process.

**Key Features:**
- Customizable approval gate workflows
- Deadline management and notifications
- Reviewer assignment and comments
- Approval status tracking and statistics
- Automated escalation and reminders

**Example Usage:**
```python
from apex_director.ui import ApprovalGateSystem

approval_system = ApprovalGateSystem()

# Create approval workflow
workflow_id = approval_system.create_approval_workflow("My Project")

# Submit for approval
approval_system.submit_for_approval(gate_id, "reviewer", "Initial submission")

# Approve or request revisions
approval_system.approve_gate(gate_id, "reviewer", "Approved")
# OR
approval_system.request_revision(gate_id, "reviewer", revision_notes)
```

### 6. Error Handler (`error_handler.py`)

Comprehensive error handling and recovery system.

**Key Features:**
- Categorized error types and severity levels
- Automatic error logging and reporting
- Multiple recovery strategies (retry, skip, escalate, etc.)
- Error statistics and trend analysis
- Custom error handlers and event notifications

**Example Usage:**
```python
from apex_director.ui import ErrorHandler, ErrorContext, ErrorCategory

error_handler = ErrorHandler()

# Handle errors
context = ErrorContext(
    component="treatment_generator",
    operation="generate_treatment",
    session_id="session_123"
)

error_id = error_handler.handle_error(
    error_exception,
    context,
    severity=ErrorSeverity.ERROR,
    category=ErrorCategory.GENERATION
)

# Get error details
error_details = error_handler.get_error_details(error_id)
```

### 7. Deliverable Packager (`deliverable_packager.py`)

Packages and organizes final deliverables for projects.

**Key Features:**
- Multiple package templates (client delivery, archive, review)
- Automated file organization and structure
- Package validation and integrity checking
- Archive creation (ZIP, TAR.GZ)
- Comprehensive package metadata and manifests

**Example Usage:**
```python
from apex_director.ui import DeliverablePackager

packager = DeliverablePackager()

# Create package
package_id = await packager.create_package(
    "My Project",
    treatment,
    storyboard,
    template_name='client_delivery'
)

# Get package info
package_info = packager.get_package_info(package_id)
print(f"Files: {package_info['file_count']}")
print(f"Size: {package_info['total_size_mb']}")
```

### 8. UI Controller (`ui_controller.py`)

Central orchestrator that manages the entire workflow.

**Key Features:**
- Session management for multiple projects
- Workflow orchestration and coordination
- Event-driven architecture
- Integration with all UI components
- Comprehensive system status and reporting

**Example Usage:**
```python
from apex_director.ui import get_ui_controller

controller = get_ui_controller()

# Create project session
session_id = await controller.create_project_session("My Music Video")

# Execute workflow steps
await controller.validate_and_process_input(session_id, input_data)
await controller.generate_creative_treatment(session_id)
await controller.create_storyboard(session_id)
await controller.create_deliverable_package(session_id)

# Monitor progress
progress = await controller.get_workflow_progress(session_id)
print(f"Status: {progress['status']}")
```

## Complete Workflow Example

```python
import asyncio
from apex_director.ui import get_ui_controller

async def create_music_video():
    # Initialize controller
    controller = get_ui_controller()
    
    # Create project session
    session_id = await controller.create_project_session(
        "My Music Video",
        user_metadata={"client": "Demo Client"}
    )
    
    # Step 1: Validate input
    input_data = {
        "project_name": "My Music Video",
        "audio_file": "/path/to/audio.mp3",
        "concept_description": "A vibrant performance video...",
        "visual_style": "cinematic",
        "duration_seconds": 180.0,
        "output_resolution": "1920x1080"
    }
    
    validation_result = await controller.validate_and_process_input(session_id, input_data)
    if not validation_result['valid']:
        print("Validation failed:", validation_result['suggested_fixes'])
        return
    
    # Step 2: Generate treatment
    treatment_result = await controller.generate_creative_treatment(session_id)
    print(f"Treatment created: {treatment_result['treatment_id']}")
    
    # Step 3: Create storyboard
    storyboard_result = await controller.create_storyboard(session_id)
    print(f"Storyboard created: {storyboard_result['storyboard_id']}")
    
    # Step 4: Setup approval workflow
    approval_id = await controller.setup_approval_workflow(session_id)
    print(f"Approval workflow: {approval_id}")
    
    # Step 5: Create deliverable package
    package_id = await controller.create_deliverable_package(session_id)
    print(f"Package created: {package_id}")
    
    # Get final status
    session_details = controller.get_session_details(session_id)
    print(f"Project completed: {session_details['status']}")

# Run the workflow
asyncio.run(create_music_video())
```

## System Requirements

- Python 3.8+
- APEX DIRECTOR core modules
- AsyncIO support
- File system access for package creation
- JSON support for data serialization

## Configuration

Each component can be configured through initialization parameters:

```python
# Custom error handler configuration
error_config = {
    'max_errors_per_category': 100,
    'auto_recovery_enabled': True,
    'retry_delay_seconds': 5,
    'log_to_file': True
}
error_handler = ErrorHandler(error_config)

# Custom deliverable packager output directory
packager = DeliverablePackager(base_output_dir="/custom/output/path")
```

## Integration with APEX DIRECTOR

The UI system is designed to integrate seamlessly with existing APEX DIRECTOR components:

- **Core Components**: Uses `orchestrator.py`, `asset_manager.py`, and `backend_manager.py`
- **Audio Processing**: Integrates with `audio/analyzer.py` for duration validation
- **Image Generation**: Works with `images/generator.py` for treatment visualization
- **Video Assembly**: Coordinates with `video/assembler.py` for final output

## Error Handling

The system provides comprehensive error handling at multiple levels:

1. **Component Level**: Each module handles its own errors with specific recovery strategies
2. **Workflow Level**: The UI controller coordinates error recovery across components
3. **System Level**: Global error handler manages critical system errors

## Logging

All components provide detailed logging:

- **Progress Logs**: Track workflow progress and task completion
- **Error Logs**: Detailed error information with context
- **Performance Logs**: Monitor system performance and resource usage
- **Audit Logs**: Track all user actions and decisions

## Performance Considerations

- **Async Operations**: All I/O operations use asyncio for responsiveness
- **Resource Management**: Automatic cleanup of temporary files and resources
- **Progress Monitoring**: Efficient status tracking without performance impact
- **Batch Operations**: Optimized for processing multiple items efficiently

## Extensibility

The system is designed for easy extension:

- **Custom Validators**: Add new validation rules and constraints
- **Treatment Styles**: Create new visual treatment templates
- **Shot Types**: Define custom shot types and camera specifications
- **Package Templates**: Create custom deliverable package structures
- **Event Handlers**: Add custom event handlers for system events

## Demo and Examples

Run the comprehensive demo to see all components in action:

```bash
python apex_director/ui/demo.py
```

This demo showcases:
- Complete workflow from start to finish
- Error handling and recovery
- Progress monitoring
- Approval workflow
- Deliverable packaging

## Testing

Each module includes comprehensive error handling and validation:

```python
# Test validation
is_valid, results = await validator.validate_project_input(test_data)
assert is_valid, "Input validation should pass"

# Test treatment generation
treatment = await generator.generate_treatment(processed_input)
assert treatment.treatment_type is not None

# Test storyboard creation
storyboard = await creator.create_storyboard(treatment)
assert len(storyboard.scenes) > 0
```

## Best Practices

1. **Always validate input** before processing
2. **Use try-catch blocks** around workflow operations
3. **Monitor progress** regularly for long-running operations
4. **Handle errors gracefully** with appropriate user feedback
5. **Clean up resources** when projects are completed
6. **Use appropriate package templates** for different delivery needs
7. **Implement proper logging** for debugging and monitoring

## Troubleshooting

### Common Issues

**Input Validation Failures:**
- Check file paths and formats
- Ensure all required fields are present
- Verify data types and constraints

**Treatment Generation Errors:**
- Review input data quality
- Check available style templates
- Verify system resources

**Workflow Progress Issues:**
- Monitor task completion status
- Check for dependency issues
- Review error logs

**Package Creation Problems:**
- Verify file permissions
- Check disk space availability
- Ensure valid template configuration

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.getLogger('apex_director.ui').setLevel(logging.DEBUG)
```

## Contributing

When extending the UI system:

1. Follow the existing error handling patterns
2. Include comprehensive logging
3. Add validation for all inputs
4. Document new features and APIs
5. Include tests for new functionality
6. Update the demo to showcase new features

## License

Part of the APEX DIRECTOR project. See main project license for details.

---

**Note**: This UI system is designed for professional music video production workflows and provides enterprise-grade functionality for managing complex creative projects from concept to delivery.
