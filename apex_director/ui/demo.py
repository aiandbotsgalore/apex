"""
APEX DIRECTOR UI System Demo
Demonstrates the complete workflow management system
"""

import asyncio
import sys
import os
from pathlib import Path
import json

# Add the parent directory to the path to import APEX DIRECTOR modules
sys.path.append(str(Path(__file__).parent.parent))

from apex_director.ui import (
    UIController, InputValidator, TreatmentGenerator, StoryboardCreator,
    ProgressMonitor, ApprovalGateSystem, ErrorHandler, DeliverablePackager
)
from apex_director.ui.progress_monitor import WorkflowStage

async def demo_complete_workflow():
    """Demonstrates the complete APEX DIRECTOR UI workflow.

    This function showcases the entire process of creating a music video project
    from start to finish, including:
    - Initializing the UI Controller
    - Creating a project session
    - Validating and processing input
    - Generating a creative treatment
    - Creating a storyboard
    - Setting up an approval workflow
    - Creating a deliverable package
    - Checking workflow progress and system status
    - Exporting a session report
    """
    print("=" * 80)
    print("APEX DIRECTOR UI SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize the UI Controller
    print("\n1. Initializing UI Controller...")
    controller = UIController()
    print("✅ UI Controller initialized successfully")
    
    # Create a project session
    print("\n2. Creating project session...")
    session_id = await controller.create_project_session(
        "Demo Music Video",
        user_metadata={"user_id": "demo_user", "project_type": "music_video"}
    )
    print(f"✅ Project session created: {session_id}")
    
    # Validate and process input
    print("\n3. Validating and processing input...")
    sample_input = {
        "project_name": "Demo Music Video",
        "audio_file": "/path/to/demo_audio.mp3",  # This would be a real file in practice
        "concept_description": "A cinematic music video featuring a person walking through a futuristic city at night, with neon lights reflecting off wet streets. The mood should be mysterious and atmospheric.",
        "visual_style": "cinematic",
        "duration_seconds": 180.0,
        "output_resolution": "1920x1080",
        "frame_rate": 30,
        "quality_level": 3
    }
    
    validation_result = await controller.validate_and_process_input(session_id, sample_input)
    
    if validation_result['valid']:
        print("✅ Input validation successful")
        processed = validation_result['processed_input']
        print(f"   - Project: {processed['project_name']}")
        print(f"   - Duration: {processed['duration_seconds']} seconds")
        print(f"   - Style: {processed['visual_style']}")
        print(f"   - Resolution: {processed['output_resolution']}")
    else:
        print("❌ Input validation failed:")
        for fix in validation_result['suggested_fixes']:
            print(f"   - {fix}")
        return
    
    # Generate creative treatment
    print("\n4. Generating creative treatment...")
    custom_requirements = {
        "treatment_type": "narrative",
        "visual_complexity": "cinematic"
    }
    
    treatment_result = await controller.generate_creative_treatment(session_id, custom_requirements)
    print("✅ Treatment generated successfully")
    print(f"   - Treatment ID: {treatment_result['treatment_id']}")
    print(f"   - Type: {treatment_result['treatment_type']}")
    print(f"   - Scenes: {treatment_result['scene_count']}")
    print(f"   - Style keywords: {treatment_result['style_keywords'][:3]}...")
    print(f"   - Color scheme: {treatment_result['color_scheme']}")
    
    # Create storyboard
    print("\n5. Creating storyboard...")
    storyboard_result = await controller.create_storyboard(session_id)
    print("✅ Storyboard created successfully")
    print(f"   - Storyboard ID: {storyboard_result['storyboard_id']}")
    print(f"   - Scenes: {storyboard_result['scene_count']}")
    print(f"   - Total shots: {storyboard_result['total_shots']}")
    
    # Show shot type distribution
    shot_dist = storyboard_result['shot_type_distribution']
    print("   - Shot distribution:")
    for shot_type, count in list(shot_dist.items())[:3]:
        print(f"     • {shot_type}: {count}")
    
    # Setup approval workflow
    print("\n6. Setting up approval workflow...")
    approval_workflow_id = await controller.setup_approval_workflow(session_id)
    print(f"✅ Approval workflow created: {approval_workflow_id}")
    
    # Create deliverable package
    print("\n7. Creating deliverable package...")
    package_id = await controller.create_deliverable_package(session_id, 'client_delivery')
    print(f"✅ Package created: {package_id}")
    
    # Get package information
    package_info = await controller.get_package_info(package_id)
    if package_info:
        print(f"   - Files: {package_info['file_count']}")
        print(f"   - Size: {package_info['total_size_mb']}")
        print(f"   - Deliverable types: {', '.join(package_info['deliverable_types'][:3])}...")
    
    # Check workflow progress
    print("\n8. Checking workflow progress...")
    progress = await controller.get_workflow_progress(session_id)
    if progress:
        print("✅ Workflow progress:")
        print(f"   - Overall: {progress['overall_progress']}")
        print(f"   - Current stage: {progress['current_stage']}")
        print(f"   - Tasks: {progress['completed_tasks']}/{progress['total_tasks']} completed")
    
    # Check system status
    print("\n9. System status:")
    system_status = controller.get_system_status()
    print(f"   - Active sessions: {system_status['active_sessions']}")
    print(f"   - Total sessions: {system_status['total_sessions']}")
    print(f"   - Packages created: {system_status['packages_created']}")
    
    # Export session report
    print("\n10. Exporting session report...")
    report_path = "demo_session_report.json"
    success = controller.export_session_report(session_id, report_path)
    if success:
        print(f"✅ Session report exported to: {report_path}")
    else:
        print("❌ Failed to export session report")
    
    # Session details
    print("\n11. Session details:")
    session_details = controller.get_session_details(session_id)
    if session_details:
        print(f"   - Session ID: {session_details['session_id']}")
        print(f"   - Status: {session_details['status']}")
        print(f"   - Current stage: {session_details['current_stage']}")
        print(f"   - Created: {session_details['created_at']}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"All components integrated and working together!")
    print(f"Check the files created:")
    print(f"  - Session report: {report_path}")
    print(f"  - Package directory: deliverables/")
    
    return session_id


async def demo_error_handling():
    """Demonstrates the error handling capabilities of the system.

    This function showcases how the ErrorHandler component can be used to:
    - Detect and log different types of errors
    - Record error context and severity
    - Generate error statistics
    - Export error reports
    """
    print("\n" + "=" * 80)
    print("ERROR HANDLING DEMO")
    print("=" * 80)
    
    # Initialize components
    controller = UIController()
    error_handler = ErrorHandler()
    
    print("\n1. Testing error detection and logging...")
    
    # Simulate different types of errors
    from apex_director.ui.error_handler import ErrorContext, ErrorSeverity, ErrorCategory
    
    # Validation error
    validation_error = ValueError("Invalid project name: contains forbidden characters")
    error_context = ErrorContext(
        component="input_validator",
        operation="validate_project_name",
        user_id="demo_user"
    )
    error_id = error_handler.handle_error(
        validation_error, 
        error_context, 
        ErrorSeverity.WARNING,
        ErrorCategory.VALIDATION
    )
    print(f"✅ Validation error recorded: {error_id}")
    
    # Generation error
    generation_error = RuntimeError("Failed to generate treatment: insufficient memory")
    error_context = ErrorContext(
        component="treatment_generator",
        operation="generate_treatment",
        workflow_id="demo_workflow"
    )
    error_id = error_handler.handle_error(
        generation_error,
        error_context,
        ErrorSeverity.ERROR,
        ErrorCategory.GENERATION
    )
    print(f"✅ Generation error recorded: {error_id}")
    
    # Show error statistics
    print("\n2. Error statistics:")
    stats = error_handler.get_error_statistics()
    print(f"   - Total errors: {stats['total_errors']}")
    print(f"   - Resolution rate: {stats['resolution_rate']}")
    print(f"   - Errors by category: {stats['errors_by_category']}")
    
    # Export error report
    print("\n3. Exporting error report...")
    error_report_path = "demo_error_report.json"
    success = error_handler.export_error_report(error_report_path)
    if success:
        print(f"✅ Error report exported to: {error_report_path}")
    
    print("\n" + "=" * 80)
    print("ERROR HANDLING DEMO COMPLETED")
    print("=" * 80)


async def demo_progress_monitoring():
    """Demonstrates the progress monitoring capabilities of the system.

    This function showcases how the ProgressMonitor component can be used to:
    - Create and manage workflows
    - Add and track tasks
    - Simulate task execution and progress updates
    - Get workflow summaries and statistics
    - Export progress reports
    """
    print("\n" + "=" * 80)
    print("PROGRESS MONITORING DEMO")
    print("=" * 80)
    
    # Initialize progress monitor
    progress_monitor = ProgressMonitor()
    progress_monitor.start_monitoring()
    
    print("\n1. Creating workflow...")
    workflow_id = progress_monitor.create_workflow("Demo Workflow")
    print(f"✅ Workflow created: {workflow_id}")
    
    print("\n2. Adding tasks...")
    task1_id = progress_monitor.add_task(
        workflow_id,
        WorkflowStage.INPUT_VALIDATION,
        "Validate Input",
        "Validate and process user input",
        priority=3,
        estimated_duration=30.0
    )
    
    task2_id = progress_monitor.add_task(
        workflow_id,
        WorkflowStage.TREATMENT_GENERATION,
        "Generate Treatment",
        "Create visual treatment",
        priority=4,
        estimated_duration=60.0
    )
    
    task3_id = progress_monitor.add_task(
        workflow_id,
        WorkflowStage.STORYBOARD_CREATION,
        "Create Storyboard",
        "Generate detailed storyboard",
        priority=4,
        estimated_duration=120.0
    )
    
    print(f"✅ Tasks added: {task1_id}, {task2_id}, {task3_id}")
    
    print("\n3. Simulating task execution...")
    
    # Start and complete first task
    print("   Starting task 1...")
    progress_monitor.start_task(task1_id)
    progress_monitor.update_task_progress(task1_id, 0.5)
    progress_monitor.update_task_progress(task1_id, 1.0)
    progress_monitor.complete_task(task1_id, {"result": "Input validated successfully"})
    print("   ✅ Task 1 completed")
    
    # Start second task
    print("   Starting task 2...")
    progress_monitor.start_task(task2_id)
    progress_monitor.update_task_progress(task2_id, 0.3)
    print(f"   🔄 Task 2 progress: 30%")
    
    # Get workflow summary
    print("\n4. Workflow progress summary:")
    summary = progress_monitor.get_workflow_summary(workflow_id)
    if summary:
        print(f"   - Overall progress: {summary['overall_progress']}")
        print(f"   - Current stage: {summary['current_stage']}")
        print(f"   - Tasks completed: {summary['completed_tasks']}/{summary['total_tasks']}")
        print(f"   - Elapsed time: {summary['elapsed_time']}")
        print(f"   - Estimated remaining: {summary['estimated_remaining']}")
    
    # Export progress report
    print("\n5. Exporting progress report...")
    progress_report_path = "demo_progress_report.json"
    success = progress_monitor.export_progress_report(workflow_id, progress_report_path)
    if success:
        print(f"✅ Progress report exported to: {progress_report_path}")
    
    # Stop monitoring
    progress_monitor.stop_monitoring()
    
    print("\n" + "=" * 80)
    print("PROGRESS MONITORING DEMO COMPLETED")
    print("=" * 80)


async def demo_approval_workflow():
    """Demonstrates the approval workflow capabilities of the system.

    This function showcases how the ApprovalGateSystem component can be used to:
    - Create and manage approval workflows
    - Check the status of workflows and gates
    - Get pending gates
    - Simulate the approval process
    - Export workflow reports
    """
    print("\n" + "=" * 80)
    print("APPROVAL WORKFLOW DEMO")
    print("=" * 80)
    
    # Initialize approval system
    approval_system = ApprovalGateSystem()
    
    print("\n1. Creating approval workflow...")
    workflow_id = approval_system.create_approval_workflow("Demo Project")
    print(f"✅ Approval workflow created: {workflow_id}")
    
    print("\n2. Checking workflow status...")
    status = approval_system.get_workflow_status(workflow_id)
    if status:
        print(f"   - Total gates: {status['gates_total']}")
        print(f"   - Pending gates: {status['gates_pending']}")
        print(f"   - Progress: {status['progress_percentage']}")
    
    print("\n3. Getting pending gates...")
    pending_gates = approval_system.get_pending_gates()
    print(f"   Found {len(pending_gates)} pending gates")
    
    if pending_gates:
        gate = pending_gates[0]
        print(f"   - Gate: {gate['name']}")
        print(f"   - Type: {gate['type']}")
        print(f"   - Deadline: {gate['days_until_deadline']} days remaining")
        
        print("\n4. Simulating approval process...")
        
        # Submit gate for approval
        print("   Submitting gate for approval...")
        approval_system.submit_for_approval(gate['gate_id'], "demo_user", "Initial submission")
        
        # Approve the gate
        print("   Approving gate...")
        approval_system.approve_gate(
            gate['gate_id'], 
            "reviewer", 
            "Approved with minor comments"
        )
        
        # Check updated status
        print("\n5. Updated workflow status:")
        updated_status = approval_system.get_workflow_status(workflow_id)
        if updated_status:
            print(f"   - Progress: {updated_status['progress_percentage']}")
            print(f"   - Approved gates: {updated_status['gates_approved']}")
    
    # Export workflow report
    print("\n6. Exporting workflow report...")
    workflow_report_path = "demo_approval_report.json"
    success = approval_system.export_workflow_report(workflow_id, workflow_report_path)
    if success:
        print(f"✅ Workflow report exported to: {workflow_report_path}")
    
    print("\n" + "=" * 80)
    print("APPROVAL WORKFLOW DEMO COMPLETED")
    print("=" * 80)


async def demo_deliverable_packaging():
    """Demonstrates the deliverable packaging capabilities of the system.

    This function showcases how the DeliverablePackager component can be used to:
    - List available package templates
    - Create a client delivery package
    - Get package information
    - List all packages
    """
    print("\n" + "=" * 80)
    print("DELIVERABLE PACKAGING DEMO")
    print("=" * 80)
    
    # Initialize packager
    packager = DeliverablePackager()
    
    print("\n1. Available package templates:")
    for template_name, template in packager.package_templates.items():
        print(f"   - {template_name}: {template.description}")
    
    print("\n2. Creating client delivery package...")
    
    # Create a sample treatment for demo
    from apex_director.ui.treatment_generator import VisualTreatment, TreatmentType, VisualComplexity
    from datetime import datetime
    
    sample_treatment = VisualTreatment(
        id="demo_treatment_123",
        project_name="Demo Music Video",
        audio_duration=180.0,
        treatment_type=TreatmentType.NARRATIVE,
        visual_complexity=VisualComplexity.CINEMATIC,
        overall_concept="A cinematic music video with dramatic lighting and atmospheric visuals.",
        scenes=[],
        color_scheme={
            "primary": ["#1a1a1a", "#333333", "#666666"],
            "secondary": ["#ff6b6b", "#4ecdc4", "#45b7d1"],
            "accent": ["#f9ca24", "#f0932b", "#eb4d4b"],
            "scheme_name": "Cinematic Dark",
            "description": "Dark cinematic palette with accent colors"
        },
        style_keywords=["cinematic", "dramatic", "atmospheric", "moody"],
        technical_specs={
            "output_resolution": [1920, 1080],
            "frame_rate": 30,
            "aspect_ratio": "16:9",
            "codec": "H.264",
            "estimated_render_time_hours": 6
        },
        creation_timestamp=datetime.utcnow()
    )
    
    # Create package
    package_id = await packager.create_package(
        "Demo Music Video",
        sample_treatment,
        None,  # No storyboard for demo
        'client_delivery'
    )
    
    print(f"✅ Package created: {package_id}")
    
    # Get package information
    print("\n3. Package information:")
    package_info = packager.get_package_info(package_id)
    if package_info:
        print(f"   - Project: {package_info['project_name']}")
        print(f"   - Files: {package_info['file_count']}")
        print(f"   - Size: {package_info['total_size_mb']}")
        print(f"   - Deliverable types:")
        for deliverable_type in package_info['deliverable_types'][:3]:
            print(f"     • {deliverable_type}")
    
    # List all packages
    print("\n4. All packages:")
    packages = packager.list_packages()
    for package in packages:
        print(f"   - {package['project_name']} v{package['version']}: {package['total_size_mb']}")
    
    print("\n" + "=" * 80)
    print("DELIVERABLE PACKAGING DEMO COMPLETED")
    print("=" * 80)


async def main():
    """Runs all demo functions."""
    print("Starting APEX DIRECTOR UI System Demo Suite...")
    print("This demo will showcase all components of the UI and workflow management system.")
    
    try:
        # Demo individual components
        await demo_progress_monitoring()
        await demo_approval_workflow()
        await demo_deliverable_packaging()
        await demo_error_handling()
        
        # Demo complete integrated workflow
        session_id = await demo_complete_workflow()
        
        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nThe APEX DIRECTOR UI System includes:")
        print("✅ Input validation and processing")
        print("✅ Creative treatment generation")
        print("✅ Detailed storyboard creation")
        print("✅ Comprehensive progress monitoring")
        print("✅ Approval workflow management")
        print("✅ Professional error handling and recovery")
        print("✅ Deliverable packaging and organization")
        print("✅ Centralized UI controller orchestrating all components")
        print("\nAll modules work together as a cohesive system!")
        print("Check the generated files for detailed reports and outputs.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            controller = UIController()
            controller.cleanup()
        except:
            pass


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
