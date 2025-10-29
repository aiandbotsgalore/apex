"""
Usage Examples for Cinematic Image Generation Pipeline
Demonstrates key features and workflows
"""

import asyncio
from pathlib import Path
from apex_director.images import (
    CinematicImageGenerator,
    GenerationRequest,
    CameraSettings,
    LightingSetup,
    Composition,
    quick_generate,
    batch_generate
)

async def basic_generation_example():
    """Basic image generation with defaults"""
    
    print("=== Basic Generation Example ===")
    
    # Quick generation with defaults
    result = await quick_generate(
        prompt="A cinematic portrait of a detective in a noir setting",
        output_dir="examples/basic_generation"
    )
    
    print(f"Generated {len(result.selected_variants)} variants")
    print(f"Generation time: {result.generation_time:.2f}s")
    print(f"Quality score: {result.overall_quality_score:.3f}")
    print(f"Results saved to: {result.request.output_dir}")

async def advanced_generation_example():
    """Advanced generation with full customization"""
    
    print("\n=== Advanced Generation Example ===")
    
    # Create generator
    generator = CinematicImageGenerator(
        style_bible_path="style_bible.json",
        output_dir="examples/advanced_generation"
    )
    
    # Create detailed request
    request = GenerationRequest(
        prompt="A space marine in a futuristic battlefield",
        scene_id="battle_scene_01",
        character_name="commander_sarah",
        genre="sci_fi",
        director_style="christopher_nolan",
        camera_settings=CameraSettings(
            lens="35mm",
            aperture="f/2.8",
            iso="ISO 800"
        ),
        lighting_setup=LightingSetup(
            key_light="volumetric_lighting",
            fill_light="blue_accents",
            rim_light="practical_glow",
            background="dark_space",
            mood="dramatic"
        ),
        composition=Composition(
            rule_of_thirds=True,
            leading_lines=True,
            depth_of_field="shallow",
            framing="medium_shot"
        ),
        backend_preference="minimax",
        num_variants=6,
        upscale=True,
        upscale_preset="broadcast_quality",
        output_dir=Path("examples/advanced_generation")
    )
    
    # Generate image
    result = await generator.generate_single_image(request)
    
    print(f"Generated {len(result.selected_variants)} variants")
    print(f"Style consistency: {result.style_consistency_score:.3f}")
    print(f"Character consistency: {result.character_consistency_score:.3f}")
    print(f"Overall quality: {result.overall_quality_score:.3f}")
    
    # Display variant details
    for i, variant in enumerate(result.selected_variants, 1):
        print(f"\nVariant {i}:")
        print(f"  Selection reason: {variant.selection_reason}")
        print(f"  CLIP aesthetic: {variant.scores.clip_aesthetic_score:.3f}")
        print(f"  Composition: {variant.scores.composition_score:.3f}")
        print(f"  Style consistency: {variant.scores.style_consistency_score:.3f}")
        print(f"  Artifact quality: {variant.scores.artifacts_score:.3f}")
        print(f"  Overall score: {variant.scores.overall_score:.3f}")

async def batch_generation_example():
    """Batch generation for multiple scenes"""
    
    print("\n=== Batch Generation Example ===")
    
    # Define multiple requests
    requests = [
        GenerationRequest(
            prompt="Opening shot of a medieval castle at dawn",
            genre="fantasy",
            output_dir=Path("examples/batch_scene_01")
        ),
        GenerationRequest(
            prompt="Interior shot of a cozy tavern with adventurers",
            genre="fantasy", 
            output_dir=Path("examples/batch_scene_02")
        ),
        GenerationRequest(
            prompt="Epic battle scene with magic and swords",
            genre="fantasy",
            output_dir=Path("examples/batch_scene_03")
        )
    ]
    
    # Generate sequence
    generator = CinematicImageGenerator(output_dir="examples/batch_generation")
    
    def progress_callback(current, total, message):
        print(f"Progress: {current}/{total} - {message}")
    
    results = await generator.generate_image_sequence(requests, progress_callback)
    
    print(f"\nBatch generation completed:")
    print(f"Successful: {len(results)}/{len(requests)}")
    
    for i, result in enumerate(results, 1):
        print(f"Scene {i}: Quality {result.overall_quality_score:.3f}")

async def character_consistency_example():
    """Demonstrate character consistency across scenes"""
    
    print("\n=== Character Consistency Example ===")
    
    generator = CinematicImageGenerator(
        output_dir="examples/character_consistency"
    )
    
    # Create character profile (would normally load from reference images)
    print("Creating character profile for 'Commander Sarah'...")
    
    # Generate multiple shots of the same character
    character_shots = [
        GenerationRequest(
            prompt="Commander Sarah in tactical gear, close-up portrait",
            character_name="commander_sarah",
            output_dir=Path("examples/character_consistency/portrait")
        ),
        GenerationRequest(
            prompt="Commander Sarah in command center, medium shot",
            character_name="commander_sarah",
            output_dir=Path("examples/character_consistency/command_center")
        ),
        GenerationRequest(
            prompt="Commander Sarah leading troops, wide action shot",
            character_name="commander_sarah",
            output_dir=Path("examples/character_consistency/action")
        )
    ]
    
    results = await generator.generate_image_sequence(character_shots)
    
    print(f"\nCharacter consistency results:")
    for i, result in enumerate(results, 1):
        print(f"Shot {i}: Character consistency {result.character_consistency_score:.3f}")

async def style_variation_example():
    """Demonstrate style variations for different moods"""
    
    print("\n=== Style Variation Example ===")
    
    # Same subject, different styles
    base_prompt = "A lonely figure walking through a foggy city street"
    
    style_requests = [
        GenerationRequest(
            prompt=base_prompt,
            genre="film_noir",
            director_style="wes_anderson",
            output_dir=Path("examples/style_variations/noir")
        ),
        GenerationRequest(
            prompt=base_prompt,
            genre="sci_fi",
            director_style="guillermo_del_toro",
            output_dir=Path("examples/style_variations/scifi")
        ),
        GenerationRequest(
            prompt=base_prompt,
            genre="romance",
            director_style="christopher_nolan",
            output_dir=Path("examples/style_variations/romance")
        )
    ]
    
    generator = CinematicImageGenerator(output_dir="examples/style_variations")
    results = await generator.generate_image_sequence(style_requests)
    
    print("Style variation results:")
    for i, result in enumerate(results):
        print(f"Style {i+1}: Overall quality {result.overall_quality_score:.3f}")

async def technical_quality_example():
    """Demonstrate technical quality features"""
    
    print("\n=== Technical Quality Example ===")
    
    generator = CinematicImageGenerator(output_dir="examples/technical_quality")
    
    # Test different upscaling presets
    test_image = "A detailed architectural shot for quality testing"
    
    presets = ["web_optimized", "high_quality", "broadcast_quality"]
    
    for preset in presets:
        print(f"\nTesting {preset} upscaling...")
        
        request = GenerationRequest(
            prompt=test_image,
            upscale=True,
            upscale_preset=preset,
            output_dir=Path(f"examples/technical_quality/{preset}")
        )
        
        result = await generator.generate_single_image(request)
        
        if result.upscaled_variants:
            upscale_result = result.upscaled_variants[0]
            quality_metrics = upscale_result.quality_metrics
            
            print(f"  Resolution: {result.request.output_dir}")
            print(f"  Overall quality: {quality_metrics.get('overall_quality', 0):.3f}")
            print(f"  Sharpness: {quality_metrics.get('sharpness', 0):.3f}")
            print(f"  Processing time: {upscale_result.processing_time:.2f}s")

async def system_monitoring_example():
    """Demonstrate system monitoring and health checks"""
    
    print("\n=== System Monitoring Example ===")
    
    generator = CinematicImageGenerator()
    
    # Get system status
    status = generator.get_system_status()
    
    print("System Status:")
    print(f"  Available backends: {status['available_backends']}")
    print(f"  Character profiles: {status['character_count']}")
    print(f"  Style bible loaded: {status['style_bible_loaded']}")
    print(f"  Upscaler presets: {status['upscaler_presets']}")
    print(f"  Output directory: {status['output_directory']}")
    
    # Health check
    health = await generator.health_check()
    
    print("\nHealth Check:")
    for component, healthy in health.items():
        status_str = "✓" if healthy else "✗"
        print(f"  {component}: {status_str}")

async def project_template_example():
    """Demonstrate project template creation"""
    
    print("\n=== Project Template Example ===")
    
    generator = CinematicImageGenerator()
    
    # Create new project template
    template_files = generator.create_project_template("my_cinematic_project")
    
    print("Created project template:")
    for name, path in template_files.items():
        print(f"  {name}: {path}")
    
    print(f"\nProject directory structure created at: {template_files['project_dir']}")

async def main():
    """Run all examples"""
    
    print("Cinematic Image Generation Pipeline - Usage Examples")
    print("=" * 60)
    
    examples = [
        basic_generation_example,
        advanced_generation_example,
        batch_generation_example,
        character_consistency_example,
        style_variation_example,
        technical_quality_example,
        system_monitoring_example,
        project_template_example
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Example failed: {e}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")

if __name__ == "__main__":
    asyncio.run(main())

# Additional usage patterns

def quick_batch_example():
    """Simple batch generation with minimal code"""
    
    # Generate multiple images quickly
    prompts = [
        "A steampunk inventor in his workshop",
        "A fantasy wizard casting a spell",
        "A cyberpunk hacker in neon-lit alley"
    ]
    
    # This would generate all three with default settings
    results = asyncio.run(batch_generate(prompts, "quick_batch"))
    
    print(f"Generated {len(results)} images")

def custom_pipeline_example():
    """Custom pipeline with manual control"""
    
    from apex_director.images import (
        BackendManager, 
        CinematographyPromptEngineer,
        VariantSelector,
        ProfessionalUpscaler
    )
    
    # Manual pipeline setup
    backend_manager = BackendManager()
    prompt_engineer = CinematographyPromptEngineer()
    variant_selector = VariantSelector()
    upscaler = ProfessionalUpscaler()
    
    # Custom workflow
    async def custom_workflow():
        # 1. Generate prompt
        prompt = prompt_engineer.generate_cinematic_prompt(
            subject="A robot in a garden",
            genre="sci_fi"
        )
        
        # 2. Generate with specific backend
        config = GenerationConfig(
            width=1024, 
            height=1024, 
            num_variants=4,
            backend="minimax"
        )
        
        variants = await backend_manager.generate_with_backend(
            "minimax", prompt, config
        )
        
        # 3. Score and select
        variant_list = [(f"custom_{i}", img) for i, img in enumerate(variants)]
        selected = variant_selector.select_best_variants(
            variant_list, prompt, num_selections=1
        )
        
        # 4. Upscale
        if selected:
            upscale_result = upscaler.upscale_image(
                selected[0].image,
                preset="high_quality"
            )
            
            return selected[0], upscale_result
        
        return None, None
    
    result, upscaled = asyncio.run(custom_workflow())
    
    if result:
        print(f"Custom pipeline completed: {result.variant_id}")
        print(f"Quality score: {result.scores.overall_score:.3f}")

# Error handling examples
async def robust_generation_example():
    """Generation with comprehensive error handling"""
    
    generator = CinematicImageGenerator()
    
    request = GenerationRequest(
        prompt="A challenging scene with complex lighting",
        output_dir=Path("examples/error_handling")
    )
    
    try:
        result = await generator.generate_single_image(request)
        print(f"Generation successful: {result.overall_quality_score:.3f}")
        
    except ValueError as e:
        print(f"Validation error: {e}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Implement fallback strategy
        
    finally:
        # Cleanup
        print("Generation attempt completed")