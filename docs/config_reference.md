# APEX DIRECTOR Configuration Reference

Comprehensive guide to configuring APEX DIRECTOR for optimal performance and functionality.

## Table of Contents

- [Configuration Overview](#configuration-overview)
- [System Configuration](#system-configuration)
- [Backend Configuration](#backend-configuration)
- [Image Generation Configuration](#image-generation-configuration)
- [Video Assembly Configuration](#video-assembly-configuration)
- [Audio Processing Configuration](#audio-processing-configuration)
- [Quality Assurance Configuration](#quality-assurance-configuration)
- [Storage and Caching](#storage-and-caching)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)
- [Environment-Specific Configurations](#environment-specific-configurations)

---

## Configuration Overview

### Configuration Files

APEX DIRECTOR uses multiple configuration files:

```
apex_director/
├── config/
│   ├── apex_config.json          # Main system configuration
│   ├── backends.json             # Backend service configurations
│   ├── quality_presets.json      # Quality presets and thresholds
│   └── user_preferences.json     # User-specific settings
├── style_bibles/
│   └── default_style.json       # Default style bible
└── plugins/
    └── custom_filters.json      # Custom plugin configurations
```

### Configuration Methods

1. **File-based**: JSON configuration files
2. **Environment variables**: For sensitive data
3. **Command-line arguments**: For runtime overrides
4. **Programmatic**: Direct API configuration

### Loading Configuration

```python
from apex_director.core.config import ConfigManager

# Load from default location
config = ConfigManager.load()

# Load from specific file
config = ConfigManager.load("/path/to/config.json")

# Load from environment
config = ConfigManager.load_from_env()

# Create new configuration
config = ConfigManager.create_default()
```

---

## System Configuration

### Main System Settings

```json
{
  "system": {
    "name": "APEX DIRECTOR",
    "version": "1.0.0",
    "debug_mode": false,
    "log_level": "INFO",
    "log_file": "/var/log/apex_director/app.log",
    "max_log_size_mb": 100,
    "log_rotation": "daily",
    "temp_directory": "/tmp/apex_director",
    "cache_directory": "/var/cache/apex_director",
    "data_directory": "/var/lib/apex_director"
  },
  "orchestrator": {
    "max_concurrent_jobs": 5,
    "job_timeout_seconds": 3600,
    "checkpoint_interval_seconds": 300,
    "auto_retry": true,
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "health_check_interval_seconds": 60,
    "shutdown_timeout_seconds": 30
  },
  "asset_manager": {
    "auto_cleanup": true,
    "cleanup_interval_hours": 24,
    "max_cache_size_gb": 10,
    "compression_enabled": true,
    "deduplication_enabled": true,
    "backup_enabled": true,
    "backup_interval_hours": 168,
    "max_backup_count": 7
  },
  "performance": {
    "enable_parallel_processing": true,
    "thread_pool_size": 4,
    "memory_limit_mb": 8192,
    "disk_io_limit_mb": 100,
    "network_timeout_seconds": 30,
    "connection_pool_size": 10
  }
}
```

### Configuration Options

#### Orchestrator Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_concurrent_jobs` | int | 5 | Maximum number of jobs to process simultaneously |
| `job_timeout_seconds` | int | 3600 | Maximum time for job completion (1 hour) |
| `checkpoint_interval_seconds` | int | 300 | Automatic checkpoint creation interval (5 minutes) |
| `auto_retry` | bool | true | Enable automatic retry for failed jobs |
| `retry_attempts` | int | 3 | Number of retry attempts for failed jobs |
| `retry_delay_seconds` | int | 5 | Delay between retry attempts |
| `health_check_interval_seconds` | int | 60 | System health check frequency |
| `shutdown_timeout_seconds` | int | 30 | Time to wait for graceful shutdown |

#### Asset Manager Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `auto_cleanup` | bool | true | Enable automatic cleanup of temporary files |
| `cleanup_interval_hours` | int | 24 | Cleanup frequency |
| `max_cache_size_gb` | int | 10 | Maximum cache size in GB |
| `compression_enabled` | bool | true | Enable file compression |
| `deduplication_enabled` | bool | true | Enable duplicate file detection |
| `backup_enabled` | bool | true | Enable automatic backups |
| `backup_interval_hours` | int | 168 | Backup frequency (1 week) |
| `max_backup_count` | int | 7 | Maximum number of backups to retain |

---

## Backend Configuration

### Multi-Backend Setup

```json
{
  "backends": {
    "nano_banana": {
      "enabled": true,
      "priority": 1,
      "api_endpoint": "https://api.nanobanana.com/v1",
      "api_key": "${NANO_BANANA_API_KEY}",
      "timeout_seconds": 60,
      "max_requests_per_minute": 100,
      "retry_attempts": 3,
      "cost_per_image": 0.01,
      "quality_level": 1,
      "capabilities": {
        "max_resolution": [512, 512],
        "supported_formats": ["png", "jpg"],
        "style_presets": ["realistic", "artistic", "fantasy"]
      }
    },
    "imagen": {
      "enabled": true,
      "priority": 2,
      "api_endpoint": "https://vision.googleapis.com/v1",
      "project_id": "${GOOGLE_PROJECT_ID}",
      "api_key": "${GOOGLE_API_KEY}",
      "timeout_seconds": 90,
      "max_requests_per_minute": 50,
      "cost_per_image": 0.05,
      "quality_level": 3,
      "capabilities": {
        "max_resolution": [1024, 1024],
        "supported_formats": ["png", "jpg"],
        "style_presets": ["photographic", "artistic", "vintage"]
      }
    },
    "minimax": {
      "enabled": true,
      "priority": 3,
      "api_endpoint": "https://api.minimax.chat/v1",
      "api_key": "${MINIMAX_API_KEY}",
      "group_id": "${MINIMAX_GROUP_ID}",
      "timeout_seconds": 120,
      "max_requests_per_minute": 30,
      "cost_per_image": 0.08,
      "quality_level": 4,
      "capabilities": {
        "max_resolution": [2048, 2048],
        "supported_formats": ["png", "jpg", "webp"],
        "style_presets": ["cinematic", "professional", "artistic"]
      }
    },
    "sdxl": {
      "enabled": false,
      "priority": 4,
      "api_endpoint": "https://api.stability.ai/v1",
      "api_key": "${STABILITY_API_KEY}",
      "timeout_seconds": 180,
      "max_requests_per_minute": 20,
      "cost_per_image": 0.12,
      "quality_level": 5,
      "capabilities": {
        "max_resolution": [2048, 2048],
        "supported_formats": ["png", "jpg"],
        "style_presets": ["realistic", "artistic", "fantasy", "cyberpunk"]
      }
    }
  },
  "backend_fallback": {
    "enabled": true,
    "max_fallback_attempts": 3,
    "fallback_delay_seconds": 2,
    "health_check_enabled": true,
    "circuit_breaker": {
      "failure_threshold": 5,
      "recovery_timeout": 60,
      "half_open_max_calls": 3
    }
  }
}
```

### Backend-Specific Configuration

#### Nano Banana Backend
```json
{
  "nano_banana": {
    "api_endpoint": "https://api.nanobanana.com/v1/generate",
    "api_key": "${NANO_BANANA_API_KEY}",
    "authentication_type": "bearer_token",
    "model": "nano-banana-v1",
    "generation_parameters": {
      "width": 512,
      "height": 512,
      "steps": 20,
      "guidance_scale": 7.5,
      "seed": null
    },
    "rate_limiting": {
      "requests_per_minute": 100,
      "requests_per_hour": 1000,
      "concurrent_requests": 5
    }
  }
}
```

#### Google Imagen Backend
```json
{
  "imagen": {
    "project_id": "${GOOGLE_PROJECT_ID}",
    "location": "us-central1",
    "endpoint": "us-central1-aiplatform.googleapis.com",
    "api_key": "${GOOGLE_API_KEY}",
    "model": "imagegeneration@002",
    "generation_parameters": {
      "aspect_ratio": "1:1",
      "guidance_scale": 10.0,
      "safety_filter_level": "block_some",
      "person_filter_level": "block_adult"
    },
    "rate_limiting": {
      "requests_per_minute": 50,
      "requests_per_hour": 500,
      "concurrent_requests": 3
    }
  }
}
```

---

## Image Generation Configuration

### Generation Settings

```json
{
  "image_generation": {
    "default_quality": "high",
    "default_resolution": [1024, 1024],
    "default_format": "png",
    "quality_presets": {
      "draft": {
        "steps": 20,
        "guidance_scale": 7.0,
        "quality_level": 1,
        "cost_multiplier": 0.5
      },
      "web": {
        "steps": 30,
        "guidance_scale": 8.0,
        "quality_level": 2,
        "cost_multiplier": 1.0
      },
      "high": {
        "steps": 50,
        "guidance_scale": 9.0,
        "quality_level": 3,
        "cost_multiplier": 1.5
      },
      "broadcast": {
        "steps": 75,
        "guidance_scale": 10.0,
        "quality_level": 4,
        "cost_multiplier": 2.0
      },
      "cinema": {
        "steps": 100,
        "guidance_scale": 11.0,
        "quality_level": 5,
        "cost_multiplier": 3.0
      }
    },
    "prompt_engineering": {
      "enable_automatic_enhancement": true,
      "cinematic_prompts": true,
      "director_styles": true,
      "camera_settings": true,
      "lighting_descriptions": true
    },
    "style_consistency": {
      "enabled": true,
      "drift_tolerance": 0.15,
      "comparison_threshold": 0.8,
      "style_bible_path": "/etc/apex_director/style_bibles/default.json"
    },
    "character_consistency": {
      "enabled": true,
      "similarity_threshold": 0.85,
      "validation_levels": ["facial_features", "body_pose", "clothing"],
      "reference_image_count": 5
    }
  }
}
```

### Style Bible Configuration

```json
{
  "style_bible": {
    "project_name": "Default Style",
    "overall_style": {
      "visual_style": "cinematic realism",
      "mood": "dramatic",
      "color_grading": "warm with high contrast"
    },
    "color_palette": {
      "primary_colors": ["#2C3E50", "#ECF0F1", "#E74C3C"],
      "secondary_colors": ["#3498DB", "#F39C12", "#9B59B6"],
      "neutral_colors": ["#34495E", "#7F8C8D", "#BDC3C7"],
      "skin_tones": ["#FDBCB4", "#F1C27D", "#E0AC69", "#C68642", "#8D5524"]
    },
    "lighting_setup": {
      "key_light": {
        "type": "soft_box",
        "position": "camera_left",
        "intensity": 0.8,
        "color_temperature": 5600
      },
      "fill_light": {
        "type": "reflector",
        "position": "camera_right",
        "intensity": 0.4,
        "color_temperature": 5600
      },
      "rim_light": {
        "type": "spotlight",
        "position": "behind_subject",
        "intensity": 0.6,
        "color_temperature": 3200
      }
    },
    "camera_profile": {
      "preferred_lenses": ["35mm", "50mm", "85mm"],
      "aperture_range": "f/2.8 to f/5.6",
      "shutter_speed_range": "1/60 to 1/125",
      "iso_range": "ISO 100 to ISO 800",
      "composition_rules": ["rule_of_thirds", "leading_lines", "symmetry"]
    },
    "genre_settings": {
      "cinematic": {
        "color_grading": "cinematic_teal_orange",
        "lighting_style": "dramatic_three_point",
        "composition": "wide_shots_with_depth"
      },
      "portrait": {
        "color_grading": "natural_skin_tones",
        "lighting_style": "soft_portrait_lighting",
        "composition": "shallow_depth_of_field"
      },
      "landscape": {
        "color_grading": "natural_colors",
        "lighting_style": "golden_hour",
        "composition": "rule_of_thirds_horizon"
      }
    }
  }
}
```

---

## Video Assembly Configuration

### Video Assembly Settings

```json
{
  "video_assembly": {
    "default_resolution": {
      "width": 1920,
      "height": 1080,
      "aspect_ratio": "16:9"
    },
    "default_frame_rate": 30,
    "default_codec": "h264",
    "default_bitrate": "10mbps",
    "audio_settings": {
      "sample_rate": 48000,
      "bit_depth": 16,
      "channels": 2,
      "codec": "aac",
      "bitrate": "320kbps"
    },
    "transition_presets": {
      "cut": {
        "duration": 0.0,
        "type": "hard_cut"
      },
      "crossfade": {
        "duration": 1.0,
        "type": "linear_crossfade"
      },
      "whip_pan": {
        "duration": 0.5,
        "type": "motion_blur"
      },
      "match_dissolve": {
        "duration": 1.5,
        "type": "color_matched"
      }
    },
    "color_grading": {
      "enabled": true,
      "stages": {
        "stage_1_primary": {
          "exposure": 0.0,
          "contrast": 15.0,
          "saturation": 10.0,
          "brightness": 5.0
        },
        "stage_2_secondary": {
          "skin_tone_balance": true,
          "selective_desaturation": false,
          "color_wheels": {
            "shadows": {"r": 0, "g": 0, "b": 0},
            "midtones": {"r": 0, "g": 0, "b": 0},
            "highlights": {"r": 0, "g": 0, "b": 0}
          }
        },
        "stage_3_creative": {
          "lut_file": "/etc/apex_director/luts/cinematic.cube",
          "lut_strength": 1.0,
          "vignette": {"enabled": false, "amount": 0.0},
          "film_grain": {"enabled": false, "intensity": 0.0}
        },
        "stage_4_finishing": {
          "sharpening": 0.3,
          "noise_reduction": 0.2,
          "chromatic_aberration": 0.0,
          "lens_distortion": 0.0
        }
      }
    },
    "motion_effects": {
      "ken_burns": {
        "enabled": true,
        "max_zoom_factor": 1.5,
        "smoothing": "ease_in_out"
      },
      "dolly_zoom": {
        "enabled": false,
        "zoom_intensity": 1.3,
        "trigger_sections": ["chorus"]
      },
      "parallax": {
        "enabled": false,
        "layers": 3,
        "depth_factor": 0.8
      }
    }
  }
}
```

### Export Format Presets

```json
{
  "export_presets": {
    "broadcast_hd": {
      "video": {
        "codec": "h264",
        "profile": "high",
        "level": 4.0,
        "resolution": [1920, 1080],
        "frame_rate": 29.97,
        "bitrate": "10mbps",
        "gop_size": 30,
        "b_frames": 3
      },
      "audio": {
        "codec": "aac",
        "sample_rate": 48000,
        "bit_depth": 16,
        "channels": 2,
        "bitrate": "320kbps"
      },
      "container": "mp4",
      "compliance": "broadcast_strict"
    },
    "web_optimized": {
      "video": {
        "codec": "h264",
        "profile": "main",
        "level": 3.1,
        "resolution": [1920, 1080],
        "frame_rate": 30,
        "bitrate": "5mbps",
        "two_pass": false
      },
      "audio": {
        "codec": "aac",
        "sample_rate": 44100,
        "bit_depth": 16,
        "channels": 2,
        "bitrate": "128kbps"
      },
      "container": "mp4",
      "streaming": true
    },
    "cinema_4k": {
      "video": {
        "codec": "h265",
        "profile": "main",
        "level": 5.0,
        "resolution": [3840, 2160],
        "frame_rate": 24,
        "bitrate": "25mbps",
        "two_pass": true,
        "color_space": "rec2020"
      },
      "audio": {
        "codec": "pcm",
        "sample_rate": 48000,
        "bit_depth": 24,
        "channels": 6,
        "bitrate": "4608kbps"
      },
      "container": "mov",
      "compliance": "dci_compliant"
    }
  }
}
```

---

## Audio Processing Configuration

### Audio Analysis Settings

```json
{
  "audio_processing": {
    "analysis": {
      "sample_rate": 44100,
      "fft_size": 2048,
      "hop_length": 512,
      "window_function": "hann"
    },
    "beat_detection": {
      "algorithm": "dynamic_programming",
      "sensitivity": 0.5,
      "minimum_bpm": 60,
      "maximum_bpm": 180,
      "confidence_threshold": 0.7
    },
    "tempo_estimation": {
      "method": "librosa",
      "trim_edges": true,
      "update_backend": true
    },
    "key_detection": {
      "method": "chroma",
      "pitch_class_profile": "default",
      "smooth_transitions": true
    },
    "section_detection": {
      "min_section_length": 5.0,
      "max_section_length": 60.0,
      "change_threshold": 0.5,
      "merge_similar": true
    },
    "spectral_analysis": {
      "compute_rms": true,
      "compute_spectral_centroid": true,
      "compute_zero_crossing_rate": true,
      "compute_mfcc": true,
      "mfcc_coefficients": 13
    }
  }
}
```

### Audio Synchronization

```json
{
  "audio_sync": {
    "frame_accuracy": true,
    "max_timing_error_frames": 1,
    "beat_lock_cutting": true,
    "rhythm_analysis": {
      "note_value_detection": true,
      "accent_detection": true,
      "syncopation_analysis": false
    },
    "automated_cutting": {
      "verse_cut_tolerance": 0.1,
      "chorus_cut_tolerance": 0.05,
      "bridge_cut_tolerance": 0.2,
      "enable_transition_effects": true
    }
  }
}
```

---

## Quality Assurance Configuration

### Quality Standards

```json
{
  "quality_assurance": {
    "broadcast_standards": {
      "enabled": true,
      "video_standards": {
        "min_resolution": [720, 480],
        "max_resolution": [4096, 2160],
        "supported_frame_rates": [23.976, 24, 25, 29.97, 30, 50, 59.94, 60],
        "color_spaces": ["rec709", "rec2020"],
        "max_bitrate_mbps": 50,
        "min_bitrate_mbps": 1
      },
      "audio_standards": {
        "min_sample_rate": 44100,
        "supported_sample_rates": [44100, 48000],
        "max_loudness_lufs": -23.0,
        "min_loudness_lufs": -30.0,
        "peak_level_db": -3.0
      }
    },
    "quality_scoring": {
      "overall_weight": 1.0,
      "criteria": {
        "visual_quality": {
          "weight": 0.4,
          "factors": ["sharpness", "noise", "artifacts", "composition"]
        },
        "technical_quality": {
          "weight": 0.3,
          "factors": ["resolution", "compression", "format_compliance"]
        },
        "style_consistency": {
          "weight": 0.2,
          "factors": ["color_harmony", "lighting_consistency", "composition_style"]
        },
        "audio_sync": {
          "weight": 0.1,
          "factors": ["beat_sync", "section_alignment", "timing_accuracy"]
        }
      },
      "thresholds": {
        "exceptional": 0.9,
        "high": 0.8,
        "good": 0.7,
        "acceptable": 0.6,
        "poor": 0.5
      }
    },
    "artifact_detection": {
      "enabled": true,
      "check_types": ["noise", "banding", "blocking", "ghosting", "ringing"],
      "sensitivity": "medium",
      "automatic_correction": false
    }
  }
}
```

### Quality Control Workflows

```json
{
  "qc_workflows": {
    "automated_qc": {
      "enabled": true,
      "stages": [
        "technical_validation",
        "quality_scoring",
        "broadcast_compliance",
        "final_approval"
      ],
      "fail_fast": false,
      "retry_failed_checks": true,
      "retry_attempts": 2
    },
    "manual_review": {
      "enabled": false,
      "triggers": ["low_quality_score", "compliance_failure"],
      "review_queue": "/var/spool/apex_director/review",
      "notification_email": "qc@company.com"
    },
    "continuous_monitoring": {
      "enabled": true,
      "metrics_collection": true,
      "trend_analysis": true,
      "alert_thresholds": {
        "quality_score_drop": 0.1,
        "failure_rate_increase": 0.05,
        "processing_time_increase": 0.2
      }
    }
  }
}
```

---

## Storage and Caching

### Storage Configuration

```json
{
  "storage": {
    "primary_storage": {
      "type": "local",
      "path": "/var/lib/apex_director/storage",
      "max_size_gb": 1000,
      "compression": {
        "enabled": true,
        "algorithm": "gzip",
        "compression_level": 6
      }
    },
    "cache_storage": {
      "type": "redis",
      "host": "localhost",
      "port": 6379,
      "password": "${REDIS_PASSWORD}",
      "db": 0,
      "max_memory_mb": 512,
      "eviction_policy": "allkeys-lru"
    },
    "backup_storage": {
      "type": "s3",
      "bucket": "apex-director-backups",
      "region": "us-east-1",
      "access_key": "${S3_ACCESS_KEY}",
      "secret_key": "${S3_SECRET_KEY}",
      "encryption": "aws-kms",
      "retention_days": 30
    },
    "asset_organization": {
      "hierarchical": true,
      "date_based": true,
      "project_based": true,
      "deduplication": true,
      "hash_verification": true
    }
  }
}
```

### Caching Strategy

```json
{
  "caching": {
    "image_cache": {
      "enabled": true,
      "max_size_mb": 2048,
      "ttl_seconds": 86400,
      "compression": true,
      "storage_path": "/var/cache/apex_director/images"
    },
    "metadata_cache": {
      "enabled": true,
      "max_size_mb": 512,
      "ttl_seconds": 3600,
      "storage_path": "/var/cache/apex_director/metadata"
    },
    "audio_cache": {
      "enabled": true,
      "max_size_mb": 1024,
      "ttl_seconds": 7200,
      "storage_path": "/var/cache/apex_director/audio"
    },
    "generation_cache": {
      "enabled": true,
      "max_size_mb": 4096,
      "ttl_seconds": 604800,
      "storage_path": "/var/cache/apex_director/generations"
    }
  }
}
```

---

## Security Configuration

### Security Settings

```json
{
  "security": {
    "authentication": {
      "enabled": true,
      "method": "jwt",
      "secret_key": "${JWT_SECRET_KEY}",
      "expiration_hours": 24,
      "refresh_enabled": true
    },
    "authorization": {
      "enabled": true,
      "rbac": true,
      "default_role": "user",
      "roles": {
        "admin": ["read", "write", "execute", "configure"],
        "operator": ["read", "write", "execute"],
        "user": ["read", "execute"],
        "viewer": ["read"]
      }
    },
    "data_encryption": {
      "at_rest": true,
      "in_transit": true,
      "algorithm": "AES-256",
      "key_rotation_days": 90
    },
    "input_validation": {
      "enabled": true,
      "file_size_limit_mb": 100,
      "file_type_restrictions": true,
      "path_traversal_protection": true,
      "xss_protection": true
    },
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 100,
      "burst_limit": 20,
      "whitelist_ips": []
    },
    "audit_logging": {
      "enabled": true,
      "log_file": "/var/log/apex_director/security.log",
      "events": ["login", "logout", "file_access", "job_submission", "configuration_change"],
      "retention_days": 90
    }
  }
}
```

### API Security

```json
{
  "api_security": {
    "cors": {
      "enabled": true,
      "allowed_origins": ["https://app.company.com"],
      "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
      "allowed_headers": ["Authorization", "Content-Type"],
      "max_age_seconds": 86400
    },
    "csrf": {
      "enabled": true,
      "token_header": "X-CSRF-Token"
    },
    "api_keys": {
      "enabled": true,
      "rotation_days": 30,
      "scope_restrictions": true,
      "rate_limiting": true
    }
  }
}
```

---

## Performance Tuning

### System Performance

```json
{
  "performance": {
    "parallel_processing": {
      "enabled": true,
      "max_workers": 8,
      "thread_pool_size": 4,
      "async_concurrency": 10
    },
    "memory_management": {
      "max_heap_mb": 8192,
      "gc_threshold": 700,
      "object_cache_size": 1000,
      "memory_monitoring": true
    },
    "disk_io": {
      "max_io_mb": 200,
      "async_io": true,
      "buffer_size_kb": 64,
      "direct_io": false
    },
    "network": {
      "connection_pool_size": 20,
      "keep_alive_seconds": 60,
      "timeout_seconds": 30,
      "retry_attempts": 3
    },
    "database": {
      "connection_pool_size": 10,
      "query_timeout_seconds": 30,
      "cache_size": 1000,
      "wal_mode": true
    }
  }
}
```

### Optimization Profiles

```json
{
  "optimization_profiles": {
    "development": {
      "max_concurrent_jobs": 2,
      "log_level": "DEBUG",
      "cache_disabled": false,
      "parallel_processing": false
    },
    "production_small": {
      "max_concurrent_jobs": 4,
      "max_memory_mb": 4096,
      "parallel_processing": true,
      "cache_enabled": true
    },
    "production_large": {
      "max_concurrent_jobs": 16,
      "max_memory_mb": 16384,
      "parallel_processing": true,
      "cache_enabled": true,
      "load_balancing": true
    },
    "high_performance": {
      "max_concurrent_jobs": 32,
      "max_memory_mb": 32768,
      "parallel_processing": true,
      "gpu_acceleration": true,
      "optimized_algorithms": true
    }
  }
}
```

---

## Environment-Specific Configurations

### Development Environment

```json
{
  "environment": "development",
  "system": {
    "debug_mode": true,
    "log_level": "DEBUG",
    "temp_directory": "/tmp/apex_director_dev"
  },
  "orchestrator": {
    "max_concurrent_jobs": 2,
    "auto_retry": false
  },
  "backends": {
    "test_backend": {
      "enabled": true,
      "api_endpoint": "http://localhost:8080",
      "mock_mode": true
    }
  },
  "quality_assurance": {
    "strict_validation": false,
    "automated_qc": false
  }
}
```

### Production Environment

```json
{
  "environment": "production",
  "system": {
    "debug_mode": false,
    "log_level": "INFO",
    "log_rotation": "daily",
    "temp_directory": "/var/tmp/apex_director",
    "max_log_size_mb": 100
  },
  "orchestrator": {
    "max_concurrent_jobs": 8,
    "auto_retry": true,
    "retry_attempts": 3,
    "checkpoint_interval_seconds": 300
  },
  "security": {
    "authentication": {"enabled": true},
    "authorization": {"enabled": true},
    "audit_logging": {"enabled": true}
  },
  "quality_assurance": {
    "broadcast_standards": {"enabled": true},
    "automated_qc": {"enabled": true}
  }
}
```

### Docker Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  apex_director:
    build: .
    environment:
      - APEX_ENVIRONMENT=production
      - NANO_BANANA_API_KEY=${NANO_BANANA_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    volumes:
      - ./config:/etc/apex_director:ro
      - apex_data:/var/lib/apex_director
      - apex_logs:/var/log/apex_director
    ports:
      - "8080:8080"
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=apex_director
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  apex_data:
  apex_logs:
  redis_data:
  postgres_data:
```

### Kubernetes Configuration

```yaml
# apex-director-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apex-director
  labels:
    app: apex-director
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apex-director
  template:
    metadata:
      labels:
        app: apex-director
    spec:
      containers:
      - name: apex-director
        image: apex-director:latest
        ports:
        - containerPort: 8080
        env:
        - name: APEX_ENVIRONMENT
          value: "kubernetes"
        - name: NANO_BANANA_API_KEY
          valueFrom:
            secretKeyRef:
              name: apex-secrets
              key: nano-banana-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: config-volume
          mountPath: /etc/apex_director
      volumes:
      - name: config-volume
        configMap:
          name: apex-config
---
apiVersion: v1
kind: Service
metadata:
  name: apex-director-service
spec:
  selector:
    app: apex-director
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

This configuration reference provides comprehensive guidance for setting up APEX DIRECTOR in any environment, from development to production, with proper security, performance, and reliability settings.

*For specific configuration questions, consult the [Troubleshooting Guide](troubleshooting.md) or the [community resources](../README.md#community).*
