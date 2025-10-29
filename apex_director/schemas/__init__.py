"""
JSON Schemas for APEX DIRECTOR Data Structures
Define the structure and validation for all data types used throughout the system
"""

# Job-related schemas
JOB_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "prompt": {"type": "string", "minLength": 1},
        "negative_prompt": {"type": "string"},
        "backend_preference": {"type": "string"},
        "priority": {"type": "integer", "minimum": 1, "maximum": 5},
        "estimated_cost": {"type": "number", "minimum": 0},
        "estimated_time": {"type": "integer", "minimum": 0},
        "metadata": {
            "type": "object",
            "properties": {
                "project_name": {"type": "string"},
                "batch_id": {"type": "string"},
                "user_id": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "source": {"type": "string"},
                "created_by": {"type": "string"}
            }
        },
        "generation_params": {
            "type": "object",
            "properties": {
                "width": {"type": "integer", "minimum": 64, "maximum": 2048},
                "height": {"type": "integer", "minimum": 64, "maximum": 2048},
                "steps": {"type": "integer", "minimum": 1, "maximum": 100},
                "guidance_scale": {"type": "number", "minimum": 1, "maximum": 20},
                "seed": {"type": "integer"},
                "model_variant": {"type": "string"},
                "style_preset": {"type": "string"}
            }
        }
    },
    "required": ["id", "prompt"]
}

JOB_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["pending", "queued", "processing", "completed", "failed", "cancelled"]
        },
        "progress": {"type": "number", "minimum": 0, "maximum": 100},
        "backend_used": {"type": "string"},
        "started_at": {"type": "string", "format": "date-time"},
        "completed_at": {"type": "string", "format": "date-time"},
        "error_message": {"type": "string"},
        "retry_count": {"type": "integer", "minimum": 0},
        "actual_cost": {"type": "number", "minimum": 0},
        "actual_time": {"type": "integer", "minimum": 0}
    },
    "required": ["job_id", "status", "progress"]
}

# Asset-related schemas
ASSET_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "filename": {"type": "string"},
        "file_path": {"type": "string"},
        "file_size": {"type": "integer", "minimum": 0},
        "file_hash": {"type": "string", "minLength": 64},
        "format": {"type": "string"},
        "width": {"type": "integer", "minimum": 0},
        "height": {"type": "integer", "minimum": 0},
        "created_at": {"type": "string", "format": "date-time"},
        "job_id": {"type": "string"},
        "prompt": {"type": "string"},
        "backend_used": {"type": "string"},
        "generation_params": {
            "type": "object",
            "properties": {
                "steps": {"type": "integer"},
                "guidance_scale": {"type": "number"},
                "seed": {"type": "integer"},
                "model": {"type": "string"},
                "version": {"type": "string"}
            }
        },
        "tags": {"type": "array", "items": {"type": "string"}},
        "quality_score": {"type": "number", "minimum": 0, "maximum": 10},
        "usage_count": {"type": "integer", "minimum": 0},
        "last_used": {"type": "string", "format": "date-time"}
    },
    "required": ["id", "filename", "file_path", "file_hash", "format", "created_at"]
}

PROJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "created_at": {"type": "string", "format": "date-time"},
        "updated_at": {"type": "string", "format": "date-time"},
        "status": {
            "type": "string",
            "enum": ["active", "archived", "completed", "failed"]
        },
        "total_jobs": {"type": "integer", "minimum": 0},
        "completed_jobs": {"type": "integer", "minimum": 0},
        "total_cost": {"type": "number", "minimum": 0},
        "settings": {
            "type": "object",
            "properties": {
                "default_backend": {"type": "string"},
                "max_cost_per_job": {"type": "number", "minimum": 0},
                "auto_cleanup": {"type": "boolean"},
                "backup_enabled": {"type": "boolean"}
            }
        }
    },
    "required": ["id", "name", "created_at", "status"]
}

# Checkpoint-related schemas
CHECKPOINT_DATA_SCHEMA = {
    "type": "object",
    "properties": {
        "checkpoint_id": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "version": {"type": "string"},
        "system_state": {
            "type": "object",
            "properties": {
                "active_jobs": {"type": "array", "items": {"type": "string"}},
                "queued_jobs": {"type": "array", "items": {"type": "string"}},
                "failed_jobs": {"type": "array", "items": {"type": "string"}},
                "completed_jobs": {"type": "array", "items": {"type": "string"}},
                "backend_status": {
                    "type": "object",
                    "properties": {
                        "backend_name": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "last_health_check": {"type": "string", "format": "date-time"},
                                "request_count": {"type": "integer"},
                                "error_count": {"type": "integer"}
                            }
                        }
                    }
                }
            }
        },
        "job_states": {
            "type": "object",
            "properties": {
                "job_id": JOB_STATUS_SCHEMA
            }
        },
        "asset_inventory": {
            "type": "object",
            "properties": {
                "total_assets": {"type": "integer"},
                "asset_breakdown": {
                    "type": "object",
                    "properties": {
                        "format": {"type": "integer"}
                    }
                },
                "total_size_mb": {"type": "number"}
            }
        }
    },
    "required": ["checkpoint_id", "timestamp", "version", "system_state"]
}

# Estimation schemas
COST_ESTIMATE_SCHEMA = {
    "type": "object",
    "properties": {
        "estimate_id": {"type": "string"},
        "job_params": {
            "type": "object",
            "properties": {
                "backend": {"type": "string"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "steps": {"type": "integer"},
                "quality_level": {"type": "integer", "minimum": 1, "maximum": 5}
            }
        },
        "estimated_cost": {"type": "number", "minimum": 0},
        "estimated_time_seconds": {"type": "integer", "minimum": 0},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        "factors": {
            "type": "object",
            "properties": {
                "backend_load": {"type": "number"},
                "image_complexity": {"type": "number"},
                "queue_length": {"type": "integer"},
                "time_of_day": {"type": "number"}
            }
        },
        "created_at": {"type": "string", "format": "date-time"},
        "expires_at": {"type": "string", "format": "date-time"}
    },
    "required": ["estimate_id", "job_params", "estimated_cost", "created_at"]
}

# Backend-related schemas
BACKEND_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "image_data": {"type": "string", "description": "Base64 encoded image data"},
        "image_url": {"type": "string", "description": "Direct URL to generated image"},
        "metadata": {
            "type": "object",
            "properties": {
                "generation_time": {"type": "number"},
                "model_used": {"type": "string"},
                "steps_used": {"type": "integer"},
                "seed": {"type": "integer"},
                "cost": {"type": "number"},
                "quality_metrics": {
                    "type": "object",
                    "properties": {
                        "sharpness": {"type": "number"},
                        "contrast": {"type": "number"},
                        "color_balance": {"type": "number"}
                    }
                }
            }
        },
        "error": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "details": {"type": "object"}
            }
        }
    },
    "required": ["success"]
}

BACKEND_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "status": {"type": "string", "enum": ["online", "offline", "maintenance", "overloaded"]},
        "last_health_check": {"type": "string", "format": "date-time"},
        "response_time_ms": {"type": "integer", "minimum": 0},
        "success_rate": {"type": "number", "minimum": 0, "maximum": 1},
        "queue_length": {"type": "integer", "minimum": 0},
        "rate_limit_remaining": {"type": "integer", "minimum": 0},
        "error_count_24h": {"type": "integer", "minimum": 0},
        "capabilities": {"type": "array", "items": {"type": "string"}},
        "current_load": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["name", "status", "last_health_check"]
}

# System health schema
SYSTEM_HEALTH_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_status": {"type": "string", "enum": ["healthy", "degraded", "critical"]},
        "timestamp": {"type": "string", "format": "date-time"},
        "component_status": {
            "type": "object",
            "properties": {
                "orchestrator": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "active_jobs": {"type": "integer"},
                        "queued_jobs": {"type": "integer"},
                        "uptime_seconds": {"type": "integer"}
                    }
                },
                "backends": {
                    "type": "object",
                    "properties": {
                        "backend_name": BACKEND_STATUS_SCHEMA
                    }
                },
                "assets": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "total_files": {"type": "integer"},
                        "total_size_mb": {"type": "number"},
                        "cache_hit_rate": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "storage": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "available_space_gb": {"type": "number"},
                        "used_space_gb": {"type": "number"},
                        "disk_usage_percent": {"type": "number", "minimum": 0, "maximum": 100}
                    }
                }
            }
        },
        "performance_metrics": {
            "type": "object",
            "properties": {
                "jobs_per_hour": {"type": "number"},
                "average_job_time": {"type": "number"},
                "cost_per_job": {"type": "number"},
                "success_rate": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },
        "alerts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["info", "warning", "error", "critical"]},
                    "message": {"type": "string"},
                    "component": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"}
                },
                "required": ["severity", "message", "timestamp"]
            }
        }
    },
    "required": ["overall_status", "timestamp", "component_status"]
}

# Export schema mappings for easy access
SCHEMAS = {
    "job_request": JOB_REQUEST_SCHEMA,
    "job_status": JOB_STATUS_SCHEMA,
    "asset_metadata": ASSET_METADATA_SCHEMA,
    "project": PROJECT_SCHEMA,
    "checkpoint_data": CHECKPOINT_DATA_SCHEMA,
    "cost_estimate": COST_ESTIMATE_SCHEMA,
    "backend_response": BACKEND_RESPONSE_SCHEMA,
    "backend_status": BACKEND_STATUS_SCHEMA,
    "system_health": SYSTEM_HEALTH_SCHEMA
}