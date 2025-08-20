"""
Configuration validation and documentation utilities for fishtools.

Provides tools for validating configuration parameters and generating
comprehensive documentation about available options.
"""

from pathlib import Path
from typing import Any, Dict, List, Type
import json

from pydantic import BaseModel, ValidationError

from .config import (
    Config,
    HardwareConfig, 
    DeconvolutionConfig,
    BasicConfig,
    StitchingConfig,
    SpotAnalysisConfig,
    SystemConfig,
    ImageProcessingConfig,
    RegisterConfig,
    Fiducial,
    ChannelConfig,
)


class ConfigValidator:
    """Configuration validation utilities."""
    
    @staticmethod
    def validate_config_dict(config_dict: dict[str, Any]) -> list[str]:
        """Validate a configuration dictionary and return any errors.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        try:
            Config(**config_dict)
        except ValidationError as e:
            for error in e.errors():
                field_path = " -> ".join(str(x) for x in error["loc"])
                message = error["msg"]
                errors.append(f"{field_path}: {message}")
        
        return errors
    
    @staticmethod
    def validate_toml_file(config_path: Path) -> list[str]:
        """Validate a TOML configuration file.
        
        Args:
            config_path: Path to TOML configuration file
            
        Returns:
            List of validation error messages
        """
        import toml
        
        try:
            config_dict = toml.load(config_path)
            return ConfigValidator.validate_config_dict(config_dict)
        except Exception as e:
            return [f"Failed to load TOML file: {e}"]
    
    @staticmethod
    def suggest_fixes(config_dict: dict[str, Any]) -> dict[str, list[str]]:
        """Suggest fixes for common configuration issues.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Dictionary mapping field paths to suggested fixes
        """
        suggestions = {}
        
        # Check for common path issues
        if "dataPath" in config_dict:
            data_path = Path(config_dict["dataPath"])
            if not data_path.exists():
                suggestions["dataPath"] = [
                    f"Path {data_path} does not exist",
                    "Consider using an absolute path",
                    "Ensure the data directory is accessible"
                ]
        
        # Check for chromatic shift files
        if "registration" in config_dict and "chromatic_shifts" in config_dict["registration"]:
            for channel, shift_path in config_dict["registration"]["chromatic_shifts"].items():
                if not Path(shift_path).exists():
                    suggestions[f"registration.chromatic_shifts.{channel}"] = [
                        f"Chromatic shift file {shift_path} not found",
                        "Check if the file path is correct",
                        "Generate calibration files if missing"
                    ]
        
        # Check for reasonable hardware limits
        if "hardware" in config_dict:
            hw = config_dict["hardware"]
            if "max_memory_mb" in hw and hw["max_memory_mb"] > 500000:
                suggestions["hardware.max_memory_mb"] = [
                    "Memory allocation seems very high",
                    "Consider reducing to avoid system instability"
                ]
            
            if "parallel_threads" in hw and hw["parallel_threads"] > 64:
                suggestions["hardware.parallel_threads"] = [
                    "Thread count seems very high",
                    "Consider using fewer threads to avoid contention"
                ]
        
        return suggestions


class ConfigDocGenerator:
    """Generate comprehensive documentation for configuration options."""
    
    @staticmethod
    def generate_field_docs(model_class: type[BaseModel]) -> dict[str, dict[str, Any]]:
        """Generate documentation for all fields in a Pydantic model.
        
        Args:
            model_class: Pydantic model class
            
        Returns:
            Dictionary with field documentation
        """
        docs = {}
        
        for field_name, field_info in model_class.model_fields.items():
            field_type = field_info.annotation
            
            # Extract default value
            default = field_info.default
            if hasattr(default, '__name__') and default.__name__ == 'default_factory':
                # Get default from factory function if available
                try:
                    default = default()
                    if hasattr(default, 'model_dump'):
                        default = default.model_dump()
                except:
                    default = "Generated by factory"
            
            docs[field_name] = {
                "type": str(field_type),
                "default": default,
                "description": field_info.description or "No description available",
                "required": field_info.is_required(),
            }
        
        return docs
    
    @staticmethod
    def generate_markdown_docs(output_path: Path) -> None:
        """Generate comprehensive Markdown documentation.
        
        Args:
            output_path: Path where to save the documentation
        """
        # Configuration sections to document
        sections = [
            ("System Configuration", SystemConfig),
            ("Hardware Configuration", HardwareConfig),
            ("Image Processing Configuration", ImageProcessingConfig),
            ("Fiducial Configuration", Fiducial),
            ("Registration Configuration", RegisterConfig),
            ("Deconvolution Configuration", DeconvolutionConfig),
            ("Basic Correction Configuration", BasicConfig),
            ("Stitching Configuration", StitchingConfig),
            ("Spot Analysis Configuration", SpotAnalysisConfig),
            ("Channel Configuration", ChannelConfig),
        ]
        
        with open(output_path, 'w') as f:
            f.write("# Fishtools Configuration Reference\n\n")
            f.write("This document provides a comprehensive reference for all configuration options in the fishtools preprocessing pipeline.\n\n")
            
            f.write("## Table of Contents\n\n")
            for title, _ in sections:
                anchor = title.lower().replace(" ", "-")
                f.write(f"- [{title}](#{anchor})\n")
            f.write("\n")
            
            f.write("## Environment Variable Overrides\n\n")
            f.write("Any configuration parameter can be overridden using environment variables with the `FISHTOOLS_` prefix.\n")
            f.write("Use double underscores (`__`) to separate nested configuration keys.\n\n")
            f.write("Examples:\n")
            f.write("```bash\n")
            f.write("export FISHTOOLS_HARDWARE__MAX_MEMORY_MB=204800\n")
            f.write("export FISHTOOLS_FIDUCIAL__FWHM=5.0\n")
            f.write("export FISHTOOLS_REGISTRATION__REFERENCE=\"2_10_18\"\n")
            f.write("```\n\n")
            
            for title, model_class in sections:
                f.write(f"## {title}\n\n")
                f.write(f"{model_class.__doc__ or 'Configuration options for this section.'}\n\n")
                
                docs = ConfigDocGenerator.generate_field_docs(model_class)
                
                f.write("| Parameter | Type | Default | Description |\n")
                f.write("|-----------|------|---------|-------------|\n")
                
                for field_name, field_info in docs.items():
                    field_type = field_info["type"].replace("|", "\\|")
                    default_str = str(field_info["default"]).replace("|", "\\|")
                    desc = field_info["description"].replace("|", "\\|")
                    required = "**Required**" if field_info["required"] else ""
                    
                    f.write(f"| `{field_name}` | {field_type} | {default_str} | {desc} {required} |\n")
                
                f.write("\n")
            
            f.write("## Configuration File Examples\n\n")
            f.write("### Minimal Configuration\n\n")
            f.write("For basic usage, you only need to specify the data path:\n\n")
            f.write("```toml\n")
            f.write("dataPath = \"/path/to/your/data\"\n")
            f.write("\n")
            f.write("[registration]\n")
            f.write("reference = \"4_12_20\"\n")
            f.write("\n")
            f.write("[registration.chromatic_shifts]\n")
            f.write("647 = \"/path/to/data/560to647.txt\"\n")
            f.write("750 = \"/path/to/data/560to750.txt\"\n")
            f.write("\n")
            f.write("[registration.fiducial]\n")
            f.write("```\n\n")
            
            f.write("### Performance Tuning Configuration\n\n")
            f.write("For high-performance processing:\n\n")
            f.write("```toml\n")
            f.write("dataPath = \"/path/to/your/data\"\n")
            f.write("\n")
            f.write("[hardware]\n")
            f.write("max_memory_mb = 204800  # 200GB\n")
            f.write("parallel_threads = 64\n")
            f.write("\n")
            f.write("[hardware.thread_pools]\n")
            f.write("deconv = 12\n")
            f.write("stitch = 16\n")
            f.write("register = 32\n")
            f.write("\n")
            f.write("[deconvolution]\n")
            f.write("projector_step = 8  # Larger step for speed\n")
            f.write("```\n\n")
            
            f.write("### Debugging Configuration\n\n")
            f.write("For troubleshooting and debugging:\n\n")
            f.write("```toml\n")
            f.write("dataPath = \"/path/to/your/data\"\n")
            f.write("\n")
            f.write("[fiducial]\n")
            f.write("fwhm = 6.0  # Larger FWHM for more spots\n")
            f.write("threshold = 2.0  # Lower threshold\n")
            f.write("\n")
            f.write("[image_processing]\n")
            f.write("max_iterations = 10  # More iterations\n")
            f.write("residual_threshold = 0.5  # More lenient\n")
            f.write("\n")
            f.write("[basic_correction]\n")
            f.write("random_sample_limit = 2000  # More samples\n")
            f.write("```\n\n")
    
    @staticmethod
    def generate_json_schema(output_path: Path) -> None:
        """Generate JSON schema for the configuration.
        
        Args:
            output_path: Path where to save the JSON schema
        """
        schema = Config.model_json_schema()
        
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)


def validate_config_file(config_path: Path) -> bool:
    """Validate a configuration file and print results.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if valid, False otherwise
    """
    errors = ConfigValidator.validate_toml_file(config_path)
    
    if not errors:
        print(f"✅ Configuration file {config_path} is valid!")
        return True
    else:
        print(f"❌ Configuration file {config_path} has {len(errors)} error(s):")
        for error in errors:
            print(f"  • {error}")
        
        # Try to provide suggestions
        try:
            import toml
            config_dict = toml.load(config_path)
            suggestions = ConfigValidator.suggest_fixes(config_dict)
            
            if suggestions:
                print("\n💡 Suggestions:")
                for field, fixes in suggestions.items():
                    print(f"  {field}:")
                    for fix in fixes:
                        print(f"    - {fix}")
        except:
            pass
        
        return False


if __name__ == "__main__":
    # Generate documentation when run as script
    from pathlib import Path
    
    print("Generating configuration documentation...")
    ConfigDocGenerator.generate_markdown_docs(Path("CONFIG_REFERENCE.md"))
    ConfigDocGenerator.generate_json_schema(Path("config_schema.json"))
    print("Documentation generated!")