"""
YAML output generator.

Generates human-readable YAML files from ModelSpec objects.
"""

import os
from typing import Any

import yaml

from piqc.models.modelspec import ModelSpec
from piqc.utils.logger import get_logger


logger = get_logger(__name__)


class YAMLGenerator:
    """
    Generates YAML output files from ModelSpec objects.
    
    Produces human-readable YAML with consistent formatting:
    - 2-space indentation
    - Sorted keys for consistency
    - Proper multi-line string handling
    """
    
    def __init__(self, indent: int = 2) -> None:
        """
        Initialize the YAML generator.
        
        Args:
            indent: Number of spaces for indentation.
        """
        self.indent = indent
    
    def generate(self, modelspec: ModelSpec, output_path: str) -> str:
        """
        Generate a YAML file from a ModelSpec.
        
        Args:
            modelspec: ModelSpec object to serialize.
            output_path: Path for the output file.
            
        Returns:
            Path to the generated file.
        """
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert to dictionary
        data = modelspec.to_dict(by_alias=True)
        
        # Generate YAML content
        yaml_content = self._to_yaml(data)
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        
        logger.debug(f"Generated YAML: {output_path}")
        return output_path
    
    def generate_multi(
        self,
        modelspecs: list[ModelSpec],
        output_dir: str,
        filename_template: str = "{namespace}_{name}.yaml",
    ) -> list[str]:
        """
        Generate multiple YAML files in a directory.
        
        Args:
            modelspecs: List of ModelSpec objects.
            output_dir: Output directory path.
            filename_template: Template for filenames. Supports {name} and {namespace}.
            
        Returns:
            List of generated file paths.
        """
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files: list[str] = []
        
        for modelspec in modelspecs:
            # Generate filename
            filename = filename_template.format(
                name=self._sanitize_filename(modelspec.metadata.name),
                namespace=self._sanitize_filename(modelspec.metadata.namespace),
            )
            
            output_path = os.path.join(output_dir, filename)
            
            self.generate(modelspec, output_path)
            generated_files.append(output_path)
        
        logger.info(f"Generated {len(generated_files)} YAML files in {output_dir}")
        return generated_files
    
    def generate_combined(
        self,
        modelspecs: list[ModelSpec],
        output_path: str,
    ) -> str:
        """
        Generate a single YAML file containing all ModelSpecs.
        
        Uses YAML document separators (---) between entries.
        
        Args:
            modelspecs: List of ModelSpec objects.
            output_path: Path for the output file.
            
        Returns:
            Path to the generated file.
        """
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate combined content
        documents = []
        for modelspec in modelspecs:
            data = modelspec.to_dict(by_alias=True)
            documents.append(self._to_yaml(data))
        
        combined_content = "---\n".join(documents)
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(combined_content)
        
        logger.info(f"Generated combined YAML with {len(modelspecs)} documents: {output_path}")
        return output_path
    
    def to_string(self, modelspec: ModelSpec) -> str:
        """
        Convert a ModelSpec to YAML string.
        
        Args:
            modelspec: ModelSpec object to serialize.
            
        Returns:
            YAML formatted string.
        """
        data = modelspec.to_dict(by_alias=True)
        return self._to_yaml(data)
    
    def _to_yaml(self, data: dict[str, Any]) -> str:
        """Convert dictionary to YAML string with custom formatting."""
        # Use custom representer for clean output
        yaml.add_representer(
            type(None),
            lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:null", ""),
        )
        
        return yaml.dump(
            data,
            default_flow_style=False,
            allow_unicode=True,
            indent=self.indent,
            sort_keys=False,
            width=120,
        )
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as a filename."""
        # Replace problematic characters
        sanitized = name.replace("/", "-").replace("\\", "-")
        sanitized = sanitized.replace(":", "-").replace("*", "-")
        sanitized = sanitized.replace("?", "-").replace('"', "-")
        sanitized = sanitized.replace("<", "-").replace(">", "-")
        sanitized = sanitized.replace("|", "-")
        
        # Remove leading/trailing whitespace and dots
        sanitized = sanitized.strip().strip(".")
        
        return sanitized or "unnamed"
