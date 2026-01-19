"""
JSON output generator.

Generates JSON files from ModelSpec objects.
"""

import json
import os
from typing import Any

from piqc.models.modelspec import ModelSpec
from piqc.utils.logger import get_logger


logger = get_logger(__name__)


class JSONGenerator:
    """
    Generates JSON output files from ModelSpec objects.
    
    Produces formatted JSON with consistent indentation.
    """
    
    def __init__(self, indent: int = 2) -> None:
        """
        Initialize the JSON generator.
        
        Args:
            indent: Number of spaces for indentation.
        """
        self.indent = indent
    
    def generate(self, modelspec: ModelSpec, output_path: str) -> str:
        """
        Generate a JSON file from a ModelSpec.
        
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
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=self.indent, ensure_ascii=False)
        
        logger.debug(f"Generated JSON: {output_path}")
        return output_path
    
    def generate_multi(
        self,
        modelspecs: list[ModelSpec],
        output_dir: str,
        filename_template: str = "{namespace}_{name}.json",
    ) -> list[str]:
        """
        Generate multiple JSON files in a directory.
        
        Args:
            modelspecs: List of ModelSpec objects.
            output_dir: Output directory path.
            filename_template: Template for filenames.
            
        Returns:
            List of generated file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files: list[str] = []
        
        for modelspec in modelspecs:
            filename = filename_template.format(
                name=self._sanitize_filename(modelspec.metadata.name),
                namespace=self._sanitize_filename(modelspec.metadata.namespace),
            )
            
            output_path = os.path.join(output_dir, filename)
            self.generate(modelspec, output_path)
            generated_files.append(output_path)
        
        logger.info(f"Generated {len(generated_files)} JSON files in {output_dir}")
        return generated_files
    
    def generate_combined(
        self,
        modelspecs: list[ModelSpec],
        output_path: str,
    ) -> str:
        """
        Generate a single JSON file containing all ModelSpecs as an array.
        
        Args:
            modelspecs: List of ModelSpec objects.
            output_path: Path for the output file.
            
        Returns:
            Path to the generated file.
        """
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        data = [spec.to_dict(by_alias=True) for spec in modelspecs]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=self.indent, ensure_ascii=False)
        
        logger.info(f"Generated combined JSON with {len(modelspecs)} entries: {output_path}")
        return output_path
    
    def to_string(self, modelspec: ModelSpec) -> str:
        """
        Convert a ModelSpec to JSON string.
        
        Args:
            modelspec: ModelSpec object to serialize.
            
        Returns:
            JSON formatted string.
        """
        data = modelspec.to_dict(by_alias=True)
        return json.dumps(data, indent=self.indent, ensure_ascii=False)
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as a filename."""
        sanitized = name.replace("/", "-").replace("\\", "-")
        sanitized = sanitized.replace(":", "-").replace("*", "-")
        sanitized = sanitized.strip().strip(".")
        return sanitized or "unnamed"
