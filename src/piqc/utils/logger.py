"""
Structured logging configuration for piqc.

Provides consistent logging across all modules with support for
different verbosity levels and structured output formatting.
"""

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


# Custom theme for consistent styling
THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
    "debug": "dim",
})

# Shared console instance for all output
console = Console(theme=THEME, stderr=True)


def setup_logging(
    verbose: bool = False,
    debug: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        verbose: Enable verbose output (INFO level with more detail).
        debug: Enable debug output (DEBUG level with full trace).
        log_file: Optional path to write logs to a file.
        
    Returns:
        Configured logger instance.
    """
    # Determine log level
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    # Configure root logger
    logger = logging.getLogger("piqc")
    logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with Rich formatting
    console_handler = RichHandler(
        console=console,
        show_time=debug,
        show_path=debug,
        rich_tracebacks=True,
        tracebacks_show_locals=debug,
        markup=True,
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("kubernetes").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name (typically __name__).
        
    Returns:
        Logger instance configured as a child of the main logger.
    """
    return logging.getLogger(f"piqc.{name}")


class LogContext:
    """
    Context manager for structured log output sections.
    
    Provides clear visual separation of log sections with
    consistent formatting.
    """
    
    def __init__(self, title: str, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize log context.
        
        Args:
            title: Section title to display.
            logger: Logger instance to use. Defaults to main logger.
        """
        self.title = title
        self.logger = logger or logging.getLogger("piqc")
    
    def __enter__(self) -> "LogContext":
        self.logger.info(f"[bold]{self.title}[/bold]")
        return self
    
    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        if exc_type is not None:
            self.logger.error(f"Failed: {self.title}")


class ProgressLogger:
    """
    Simple progress logging without animation.
    
    Provides clean status updates for long-running operations
    without terminal animation dependencies.
    """
    
    def __init__(self, description: str, total: Optional[int] = None) -> None:
        """
        Initialize progress logger.
        
        Args:
            description: Description of the operation.
            total: Total number of items (if known).
        """
        self.description = description
        self.total = total
        self.current = 0
        self.logger = logging.getLogger("piqc")
    
    def update(self, count: int = 1, status: Optional[str] = None) -> None:
        """
        Update progress.
        
        Args:
            count: Number of items completed in this update.
            status: Optional status message.
        """
        self.current += count
        if self.total:
            progress = f"[{self.current}/{self.total}]"
        else:
            progress = f"[{self.current}]"
        
        message = f"{self.description} {progress}"
        if status:
            message += f" - {status}"
        
        self.logger.debug(message)
    
    def complete(self, message: Optional[str] = None) -> None:
        """
        Mark operation as complete.
        
        Args:
            message: Optional completion message.
        """
        final_message = message or f"{self.description} complete"
        if self.total:
            final_message += f" ({self.current}/{self.total})"
        self.logger.info(final_message)
