"""
Logging utilities for the hybrid pipeline
"""

import sys
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

# Configure console for rich output
console = Console()

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add file logging
logger.add(
    "hybrid_pipeline.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"  # Changed from DEBUG to INFO to reduce log file size
)


def get_progress_bar() -> Progress:
    """Create a progress bar for long-running operations"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    )


def log_extraction_result(extractor_name: str, success: bool, details: str = ""):
    """Log extraction result with appropriate formatting"""
    if success:
        logger.success(f"{extractor_name}: Extraction completed. {details}")
    else:
        logger.error(f"{extractor_name}: Extraction failed. {details}")


def log_step(step: str, description: str = ""):
    """Log a pipeline step"""
    logger.info(f"[STEP] {step}: {description}")
    console.print(f"[bold blue]→[/bold blue] {step}", style="bold")
    if description:
        console.print(f"  {description}", style="dim")


def setup_logger(name: str = "hybrid_pipeline", level: str = "INFO"):
    """
    Setup logger with specified level
    
    Args:
        name: Logger name (not used with loguru)
        level: Logging level
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level.upper()
    )
    # Keep file logging at DEBUG level
    logger.add(
        "hybrid_pipeline.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG"
    )
    return logger


__all__ = ['logger', 'console', 'get_progress_bar', 'log_extraction_result', 'log_step', 'setup_logger']