#!/usr/bin/env python3
"""
PDF Document Extraction System - Unified Entry Point

Runs the web application (FastAPI + Next.js), CLI extraction,
batch processing, and configuration management from a single file.

Usage:
    # Web Application (default)
    python main.py                              Run both backend and frontend
    python main.py --backend                    Run backend only
    python main.py --frontend                   Run frontend only
    python main.py --check                      Check dependencies

    # CLI Extraction
    python main.py extract <pdf_file>                    Extract with defaults
    python main.py extract <pdf_file> --output results/  Custom output dir
    python main.py extract <pdf_file> --pages 1-5        Specific pages
    python main.py extract <pdf_file> --no-excel         Skip Excel export

    # Batch Processing
    python main.py batch <directory>             Process all PDFs in directory
    python main.py batch <directory> --parallel 4    Use 4 parallel workers

    # Configuration
    python main.py config --show                 Show current configuration
    python main.py config --set-dpi 200          Set DPI for extraction
    python main.py config --set-model <model>    Set VLM model
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
BACKEND_DIR = PROJECT_ROOT / "src"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
CONFIG_FILE = PROJECT_ROOT / "config.json"


# ==================== Terminal Colors ====================

class Color:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"


# ==================== Logging Helpers ====================

def log(message: str, color: str = "", prefix: str = "MAIN") -> None:
    """Print a formatted log message."""
    timestamp = time.strftime("%H:%M:%S")
    reset = Color.RESET if color else ""
    print(f"{color}[{timestamp}] [{prefix}] {message}{reset}")


def log_backend(message: str) -> None:
    log(message, Color.CYAN, "BACKEND")


def log_frontend(message: str) -> None:
    log(message, Color.MAGENTA, "FRONTEND")


def log_success(message: str) -> None:
    log(message, Color.GREEN)


def log_error(message: str) -> None:
    log(message, Color.RED)


def log_warning(message: str) -> None:
    log(message, Color.YELLOW)


# ==================== Configuration ====================

class ExtractionMode(Enum):
    """Extraction mode enumeration."""
    SINGLE_RECORD = "single"
    MULTI_RECORD = "multi"
    UNIVERSAL = "universal"


@dataclass
class ExtractionConfig:
    """Configuration for extraction."""
    dpi: int = 300
    max_retries: int = 3
    retry_delay: int = 5
    vlm_model: str = "qwen/qwen3-vl-8b"
    vlm_endpoint: str = "http://localhost:1234/v1"
    enable_multi_record: bool = True
    enable_duplicate_detection: bool = True
    export_excel: bool = True
    export_markdown: bool = True
    export_json: bool = True
    batch_size: int = 10
    parallel_workers: int = 1
    log_level: str = "INFO"

    # ─── Phase 2: Multi-Stage Validation (default False for safety) ───
    enable_validation_stage: bool = False
    enable_self_correction: bool = False

    # ─── Phase 3: Consensus & Ensemble Mechanisms ───
    enable_consensus_for_critical_fields: bool = False
    critical_field_keywords: list[str] | None = None

    # ─── Thresholds & Parameters ───
    validation_confidence_threshold: float = 0.85

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.critical_field_keywords is None:
            self.critical_field_keywords = [
                "id", "number", "code", "mrn", "ssn",
                "date", "dob", "amount", "charge", "total", "balance"
            ]


def load_config() -> ExtractionConfig:
    """Load configuration from file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            return ExtractionConfig(**data)
    return ExtractionConfig()


def save_config(config: ExtractionConfig) -> None:
    """Save configuration to file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config.__dict__, f, indent=2)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup enterprise-grade logging."""
    logger = logging.getLogger("pdf_extraction")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        f"{Color.CYAN}%(asctime)s{Color.RESET} - "
        f"{Color.BOLD}%(name)s{Color.RESET} - "
        f"%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # File handler
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"extraction_{time.strftime('%Y%m%d')}.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ==================== Dependency Checks ====================

def check_python_version() -> bool:
    """Check Python version is 3.11+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        log_error(f"Python 3.11+ required. Current: {version.major}.{version.minor}")
        return False
    log_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_node_version() -> bool:
    """Check Node.js is installed."""
    try:
        result = subprocess.run(
            ["node", "--version"], check=False, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            log_success(f"Node.js version: {version}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    log_error("Node.js not found. Please install Node.js 18+")
    return False


def check_npm() -> bool:
    """Check npm is installed."""
    try:
        result = subprocess.run(
            ["npm", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            shell=True,  # Required for Windows
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            log_success(f"npm version: {version}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    log_error("npm not found. Please install Node.js/npm")
    return False


def check_backend_dependencies() -> bool:
    """Check backend Python dependencies."""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "openai",
        "openpyxl",
        "Pillow",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        log_warning(f"Missing Python packages: {', '.join(missing)}")
        log_warning("Run: pip install -r requirements.txt")
        return False

    log_success("Backend dependencies: OK")
    return True


def check_frontend_dependencies() -> bool:
    """Check frontend Node.js dependencies."""
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        log_warning("Frontend dependencies not installed")
        log_warning(f"Run: cd {FRONTEND_DIR} && npm install")
        return False

    log_success("Frontend dependencies: OK")
    return True


def check_env_file() -> bool:
    """Check .env file exists and validate security configuration."""
    env_file = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"

    if not env_file.exists():
        if env_example.exists():
            log_warning(".env file not found. Creating from .env.example")
            import shutil

            shutil.copy(env_example, env_file)
        else:
            log_warning(".env file not found. Using defaults")
        return True

    log_success("Environment file: OK")

    # Verify critical security environment variables
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        log_warning("python-dotenv not installed, skipping .env validation")
        return True

    critical_vars = ["SECRET_KEY", "ENCRYPTION_KEY"]
    missing_vars = []

    for var in critical_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        elif len(value) < 32:
            log_warning(f"{var} is too short (< 32 characters). Use strong keys in production!")

    if missing_vars:
        log_warning(f"Missing critical environment variables: {', '.join(missing_vars)}")
        log_warning(
            "Generate secure keys with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
        )
        log_warning("Using defaults for development. Set strong keys for production!")

    log_success("Security configuration: OK")
    return True


def run_checks() -> bool:
    """Run all dependency and configuration checks."""
    log(f"{Color.BOLD}Running pre-flight checks...{Color.RESET}")

    checks = [
        ("Python Version", check_python_version),
        ("Node.js", check_node_version),
        ("npm", check_npm),
        ("Backend Dependencies", check_backend_dependencies),
        ("Frontend Dependencies", check_frontend_dependencies),
        ("Environment File", check_env_file),
    ]

    all_passed = True
    for name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            log_error(f"{name} check failed: {e}")
            all_passed = False

    if all_passed:
        log_success("All checks passed!")
    else:
        log_error("Some checks failed. Please fix the issues above.")

    return all_passed


# ==================== Web Server (ProcessManager) ====================

class ServerStatus(Enum):
    """Server status enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ServerConfig:
    """Server configuration."""
    name: str
    color: str
    command: list[str]
    cwd: Path
    port: int
    health_url: str
    env: dict | None = None


class ProcessManager:
    """Manages backend and frontend server processes."""

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self.configs: dict[str, ServerConfig] = {}
        self.running = False
        self._shutdown_event = asyncio.Event()

    def _get_backend_config(self) -> ServerConfig:
        """Get backend server configuration."""
        lm_model = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-vl-8b")

        return ServerConfig(
            name="backend",
            color=Color.CYAN,
            command=[
                sys.executable,
                "-m",
                "uvicorn",
                "src.api.app:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--reload",
                "--reload-dir",
                "src",
            ],
            cwd=PROJECT_ROOT,
            port=8000,
            health_url="http://localhost:8000/api/v1/health",
            env={
                "PYTHONPATH": str(PROJECT_ROOT),
                "PYTHONUNBUFFERED": "1",
                "LM_STUDIO_MODEL": lm_model,
            },
        )

    def _get_frontend_config(self) -> ServerConfig:
        """Get frontend server configuration."""
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

        return ServerConfig(
            name="frontend",
            color=Color.MAGENTA,
            command=[npm_cmd, "run", "dev", "--", "--port", "3000"],
            cwd=FRONTEND_DIR,
            port=3000,
            health_url="http://localhost:3000",
            env={
                "NEXT_PUBLIC_API_URL": "http://localhost:8000",
                "PORT": "3000",
            },
        )

    def _stream_output(self, process: subprocess.Popen, config: ServerConfig) -> None:
        """Stream process output to console."""

        def read_stream(stream, is_error: bool = False):
            try:
                for line in iter(stream.readline, ""):
                    if not line:
                        break
                    line = line.rstrip()
                    if line:
                        prefix = config.name.upper()
                        color = Color.RED if is_error else config.color
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"{color}[{timestamp}] [{prefix}] {line}{Color.RESET}")
            except (ValueError, OSError):
                pass  # Stream closed

        import threading

        if process.stdout:
            stdout_thread = threading.Thread(
                target=read_stream, args=(process.stdout, False), daemon=True
            )
            stdout_thread.start()

        if process.stderr:
            stderr_thread = threading.Thread(
                target=read_stream, args=(process.stderr, True), daemon=True
            )
            stderr_thread.start()

    def start_server(self, config: ServerConfig) -> bool:
        """Start a server process."""
        try:
            log(f"Starting {config.name} server on port {config.port}...", config.color)

            env = os.environ.copy()
            if config.env:
                env.update(config.env)

            process = subprocess.Popen(
                config.command,
                cwd=config.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )

            self.processes[config.name] = process
            self.configs[config.name] = config

            self._stream_output(process, config)

            log(f"{config.name.capitalize()} server started (PID: {process.pid})", config.color)
            return True

        except Exception as e:
            log_error(f"Failed to start {config.name}: {e}")
            return False

    def stop_server(self, name: str) -> None:
        """Stop a server process."""
        process = self.processes.get(name)
        if not process:
            return

        log(f"Stopping {name} server...", Color.YELLOW)

        try:
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()

            try:
                process.wait(timeout=5)
                log(f"{name.capitalize()} server stopped", Color.YELLOW)
            except subprocess.TimeoutExpired:
                log_warning(f"{name} server did not stop gracefully, killing...")
                process.kill()
                process.wait(timeout=2)
        except Exception as e:
            log_error(f"Error stopping {name}: {e}")
            try:
                process.kill()
            except Exception:
                pass

        del self.processes[name]

    def stop_all(self) -> None:
        """Stop all server processes."""
        log("Shutting down all servers...", Color.YELLOW)

        for name in list(self.processes.keys()):
            self.stop_server(name)

        log_success("All servers stopped")

    def is_running(self, name: str) -> bool:
        """Check if a server is running."""
        process = self.processes.get(name)
        if not process:
            return False
        return process.poll() is None

    async def wait_for_health(self, config: ServerConfig, timeout: int = 60) -> bool:
        """Wait for server health check to pass."""
        import urllib.error
        import urllib.request

        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.is_running(config.name):
                return False

            urls_to_try = [config.health_url]
            if config.name == "frontend":
                urls_to_try.extend(
                    [
                        "http://localhost:3001",
                        "http://localhost:3002",
                    ]
                )

            for url in urls_to_try:
                try:
                    with urllib.request.urlopen(url, timeout=2) as response:  # nosec B310
                        if response.status == 200:
                            log_success(f"{config.name.capitalize()} server is healthy on {url}")
                            config.health_url = url
                            config.port = int(url.split(":")[-1])
                            return True
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
                    continue

            await asyncio.sleep(1)

        log_warning(f"{config.name.capitalize()} health check timed out")
        return False

    async def run(
        self, run_backend: bool = True, run_frontend: bool = True, wait_for_health: bool = True
    ) -> None:
        """Run the servers."""
        self.running = True

        def signal_handler(signum, frame):
            log("\nReceived shutdown signal...", Color.YELLOW)
            self.running = False
            self._shutdown_event.set()

        if sys.platform != "win32":
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        else:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGBREAK, signal_handler)

        try:
            # Start backend
            if run_backend:
                backend_config = self._get_backend_config()
                if not self.start_server(backend_config):
                    return

                if wait_for_health:
                    await asyncio.sleep(2)
                    await self.wait_for_health(backend_config)

            # Start frontend
            if run_frontend:
                frontend_config = self._get_frontend_config()
                if not self.start_server(frontend_config):
                    self.stop_all()
                    return

                if wait_for_health:
                    await asyncio.sleep(5)
                    await self.wait_for_health(frontend_config, timeout=120)

            # Print access information
            log(f"{Color.BOLD}{'='*50}{Color.RESET}")
            log(f"{Color.BOLD}PDF Document Extraction System is running!{Color.RESET}")
            log(f"{Color.BOLD}{'='*50}{Color.RESET}")
            if run_backend:
                log(f"  Backend API:    {Color.CYAN}http://localhost:8000{Color.RESET}")
                log(f"  API Docs:       {Color.CYAN}http://localhost:8000/docs{Color.RESET}")
            if run_frontend:
                frontend_config = self.configs.get("frontend")
                frontend_port = frontend_config.port if frontend_config else 3000
                log(
                    f"  Frontend:       {Color.MAGENTA}http://localhost:{frontend_port}{Color.RESET}"
                )
            log(f"Press {Color.BOLD}Ctrl+C{Color.RESET} to stop all servers")

            # Monitor processes
            while self.running:
                if run_backend and not self.is_running("backend"):
                    log_error("Backend server crashed!")
                    break

                if run_frontend and not self.is_running("frontend"):
                    log_error("Frontend server crashed!")
                    break

                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=1.0)
                    break
                except TimeoutError:
                    continue

        except KeyboardInterrupt:
            log("\nKeyboard interrupt received...", Color.YELLOW)

        finally:
            self.running = False
            self.stop_all()


# ==================== CLI Extraction ====================

def extract_pdf_cli(
    pdf_path: str,
    output_dir: Optional[str] = None,
    pages: Optional[str] = None,
    config: Optional[ExtractionConfig] = None,
) -> Dict[str, Any]:
    """
    Extract data from PDF using CLI mode.

    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory for results
        pages: Page range (e.g., "1-5", "1,3,5", "all")
        config: Extraction configuration

    Returns:
        Extraction results dictionary
    """
    if config is None:
        config = load_config()

    logger = setup_logging(config.log_level)
    logger.info(f"Starting extraction: {pdf_path}")

    # Import extraction modules
    from src.client.lm_client import LMStudioClient
    from src.export.consolidated_export import export_excel, export_json, export_markdown
    from src.pipeline.runner import PipelineRunner

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        log_error(f"PDF file not found: {pdf_path}")
        return {"status": "error", "message": "PDF file not found"}

    # Determine output directory
    if output_dir is None:
        out = PROJECT_ROOT / "output" / pdf_file.stem
    else:
        out = Path(output_dir)

    out.mkdir(parents=True, exist_ok=True)

    # Parse page range
    start_page = 1
    end_page = None

    if pages:
        if pages.lower() == "all":
            pass
        elif "-" in pages:
            parts = pages.split("-")
            start_page = int(parts[0])
            end_page = int(parts[1]) if parts[1] else None
        elif "," in pages:
            log_warning("Non-contiguous pages not yet supported, processing all")

    try:
        log(f"Processing: {pdf_file.name}", Color.CYAN)
        log(f"Output: {out}", Color.CYAN)
        log(f"Mode: {'Multi-record' if config.enable_multi_record else 'Single-record'}", Color.CYAN)

        # Create pipeline runner
        client = LMStudioClient(
            base_url=config.vlm_endpoint,
            model=config.vlm_model,
        )
        runner = PipelineRunner(
            client=client,
            enable_checkpointing=False,
            dpi=config.dpi,
        )

        if config.enable_multi_record:
            # Multi-record extraction
            result = runner.extract_multi_record(
                pdf_path=str(pdf_file),
                start_page=start_page if start_page > 1 else None,
                end_page=end_page,
                enable_validation=config.enable_validation_stage,
                enable_self_correction=config.enable_self_correction,
                confidence_threshold=config.validation_confidence_threshold,
                enable_consensus=config.enable_consensus_for_critical_fields,
                critical_field_keywords=config.critical_field_keywords,
            )

            # Export
            if config.export_json:
                json_file = out / f"{pdf_file.stem}_results.json"
                export_json(result, json_file)
                logger.info(f"JSON export: {json_file}")

            if config.export_excel:
                excel_file = out / f"{pdf_file.stem}_consolidated.xlsx"
                export_excel(result, excel_file)
                logger.info(f"Excel export: {excel_file}")

            if config.export_markdown:
                md_file = out / f"{pdf_file.stem}_report.md"
                export_markdown(result, md_file)
                logger.info(f"Markdown export: {md_file}")

            # Print summary
            log_success("Extraction Complete!")
            log(f"Pages: {result.total_pages}", Color.GREEN)
            log(f"Records: {result.total_records}", Color.GREEN)
            log(f"Document Type: {result.document_type}", Color.GREEN)
            log(f"Entity Type: {result.entity_type}", Color.GREEN)
            log(f"Time: {result.total_processing_time_ms / 1000:.2f}s", Color.GREEN)
            log(f"VLM Calls: {result.total_vlm_calls}", Color.GREEN)

            return {
                "status": "success",
                "total_records": result.total_records,
                "output_dir": str(out),
            }
        else:
            # Single-record extraction (legacy pipeline)
            result = runner.extract_from_pdf(str(pdf_file))

            results_file = out / f"{pdf_file.stem}_results.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(dict(result), f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Results saved: {results_file}")

            log_success("Extraction Complete!")
            log(f"Fields: {len(result.get('merged_extraction', {}))}", Color.GREEN)
            log(f"Confidence: {result.get('overall_confidence', 0):.0%}", Color.GREEN)

            return {
                "status": "success",
                "output_dir": str(out),
            }

    except Exception as e:
        log_error(f"Extraction failed: {str(e)}")
        logger.exception("Extraction error")
        return {"status": "error", "message": str(e)}


# ==================== Batch Processing ====================

def batch_process_cli(
    directory: str,
    output_dir: Optional[str] = None,
    config: Optional[ExtractionConfig] = None,
) -> Dict[str, Any]:
    """
    Batch process all PDFs in a directory.

    Args:
        directory: Directory containing PDF files
        output_dir: Output directory for all results
        config: Extraction configuration

    Returns:
        Batch processing results
    """
    if config is None:
        config = load_config()

    logger = setup_logging(config.log_level)

    input_dir = Path(directory)
    if not input_dir.exists():
        log_error(f"Directory not found: {directory}")
        return {"status": "error", "message": "Directory not found"}

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        log_warning("No PDF files found in directory")
        return {"status": "success", "processed": 0, "failed": 0}

    log(f"Found {len(pdf_files)} PDF files", Color.CYAN)

    if output_dir is None:
        out = PROJECT_ROOT / "output" / "batch" / input_dir.name
    else:
        out = Path(output_dir)

    out.mkdir(parents=True, exist_ok=True)

    results = []
    failed = []

    if config.parallel_workers > 1:
        log(f"Processing with {config.parallel_workers} parallel workers", Color.CYAN)

        with ProcessPoolExecutor(max_workers=config.parallel_workers) as executor:
            futures = {
                executor.submit(
                    extract_pdf_cli,
                    str(pdf_file),
                    str(out / pdf_file.stem),
                    "all",
                    config,
                ): pdf_file
                for pdf_file in pdf_files
            }

            for future in as_completed(futures):
                pdf_file = futures[future]
                try:
                    result = future.result()
                    if result["status"] == "success":
                        results.append(pdf_file.name)
                        log_success(f"Completed: {pdf_file.name}")
                    else:
                        failed.append(pdf_file.name)
                        log_error(f"Failed: {pdf_file.name}")
                except Exception as e:
                    failed.append(pdf_file.name)
                    log_error(f"Error processing {pdf_file.name}: {e}")
    else:
        for pdf_file in pdf_files:
            log(f"Processing: {pdf_file.name}", Color.CYAN)
            result = extract_pdf_cli(
                str(pdf_file),
                str(out / pdf_file.stem),
                "all",
                config,
            )

            if result["status"] == "success":
                results.append(pdf_file.name)
            else:
                failed.append(pdf_file.name)

    log_success("Batch processing complete!")
    log(f"Processed: {len(results)}/{len(pdf_files)}", Color.GREEN)
    if failed:
        log(f"Failed: {len(failed)}", Color.RED)
        for name in failed:
            log(f"  - {name}", Color.RED)

    return {
        "status": "success",
        "total": len(pdf_files),
        "processed": len(results),
        "failed": len(failed),
        "failed_files": failed,
    }


# ==================== Config CLI ====================

def config_cli(args: argparse.Namespace) -> int:
    """Handle config command."""
    config = load_config()

    if args.show:
        print(f"\n{Color.BOLD}Current Configuration:{Color.RESET}")
        for key, value in config.__dict__.items():
            print(f"  {key}: {Color.CYAN}{value}{Color.RESET}")
        print()
        return 0

    changed = False

    if args.set_dpi:
        config.dpi = args.set_dpi
        changed = True
        log_success(f"DPI set to {args.set_dpi}")

    if args.set_model:
        config.vlm_model = args.set_model
        changed = True
        log_success(f"VLM model set to {args.set_model}")

    if args.set_endpoint:
        config.vlm_endpoint = args.set_endpoint
        changed = True
        log_success(f"VLM endpoint set to {args.set_endpoint}")

    if args.enable_multi_record is not None:
        config.enable_multi_record = args.enable_multi_record
        changed = True
        log_success(f"Multi-record mode: {args.enable_multi_record}")

    if args.parallel_workers:
        config.parallel_workers = args.parallel_workers
        changed = True
        log_success(f"Parallel workers set to {args.parallel_workers}")

    if changed:
        save_config(config)
        log_success("Configuration saved")

    return 0


# ==================== Web Server Entry ====================

async def run_web_server(args: argparse.Namespace) -> int:
    """Run the web application (FastAPI backend + Next.js frontend)."""
    # Check only mode
    if args.check:
        return 0 if run_checks() else 1

    # Run pre-flight checks
    if not args.skip_checks:
        if not run_checks():
            log_error("Pre-flight checks failed. Use --skip-checks to bypass.")
            return 1

    # Determine what to run
    if args.both:
        run_backend = True
        run_frontend = True
    elif args.backend:
        run_backend = True
        run_frontend = False
    elif args.frontend:
        run_backend = False
        run_frontend = True
    else:
        # Default: run both
        run_backend = True
        run_frontend = True

    manager = ProcessManager()
    await manager.run(
        run_backend=run_backend,
        run_frontend=run_frontend,
        wait_for_health=not args.no_health_check,
    )

    return 0


# ==================== Main Entry Point ====================

def main() -> int:
    """Unified main entry point with CLI and web server support."""
    parser = argparse.ArgumentParser(
        description="PDF Document Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Web Application:
    python main.py                              Run both backend and frontend
    python main.py --backend                    Backend only
    python main.py --frontend                   Frontend only
    python main.py --check                      Run dependency checks only

  CLI Extraction:
    python main.py extract invoice.pdf                  Extract with defaults
    python main.py extract invoice.pdf --pages 1-5      Extract pages 1-5
    python main.py extract invoice.pdf --output out/    Custom output directory

  Batch Processing:
    python main.py batch documents/                     Process all PDFs
    python main.py batch documents/ --parallel 4        Use 4 parallel workers

  Configuration:
    python main.py config --show                        Show current config
    python main.py config --set-dpi 200                 Set DPI to 200
    python main.py config --set-model qwen3-vl-8b       Set VLM model
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract data from a PDF")
    extract_parser.add_argument("pdf_file", help="Path to PDF file")
    extract_parser.add_argument("-o", "--output", help="Output directory")
    extract_parser.add_argument("-p", "--pages", help="Page range (e.g., '1-5', 'all')")
    extract_parser.add_argument("--dpi", type=int, help="DPI for image conversion")
    extract_parser.add_argument("--no-excel", action="store_true", help="Skip Excel export")
    extract_parser.add_argument("--no-markdown", action="store_true", help="Skip Markdown export")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process PDFs in directory")
    batch_parser.add_argument("directory", help="Directory containing PDF files")
    batch_parser.add_argument("-o", "--output", help="Output directory")
    batch_parser.add_argument("--parallel", type=int, help="Number of parallel workers")

    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("--show", action="store_true", help="Show current config")
    config_parser.add_argument("--set-dpi", type=int, help="Set DPI")
    config_parser.add_argument("--set-model", help="Set VLM model name")
    config_parser.add_argument("--set-endpoint", help="Set VLM endpoint URL")
    config_parser.add_argument(
        "--enable-multi-record", type=bool, help="Enable multi-record detection"
    )
    config_parser.add_argument("--parallel-workers", type=int, help="Set parallel workers")

    # Web server flags (top-level, for default web server mode)
    parser.add_argument("--backend", action="store_true", help="Run backend server only")
    parser.add_argument("--frontend", action="store_true", help="Run frontend server only")
    parser.add_argument(
        "--both", action="store_true", help="Run both backend and frontend servers (default)"
    )
    parser.add_argument("--check", action="store_true", help="Run dependency checks only")
    parser.add_argument(
        "--skip-checks", action="store_true", help="Skip pre-flight dependency checks"
    )
    parser.add_argument("--no-health-check", action="store_true", help="Skip health check wait")

    args = parser.parse_args()

    # Print banner
    print(f"\n{Color.BOLD}{Color.BLUE}{'='*60}{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}  PDF Document Extraction System{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}  Universal Multi-Record Extraction with VLM{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}{'='*60}{Color.RESET}\n")

    try:
        if args.command == "extract":
            config = load_config()
            if args.dpi:
                config.dpi = args.dpi
            if args.no_excel:
                config.export_excel = False
            if args.no_markdown:
                config.export_markdown = False

            result = extract_pdf_cli(args.pdf_file, args.output, args.pages, config)
            return 0 if result["status"] == "success" else 1

        elif args.command == "batch":
            config = load_config()
            if args.parallel:
                config.parallel_workers = args.parallel

            result = batch_process_cli(args.directory, args.output, config)
            return 0 if result["status"] == "success" else 1

        elif args.command == "config":
            return config_cli(args)

        else:
            # Default: run web server
            return asyncio.run(run_web_server(args))

    except KeyboardInterrupt:
        log("\nInterrupted by user", Color.YELLOW)
        return 130
    except Exception as e:
        log_error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
