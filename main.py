#!/usr/bin/env python3
"""
PDF Document Extraction System - Main Entry Point

Runs both the FastAPI backend and Next.js frontend servers together.
Provides unified logging, graceful shutdown, and health monitoring.

Usage:
    python main.py              # Run both backend and frontend
    python main.py --backend    # Run backend only
    python main.py --frontend   # Run frontend only
    python main.py --check      # Check dependencies and configuration
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
BACKEND_DIR = PROJECT_ROOT / "src"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


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
    env: Optional[dict] = None


def log(message: str, color: str = "", prefix: str = "MAIN") -> None:
    """Print a formatted log message."""
    timestamp = time.strftime("%H:%M:%S")
    reset = Color.RESET if color else ""
    print(f"{color}[{timestamp}] [{prefix}] {message}{reset}")


def log_backend(message: str) -> None:
    """Log backend server message."""
    log(message, Color.CYAN, "BACKEND")


def log_frontend(message: str) -> None:
    """Log frontend server message."""
    log(message, Color.MAGENTA, "FRONTEND")


def log_success(message: str) -> None:
    """Log success message."""
    log(message, Color.GREEN)


def log_error(message: str) -> None:
    """Log error message."""
    log(message, Color.RED)


def log_warning(message: str) -> None:
    """Log warning message."""
    log(message, Color.YELLOW)


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
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=10
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
            capture_output=True,
            text=True,
            timeout=10,
            shell=True  # Required for Windows
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
        "celery",
        "redis",
        "langgraph",
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
    from dotenv import load_dotenv
    load_dotenv(env_file)

    critical_vars = ['JWT_SECRET_KEY', 'SECRET_KEY', 'ENCRYPTION_KEY']
    missing_vars = []

    for var in critical_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        elif len(value) < 32:
            log_warning(f"{var} is too short (< 32 characters). Use strong keys in production!")

    if missing_vars:
        log_error(f"Missing critical environment variables: {', '.join(missing_vars)}")
        log_error("Generate secure keys with: python -c 'import secrets; print(secrets.token_urlsafe(64))'")
        return False

    log_success("Security configuration: OK")
    return True


def run_checks() -> bool:
    """Run all dependency and configuration checks."""
    log(f"{Color.BOLD}Running pre-flight checks...{Color.RESET}")
    print()

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

    print()
    if all_passed:
        log_success("All checks passed!")
    else:
        log_error("Some checks failed. Please fix the issues above.")

    return all_passed


class ProcessManager:
    """Manages backend and frontend server processes."""

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self.running = False
        self._shutdown_event = asyncio.Event()

    def _get_backend_config(self) -> ServerConfig:
        """Get backend server configuration."""
        return ServerConfig(
            name="backend",
            color=Color.CYAN,
            command=[
                sys.executable, "-m", "uvicorn",
                "src.api.app:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload",
                "--reload-dir", "src",
            ],
            cwd=PROJECT_ROOT,
            port=8000,
            health_url="http://localhost:8000/api/v1/health",
            env={
                "PYTHONPATH": str(PROJECT_ROOT),
                "PYTHONUNBUFFERED": "1",
            }
        )

    def _get_frontend_config(self) -> ServerConfig:
        """Get frontend server configuration."""
        # Use npm.cmd on Windows, npm on Unix
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

        return ServerConfig(
            name="frontend",
            color=Color.MAGENTA,
            command=[npm_cmd, "run", "dev"],
            cwd=FRONTEND_DIR,
            port=3000,
            health_url="http://localhost:3000",
            env={
                "NEXT_PUBLIC_API_URL": "http://localhost:8000",
            }
        )

    def _stream_output(self, process: subprocess.Popen, config: ServerConfig) -> None:
        """Stream process output to console."""
        def read_stream(stream, is_error: bool = False):
            try:
                for line in iter(stream.readline, ''):
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
                target=read_stream,
                args=(process.stdout, False),
                daemon=True
            )
            stdout_thread.start()

        if process.stderr:
            stderr_thread = threading.Thread(
                target=read_stream,
                args=(process.stderr, True),
                daemon=True
            )
            stderr_thread.start()

    def start_server(self, config: ServerConfig) -> bool:
        """Start a server process."""
        try:
            log(f"Starting {config.name} server on port {config.port}...", config.color)

            # Merge environment variables
            env = os.environ.copy()
            if config.env:
                env.update(config.env)

            # Start process
            process = subprocess.Popen(
                config.command,
                cwd=config.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                # On Windows, don't create a new console window
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )

            self.processes[config.name] = process

            # Stream output in background threads
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
                # Windows: send CTRL+BREAK signal
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Unix: send SIGTERM
                process.terminate()

            # Wait for graceful shutdown
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

    async def wait_for_health(self, config: ServerConfig, timeout: int = 30) -> bool:
        """Wait for server health check to pass."""
        import urllib.request
        import urllib.error

        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.is_running(config.name):
                return False

            try:
                with urllib.request.urlopen(config.health_url, timeout=2) as response:
                    if response.status == 200:
                        log_success(f"{config.name.capitalize()} server is healthy")
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                pass

            await asyncio.sleep(1)

        log_warning(f"{config.name.capitalize()} health check timed out")
        return False

    async def run(
        self,
        run_backend: bool = True,
        run_frontend: bool = True,
        wait_for_health: bool = True
    ) -> None:
        """Run the servers."""
        self.running = True

        # Setup signal handlers
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
                    await asyncio.sleep(2)  # Give server time to start
                    await self.wait_for_health(backend_config)

            # Start frontend
            if run_frontend:
                frontend_config = self._get_frontend_config()
                if not self.start_server(frontend_config):
                    self.stop_all()
                    return

                if wait_for_health:
                    await asyncio.sleep(3)  # Next.js takes longer to compile
                    # Don't wait for frontend health - it compiles on first request

            # Print access information
            print()
            log(f"{Color.BOLD}{'='*50}{Color.RESET}")
            log(f"{Color.BOLD}PDF Document Extraction System is running!{Color.RESET}")
            log(f"{Color.BOLD}{'='*50}{Color.RESET}")
            print()
            if run_backend:
                log(f"  Backend API:    {Color.CYAN}http://localhost:8000{Color.RESET}")
                log(f"  API Docs:       {Color.CYAN}http://localhost:8000/docs{Color.RESET}")
            if run_frontend:
                log(f"  Frontend:       {Color.MAGENTA}http://localhost:3000{Color.RESET}")
            print()
            log(f"Press {Color.BOLD}Ctrl+C{Color.RESET} to stop all servers")
            print()

            # Monitor processes
            while self.running:
                # Check if processes are still running
                if run_backend and not self.is_running("backend"):
                    log_error("Backend server crashed!")
                    break

                if run_frontend and not self.is_running("frontend"):
                    log_error("Frontend server crashed!")
                    break

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=1.0
                    )
                    break
                except asyncio.TimeoutError:
                    continue

        except KeyboardInterrupt:
            log("\nKeyboard interrupt received...", Color.YELLOW)

        finally:
            self.running = False
            self.stop_all()


async def main_async(args: argparse.Namespace) -> int:
    """Async main entry point."""
    # Check only mode
    if args.check:
        return 0 if run_checks() else 1

    # Run pre-flight checks
    if not args.skip_checks:
        if not run_checks():
            log_error("Pre-flight checks failed. Use --skip-checks to bypass.")
            return 1

    print()

    # Determine what to run
    run_backend = args.backend or (not args.frontend)
    run_frontend = args.frontend or (not args.backend)

    # Create and run process manager
    manager = ProcessManager()
    await manager.run(
        run_backend=run_backend,
        run_frontend=run_frontend,
        wait_for_health=not args.no_health_check
    )

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PDF Document Extraction System - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              Run both backend and frontend
  python main.py --backend    Run backend only
  python main.py --frontend   Run frontend only
  python main.py --check      Run dependency checks only
        """
    )

    parser.add_argument(
        "--backend",
        action="store_true",
        help="Run backend server only"
    )
    parser.add_argument(
        "--frontend",
        action="store_true",
        help="Run frontend server only"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run dependency checks only"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip pre-flight dependency checks"
    )
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="Skip health check wait"
    )

    args = parser.parse_args()

    # Print banner
    print()
    print(f"{Color.BOLD}{Color.BLUE}{'='*50}{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}  PDF Document Extraction System{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}  4-Agent Architecture with Anti-Hallucination{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}{'='*50}{Color.RESET}")
    print()

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print()
        log("Interrupted by user", Color.YELLOW)
        return 130


if __name__ == "__main__":
    sys.exit(main())
