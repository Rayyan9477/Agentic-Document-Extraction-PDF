"""
Worker management for Celery document processing.

Provides utilities for managing Celery workers,
including configuration, health checks, and scaling.
"""

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import get_logger
from src.queue.celery_app import celery_app


logger = get_logger(__name__)


class WorkerState(str, Enum):
    """Worker state enumeration."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(slots=True)
class WorkerConfig:
    """
    Worker configuration settings.

    Attributes:
        concurrency: Number of worker processes.
        queues: List of queues to consume from.
        loglevel: Logging level.
        hostname: Worker hostname pattern.
        pool: Pool implementation (prefork/eventlet/gevent).
        max_tasks_per_child: Tasks before worker restart.
        max_memory_per_child: Memory limit per child (KB).
        autoscale: Autoscale min,max workers.
        beat: Enable beat scheduler.
        events: Enable worker events.
        prefetch_multiplier: Number of tasks to prefetch.
    """

    concurrency: int = 4
    queues: list[str] = field(default_factory=lambda: [
        "document_processing",
        "batch_processing",
        "reprocessing",
    ])
    loglevel: str = "INFO"
    hostname: str = "worker@%h"
    pool: str = "prefork"
    max_tasks_per_child: int = 100
    max_memory_per_child: int = 512000  # 500 MB
    autoscale: tuple[int, int] | None = None
    beat: bool = False
    events: bool = True
    prefetch_multiplier: int = 1


class WorkerManager:
    """
    Manage Celery workers for document processing.

    Provides methods for starting, stopping, and monitoring workers.
    """

    def __init__(self, config: WorkerConfig | None = None) -> None:
        """
        Initialize the worker manager.

        Args:
            config: Worker configuration (uses defaults if not provided).
        """
        self.config = config or WorkerConfig()
        self._logger = logger
        self._worker_process: subprocess.Popen | None = None

    def build_worker_command(self) -> list[str]:
        """
        Build the Celery worker command.

        Returns:
            Command as list of arguments.
        """
        cmd = [
            sys.executable,
            "-m",
            "celery",
            "-A",
            "src.queue.celery_app",
            "worker",
            "--loglevel",
            self.config.loglevel,
            "--concurrency",
            str(self.config.concurrency),
            "--hostname",
            self.config.hostname,
            "--pool",
            self.config.pool,
            "--max-tasks-per-child",
            str(self.config.max_tasks_per_child),
            "--max-memory-per-child",
            str(self.config.max_memory_per_child),
            "--prefetch-multiplier",
            str(self.config.prefetch_multiplier),
        ]

        if self.config.queues:
            cmd.extend(["--queues", ",".join(self.config.queues)])

        if self.config.autoscale:
            min_workers, max_workers = self.config.autoscale
            cmd.extend(["--autoscale", f"{max_workers},{min_workers}"])

        if self.config.events:
            cmd.append("--events")

        return cmd

    def start_worker(self, background: bool = True) -> dict[str, Any]:
        """
        Start a Celery worker.

        Args:
            background: Run worker in background.

        Returns:
            Start result with worker info.
        """
        if self._worker_process and self._worker_process.poll() is None:
            return {
                "status": "already_running",
                "pid": self._worker_process.pid,
            }

        cmd = self.build_worker_command()

        self._logger.info(
            "worker_starting",
            command=" ".join(cmd),
            concurrency=self.config.concurrency,
            queues=self.config.queues,
        )

        try:
            if background:
                self._worker_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )

                return {
                    "status": "started",
                    "pid": self._worker_process.pid,
                    "command": " ".join(cmd),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
            else:
                # Run in foreground (blocking)
                subprocess.run(cmd, check=True)
                return {
                    "status": "completed",
                }

        except Exception as e:
            self._logger.error("worker_start_failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }

    def stop_worker(self, graceful: bool = True) -> dict[str, Any]:
        """
        Stop the worker.

        Args:
            graceful: Wait for current tasks to complete.

        Returns:
            Stop result.
        """
        if not self._worker_process:
            return {
                "status": "not_running",
            }

        if self._worker_process.poll() is not None:
            return {
                "status": "already_stopped",
            }

        self._logger.info(
            "worker_stopping",
            pid=self._worker_process.pid,
            graceful=graceful,
        )

        try:
            if graceful:
                self._worker_process.terminate()
                self._worker_process.wait(timeout=30)
            else:
                self._worker_process.kill()
                self._worker_process.wait(timeout=5)

            return {
                "status": "stopped",
                "pid": self._worker_process.pid,
            }

        except subprocess.TimeoutExpired:
            self._worker_process.kill()
            return {
                "status": "force_killed",
                "pid": self._worker_process.pid,
            }

        except Exception as e:
            self._logger.error("worker_stop_failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }

    def get_worker_status(self) -> dict[str, Any]:
        """
        Get status of worker processes.

        Returns:
            Worker status information.
        """
        # Get registered workers from Celery
        inspect = celery_app.control.inspect()

        try:
            active = inspect.active() or {}
            reserved = inspect.reserved() or {}
            scheduled = inspect.scheduled() or {}
            stats = inspect.stats() or {}
            registered = inspect.registered() or {}

            workers = []
            for worker_name in set(active.keys()) | set(stats.keys()):
                worker_stats = stats.get(worker_name, {})
                worker_active = active.get(worker_name, [])
                worker_reserved = reserved.get(worker_name, [])

                workers.append({
                    "name": worker_name,
                    "state": WorkerState.RUNNING.value,
                    "active_tasks": len(worker_active),
                    "reserved_tasks": len(worker_reserved),
                    "total": worker_stats.get("total", {}),
                    "pool": worker_stats.get("pool", {}),
                    "broker": worker_stats.get("broker", {}),
                })

            return {
                "status": "ok" if workers else "no_workers",
                "worker_count": len(workers),
                "workers": workers,
                "registered_tasks": list(registered.get(list(registered.keys())[0], [])) if registered else [],
            }

        except Exception as e:
            self._logger.error("worker_status_failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }

    def get_queue_stats(self) -> dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Queue statistics.
        """
        try:
            inspect = celery_app.control.inspect()
            active = inspect.active() or {}
            reserved = inspect.reserved() or {}

            queue_stats = {}
            for queue_name in self.config.queues:
                queue_stats[queue_name] = {
                    "active": 0,
                    "reserved": 0,
                }

            # Count tasks per queue
            for worker_tasks in active.values():
                for task in worker_tasks:
                    queue = task.get("delivery_info", {}).get("routing_key", "default")
                    if queue in queue_stats:
                        queue_stats[queue]["active"] += 1

            for worker_tasks in reserved.values():
                for task in worker_tasks:
                    queue = task.get("delivery_info", {}).get("routing_key", "default")
                    if queue in queue_stats:
                        queue_stats[queue]["reserved"] += 1

            return {
                "status": "ok",
                "queues": queue_stats,
            }

        except Exception as e:
            self._logger.error("queue_stats_failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }

    def purge_queue(self, queue_name: str) -> dict[str, Any]:
        """
        Purge all messages from a queue.

        Args:
            queue_name: Name of queue to purge.

        Returns:
            Purge result.
        """
        if queue_name not in self.config.queues:
            return {
                "status": "error",
                "error": f"Unknown queue: {queue_name}",
            }

        try:
            count = celery_app.control.purge()

            self._logger.info(
                "queue_purged",
                queue=queue_name,
                count=count,
            )

            return {
                "status": "ok",
                "queue": queue_name,
                "purged_count": count,
            }

        except Exception as e:
            self._logger.error("queue_purge_failed", queue=queue_name, error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }

    def scale_workers(self, concurrency: int) -> dict[str, Any]:
        """
        Scale worker concurrency.

        Args:
            concurrency: New concurrency level.

        Returns:
            Scale result.
        """
        if concurrency < 1:
            return {
                "status": "error",
                "error": "Concurrency must be at least 1",
            }

        try:
            celery_app.control.pool_resize(concurrency)

            self._logger.info(
                "workers_scaled",
                new_concurrency=concurrency,
            )

            return {
                "status": "ok",
                "new_concurrency": concurrency,
            }

        except Exception as e:
            self._logger.error("worker_scale_failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }

    def broadcast_shutdown(self, graceful: bool = True) -> dict[str, Any]:
        """
        Broadcast shutdown signal to all workers.

        Args:
            graceful: Wait for current tasks to complete.

        Returns:
            Shutdown result.
        """
        try:
            if graceful:
                celery_app.control.broadcast("shutdown")
            else:
                celery_app.control.broadcast("pool_restart", arguments={"reload": False})

            self._logger.info(
                "shutdown_broadcast",
                graceful=graceful,
            )

            return {
                "status": "ok",
                "graceful": graceful,
            }

        except Exception as e:
            self._logger.error("shutdown_failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check on workers.

        Returns:
            Health check result.
        """
        try:
            ping_result = celery_app.control.ping(timeout=5)

            if not ping_result:
                return {
                    "healthy": False,
                    "reason": "No workers responding",
                    "workers": [],
                }

            responding_workers = []
            for worker_response in ping_result:
                for worker_name, response in worker_response.items():
                    if response.get("ok") == "pong":
                        responding_workers.append(worker_name)

            return {
                "healthy": True,
                "workers": responding_workers,
                "count": len(responding_workers),
            }

        except Exception as e:
            return {
                "healthy": False,
                "reason": str(e),
                "workers": [],
            }
