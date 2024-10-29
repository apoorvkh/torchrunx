"""Primary logic for agent processes."""

from __future__ import annotations

__all__ = ["main"]

import logging
import os
import socket
import sys
import tempfile

import torch
import torch.distributed.elastic.multiprocessing as dist_mp

from .utils.comm import (
    AgentPayload,
    AgentStatus,
    LauncherAgentGroup,
    get_open_port,
)
from .utils.logging import log_records_to_socket, redirect_stdio_to_logger
from .worker import WorkerArgs, worker_entrypoint


def main(launcher_agent_group: LauncherAgentGroup, logger_hostname: str, logger_port: int) -> None:
    """Main function for agent processes (started on each node).

    This function spawns local worker processes (which run the target function). All agents monitor
    their worker statuses (including returned objects and raised exceptions) and communicate these
    with each other (and launcher). All agents terminate if failure occurs in any agent.

    Arguments:
        launcher_agent_group: The communication group between launcher and all agents.
        logger_hostname: The hostname of the launcher (for logging).
        logger_port: The port of the launcher (for logging).
    """
    agent_rank = launcher_agent_group.rank - 1

    # Communicate initial payloads between launcher/agents

    payload = AgentPayload(
        hostname=socket.getfqdn(),
        port=get_open_port(),
        process_id=os.getpid(),
    )

    launcher_payload, agent_payloads = launcher_agent_group.sync_payloads(payload=payload)
    main_agent_payload = agent_payloads[0]

    hostname = launcher_payload.hostnames[agent_rank]
    worker_world_size = launcher_payload.worker_world_size
    worker_global_ranks = launcher_payload.worker_global_ranks[agent_rank]
    num_workers = len(worker_global_ranks)

    # Stream logs to logging server

    logger = logging.getLogger()

    log_records_to_socket(
        logger=logger,
        hostname=hostname,
        local_rank=None,
        logger_hostname=logger_hostname,
        logger_port=logger_port,
    )

    redirect_stdio_to_logger(logger)

    # Spawn worker processes

    ctx = dist_mp.start_processes(
        name=f"{hostname}_",
        entrypoint=worker_entrypoint,
        args={
            i: (
                WorkerArgs(
                    function=launcher_payload.fn,
                    logger_hostname=logger_hostname,
                    logger_port=logger_port,
                    main_agent_hostname=main_agent_payload.hostname,
                    main_agent_port=main_agent_payload.port,
                    backend=launcher_payload.backend,
                    rank=worker_global_ranks[i],
                    local_rank=i,
                    local_world_size=num_workers,
                    world_size=worker_world_size,
                    hostname=launcher_payload.hostnames[agent_rank],
                    timeout=launcher_payload.timeout,
                ).serialize(),
            )
            for i in range(num_workers)
        },
        # environment variables from agent are already automatically copied to workers
        envs={i: {} for i in range(num_workers)},
        # we handle logging ourselves, so we can discard these
        **(
            {"logs_specs": dist_mp.DefaultLogsSpecs(log_dir=tempfile.mkdtemp())}
            if torch.__version__ >= "2.3"
            else {"log_dir": tempfile.mkdtemp()}
        ),  # pyright: ignore [reportArgumentType]
    )

    # Monitor and communicate agent statuses
    # Terminate gracefully upon failure

    try:
        status = None
        while True:
            if status is None or status.state == "running":
                # status can contain ExceptionFromWorker or WorkerFailedError
                status = AgentStatus.from_result(result=ctx.wait(5))

            # can raise AgentFailedError in launcher and all agents
            agent_statuses = launcher_agent_group.sync_agent_statuses(status=status)

            all_done = all(s.state == "done" for s in agent_statuses)
            any_failed = any(s.state == "failed" for s in agent_statuses)
            if all_done or any_failed:
                break
    finally:
        ctx.close()
        sys.stdout.flush()
        sys.stderr.flush()
