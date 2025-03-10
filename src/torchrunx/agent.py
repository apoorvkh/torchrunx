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
from .utils.log_streaming import log_records_to_socket, redirect_stdio_to_logger
from .worker import WorkerArgs, worker_entrypoint


def main(
    launcher_hostname: str,
    launcher_port: int,
    world_size: int,
    rank: int,
    logger_hostname: str,
    logger_port: int,
    hostname: str,
) -> None:
    """Main function for agent processes (started on each node).

    This function spawns local worker processes (which run the target function). All agents monitor
    their worker statuses (including returned objects and raised exceptions) and communicate these
    with each other (and launcher). All agents terminate if failure occurs in any agent.

    Arguments:
        launcher_hostname: Hostname of the launcher process.
        launcher_port: Port for the process group on the launcher.
        world_size: Number of agents + 1 (launcher).
        rank: Rank of this agent.
        logger_hostname: Hostname of the logging server.
        logger_port: Port for the logging server.
        hostname: Hostname of this agent.
    """
    # Setup logging & stream logs to server

    log_records_to_socket(
        hostname=hostname, local_rank=None, logger_hostname=logger_hostname, logger_port=logger_port
    )

    logger = logging.getLogger()
    redirect_stdio_to_logger(logger)

    logger.debug("Initializing launcher-agent group.")

    launcher_agent_group = LauncherAgentGroup(
        launcher_hostname=launcher_hostname,
        launcher_port=launcher_port,
        world_size=world_size,
        rank=rank,
    )

    agent_rank = launcher_agent_group.rank - 1

    logger.debug("Synchronizing launcher and agents.")

    payload = AgentPayload(
        hostname=socket.getfqdn(),
        port=get_open_port(),
        process_id=os.getpid(),
    )

    launcher_payload, agent_payloads = launcher_agent_group.sync_payloads(payload=payload)

    hostname = launcher_payload.hostnames[agent_rank]
    worker_world_size = launcher_payload.worker_world_size
    worker_global_ranks = launcher_payload.worker_global_ranks[agent_rank]
    num_workers = len(worker_global_ranks)

    logger.info(f"Starting {num_workers} worker processes.")

    ctx = dist_mp.start_processes(
        name=f"{hostname}_",
        entrypoint=worker_entrypoint,
        args={
            i: (
                WorkerArgs(
                    function=launcher_payload.fn,
                    logger_hostname=logger_hostname,
                    logger_port=logger_port,
                    master_hostname=agent_payloads[0].hostname,
                    master_port=agent_payloads[0].port,
                    backend=launcher_payload.backend,
                    rank=worker_global_ranks[i],
                    local_rank=i,
                    node_rank=agent_rank,
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

    logger.debug("Entering worker monitoring and agent communication loop.")

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
                logger.info(f"Workers exited {'with' if any_failed else 'without'} errors.")
                break
    finally:
        ctx.close()
        sys.stdout.flush()
        sys.stderr.flush()
        launcher_agent_group.shutdown()

    logger.debug("Terminating agent process.")
