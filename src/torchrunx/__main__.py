"""CLI entrypoint used for starting agents on different nodes."""

from argparse import ArgumentParser

from .agent import main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--launcher-hostname", type=str)
    parser.add_argument("--launcher-port", type=int)
    parser.add_argument("--logger-port", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--hostname", type=str)
    parser.add_argument("--agent-timeout", type=int, default=30)
    args = parser.parse_args()

    main(
        launcher_hostname=args.launcher_hostname,
        launcher_port=args.launcher_port,
        world_size=args.world_size,
        rank=args.rank,
        logger_hostname=args.launcher_hostname,
        logger_port=args.logger_port,
        hostname=args.hostname,
        agent_timeout=args.agent_timeout,
    )
