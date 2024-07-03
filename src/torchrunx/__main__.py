from argparse import ArgumentParser

from . import agent
from .utils import LauncherAgentGroup

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--launcher-hostname", type=str)
    parser.add_argument("--launcher-port", type=int)
    args = parser.parse_args()

    launcher_group = LauncherAgentGroup(
        world_size=args.world_size,
        rank=args.rank,
        launcher_hostname=args.launcher_hostname,
        launcher_port=args.launcher_port,
    )

    agent.main(launcher_group=launcher_group)
