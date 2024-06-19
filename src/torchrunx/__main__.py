from argparse import ArgumentParser

from . import agent

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--launcher-ip", type=str)
    parser.add_argument("--launcher-port", type=int)
    args = parser.parse_args()

    agent.main(
        world_size=args.world_size,
        rank=args.rank,
        launcher_ip=args.launcher_ip,
        launcher_port=args.launcher_port,
    )
