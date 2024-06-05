from . import agent
import sys


if __name__ == "__main__":
    # parse arguments, TODO: use argparse
    agent.main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
