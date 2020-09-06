import argparse

from model.gat import GAT
from generate import generate_sample


def main():
    loc, marker, label = generate_sample(200, 0, False)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    main()
