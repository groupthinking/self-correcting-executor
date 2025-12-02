import argparse


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent")
    parser.add_argument("--context", help="JSON context string")
    args = parser.parse_args()
    print(f"[MCP] Running agent: {args.agent} with context {args.context}")


if __name__ == "__main__":
    run()
