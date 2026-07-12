#!/usr/bin/env python3
from v2_2_workflow import main

if __name__ == "__main__":
    import sys

    command = "smoke-train" if "--smoke-test" in sys.argv else "train"
    sys.argv = [sys.argv[0], command, *[arg for arg in sys.argv[1:] if arg != "--smoke-test"]]
    raise SystemExit(main())
