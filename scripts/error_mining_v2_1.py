#!/usr/bin/env python3
from v2_2_workflow import main

if __name__ == "__main__":
    import sys

    sys.argv = [sys.argv[0], "error-mine-real", *sys.argv[1:]]
    raise SystemExit(main())
