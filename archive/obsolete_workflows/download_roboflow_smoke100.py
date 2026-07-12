#!/usr/bin/env python3
from smoke100_workflow import main

if __name__ == "__main__":
    import sys

    sys.argv = [sys.argv[0], "download", *sys.argv[1:]]
    raise SystemExit(main())
