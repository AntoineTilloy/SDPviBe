"""python -m sdpvibe ...  (same command line as SDPB's `sdpb`; see sdpvibe/cli.py)"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # streaming: less fragmentation
import sys
from sdpvibe.cli import main

sys.exit(main())
