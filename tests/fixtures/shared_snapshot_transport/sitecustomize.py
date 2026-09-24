"""Install invented HTTP bodies only; production collector and policy remain real."""
import os

if os.environ.get("GREYHOUND_SHARED_SNAPSHOT_FIXTURE"):
    from snapshot_transport import install
    install()
