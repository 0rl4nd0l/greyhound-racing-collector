"""Lower HTTP transport fixture inherited by actual spawn workers."""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def install():
    import requests
    fixture = json.loads(Path(os.environ["GREYHOUND_SHARED_SNAPSHOT_FIXTURE"]).read_text())

    def send(adapter, request, **kwargs):
        url = urlsplit(request.url)
        key = url.netloc + url.path
        assert key in fixture["responses"], "unconfigured synthetic transport: " + key
        row = fixture["responses"][key]
        fd = os.open(fixture["log"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, (json.dumps({"pid": os.getpid(), "path": url.path, "host": url.netloc}) + "\n").encode())
        finally:
            os.close(fd)
        response = requests.Response()
        response.status_code = row.get("status", 200)
        response.url, response.request = request.url, request
        response.headers = {"Content-Type": row.get("content_type", "text/html")}
        response._content = row["body"].encode()
        return response

    requests.adapters.HTTPAdapter.send = send
