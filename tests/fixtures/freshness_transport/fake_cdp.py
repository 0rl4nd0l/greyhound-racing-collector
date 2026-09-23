"""Fabricated browser transport; real child ownership and asynchronous CDP seam."""
import json
import subprocess
import sys
import threading
from types import SimpleNamespace


class BrowserTransport:
    def __init__(self):
        self.process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
        self.stopped = threading.Event()
        self._ws = SimpleNamespace(on_message=lambda *args: None,
                                   on_close=lambda *args: None,
                                   on_error=lambda *args: None,
                                   close=lambda: self._ws.on_close(None))

    def execute(self, command):
        request = next(command)
        if request['method'] == 'Page.stopLoading':
            self.stopped.set()
        try:
            command.send({})
        except StopIteration as result:
            return result.value

    def response(self, url, status, headers=None, resource_type="Document"):
        self._ws.on_message(None, json.dumps({
            'method': 'Network.responseReceived',
            'params': {'type': resource_type, 'response': {
                'url': url, 'status': status, 'headers': headers or {}}}}))

    def close(self):
        pass

    def quit(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=5)
