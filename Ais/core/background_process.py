import threading
from itertools import count

class BackgroundProcess:
    idgen = count(0)

    def __init__(self, function, args, name=None):
        self.uid = next(BackgroundProcess.idgen)
        self.function = function
        self.args = args
        self.name = name
        self.thread = None
        self.progress = 0.0
        self.stop_request = threading.Event()

    def start(self):
        _name = f"BackgroundProcess {self.uid} - "+(self.name if self.name is not None else "")
        self.thread = threading.Thread(daemon=True, target=self._run, name=_name)
        self.thread.start()

    def _run(self):
        self.function(*self.args, self)

    def set_progress(self, progress):
        self.progress = progress

    def stop(self):
        self.stop_request.set()
        self.progress = 1.0

    def __str__(self):
        return f"BackgroundProcess {self.uid} with function {self.function} and args {self.args}"