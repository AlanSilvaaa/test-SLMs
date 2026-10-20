"""Sample host RAM and the local experiment process tree."""

import os
import threading

import psutil


class MemoryMonitor:
    def __init__(self):
        self.pid = os.getpid()
        self.stop = threading.Event()
        self.samples = []
        self.errors = []

    def sample(self):
        vm, swap = psutil.virtual_memory(), psutil.swap_memory()
        rss = None
        if self.pid is not None:
            try:
                parent = psutil.Process(self.pid)
                rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)])
            except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
                self.errors.append(str(error))
        self.samples.append(
            {
                "host_used_bytes": vm.used,
                "swap_used_bytes": swap.used,
                "process_tree_rss_bytes": rss,
            }
        )

    def loop(self):
        while not self.stop.wait(0.1):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()

    def result(self):
        return {
            "baseline": self.samples[0],
            "peak": {
                key: max(values)
                if (values := [s[key] for s in self.samples if s[key] is not None])
                else None
                for key in self.samples[0]
            },
            "sampling_interval_s": 0.1,
            "errors": self.errors,
        }
