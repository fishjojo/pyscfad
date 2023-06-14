import sys
import time

class Timer:
    def __init__(self, stdout=None):
        if stdout is None:
            self.stdout = sys.stdout
        else:
            self.stdout = stdout

        self._t0, self._w0 = (time.process_time(), time.perf_counter())

    def timer(self, msg, stdout=None):
        if stdout is None:
            stdout = self.stdout

        t0, w0 = (self._t0, self._w0)
        t1, w1 = (time.process_time(), time.perf_counter())
        self._t0, self._w0 = (t1, w1)
        stdout.write(f'    CPU time for {msg}'
                     f' {t1-t0: 9.2f} sec,'
                     f' wall time {w1-w0: 9.2f} sec\n')
        stdout.flush()
