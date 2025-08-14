# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
