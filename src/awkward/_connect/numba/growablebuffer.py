# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numba
import numba.core.typing
import numba.core.typing.ctypes_utils
import numpy


class GrowableBuffer:
    def __init__(self, dtype, *, initial=1024, resize=10.0):
        self._panels = [numpy.zeros((initial,), dtype=dtype)]
        self._last_panel = self._panels[-1]
        self._length = 0
        self._last_panel_length = 0
        self._resize = resize

    def __repr__(self):
        return f"<GrowableBuffer({self._last_panel.dtype!r}) len {self._length}>"

    def __len__(self):
        return self._length

    def append(self, datum):
        if self._last_panel_length == len(self._last_panel):
            self._add_panel()

        self._last_panel[self._last_panel_length] = datum
        self._last_panel_length += 1
        self._length += 1

    def extend(self, data):
        panel_index = len(self._panels) - 1
        pos = self._last_panel_length

        available = len(self._last_panel) - self._last_panel_length
        while len(data) > available:
            self._add_panel()
            available += len(self._last_panel)

        remaining = len(data)
        while remaining > 0:
            panel = self._panels[panel_index]
            available_in_panel = len(panel) - pos
            to_write = min(remaining, available_in_panel)

            start = len(data) - remaining
            panel[pos : pos + to_write] = data[start : start + to_write]

            if panel_index == len(self._panels) - 1:
                self._last_panel_length += to_write
            remaining -= to_write
            pos = 0
            panel_index += 1

        self._length += len(data)

    def _add_panel(self):
        panel_length = len(self._last_panel)
        if len(self._panels) == 1:
            # only resize the first time, and by a large factor (C++ should do this, too!)
            panel_length = int(numpy.ceil(panel_length * self._resize))

        self._last_panel = numpy.zeros((panel_length,), dtype=self._last_panel.dtype)
        self._panels.append(self._last_panel)
        self._last_panel_length = 0

    def snapshot(self):
        out = numpy.zeros((self._length,), dtype=self._last_panel.dtype)

        start = 0
        stop = 0
        for panel in self._panels[:-1]:  # full panels, not including the last
            stop += len(panel)
            out[start:stop] = panel
            start = stop

        stop += self._last_panel_length
        out[start:stop] = self._last_panel[:self._last_panel_length]

        return out
