import time
from collections import OrderedDict
import sys
import curses


class TimeProbe():
    def __init__(self, window_size=None):
        self.sessions = OrderedDict()
        self.window_size = window_size
        self.console = curses.initscr()
        self.strings = OrderedDict()
        self.palette = {
            0: 0,
            "blue": 1,
            "red": 2,
            "green": 3
        }
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLUE, -1)
        curses.init_pair(2, curses.COLOR_RED, -1)
        curses.init_pair(3, curses.COLOR_GREEN, -1)
        

    def __getAvg(self, session_idx):
        if self.window_size is not None:
            return sum(self.sessions[session_idx]['window']) / len(self.sessions[session_idx]['window'])
        else:
            if self.sessions[session_idx]['num'] == 0:
                return 0
            return self.sessions[session_idx]['sum']/self.sessions[session_idx]['num']

    def start(self, session_idx):
        if not session_idx in self.sessions:
            self.sessions[session_idx] = {}
            if self.window_size is not None:
                self.sessions[session_idx]['window'] = []
            else:
                self.sessions[session_idx]['sum'] = 0
                self.sessions[session_idx]['num'] = 0
        self.sessions[session_idx]['start_time'] = time.time()

    def end(self, session_idx):
        assert session_idx in self.sessions, f"{session_idx} not in existed sessions."
        if self.window_size is not None:
            data = time.time() - self.sessions[session_idx]['start_time']
            self.sessions[session_idx]['window'].append(data)
            if len(self.sessions[session_idx]['window']) > self.window_size:
                self.sessions[session_idx]['window'].pop(0)
        else:
            self.sessions[session_idx]['sum'] += (
                time.time() - self.sessions[session_idx]['start_time'])
            self.sessions[session_idx]['num'] += 1

    def print(self, *strings, color=0):
        assert color in self.palette.keys(), "Color not in palette"
        clr = self.palette[color]
        print_idx = str(sys._getframe().f_back.f_lineno)
        for i, string in enumerate(strings):
            for j, line in enumerate(str(string).splitlines()):
                self.strings[print_idx+"_"+str(i)+str(j)] = {
                    "string": str(line),
                    "color": clr
                }

    def refresh(self):
        try:
            self.console.clear()
            for i, print_idx in enumerate(self.strings.keys()):
                self.console.addstr(i, 0, self.strings[print_idx]["string"], curses.color_pair(
                    self.strings[print_idx]["color"]))
            self.console.refresh()
        except Exception as e:
            print("Can't display in console properly, error:", e)

    def getResults(self, session_indices=None):
        """Get recorded time results.
        Args:
            session_indices (list, optional): List of session index to be print out. Defaults to None, means print out all sessions.
        Returns:
            str: results
        """
        if session_indices is None:
            session_indices = self.sessions.keys()
        strings = "[TIME USAGE] "
        for session_idx in session_indices:
            strings += f"{session_idx}: {self.__getAvg(session_idx):.5f} "
        return strings
