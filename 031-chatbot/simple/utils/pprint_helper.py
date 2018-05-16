import sys


class Head(object):
    def __init__(self, lines, fd=sys.stdout):
        self.lines = lines
        self.fd = fd

    def write(self, msg):
        if self.lines <= 0: return
        n = msg.count('\n')
        if n < self.lines:
            self.lines -= n
            return self.fd.write(msg)
        ix = 0

        while(self.lines > 0):
            iy = msg.find('\n', ix + 1)
            self.lines -= 1
            ix = iy
        return self.fd.write(msg[:ix])