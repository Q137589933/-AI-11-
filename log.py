# log.py
import sys


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



# 下面所有的方法，只要控制台输出，都将写入"D:\\1.txt"
