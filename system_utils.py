import psutil
import os
def printRAM():
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss/1e6)  # in bytes 
