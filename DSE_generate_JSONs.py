import subprocess
import numpy
import os
import math
import queue
import argparse


parser = argparse.ArgumentParser(
                prog = 'ProgramName',
                description = 'What the program does',
                epilog = 'Text at the bottom of help')

parser.add_argument('binary')           # positional argument
parser.add_argument('filename')           # positional argument
parser.add_arguemnt('-t', '--trainIterations')
parser.add_arguemnt('-r', '--regions')
parser.add_arguemnt('-e', '--executions')

args=parser.parse_args()

try:
    os.remove(args.filename)
except Exception:
    pass


fl=open(args.filename, 'wb')


if (args.executions is None):
    executions=50000
else:
    executions=int(args.executions)

workers_num=8

#procArr = numpy.full(-1, arrSize)
procArr=queue.Queue(workers_num)
fl.write(b'[')

first=True
for r in args.regions.split(","):
    for t in args.trainIterations.split(","):
        reg=int(r)
        train=int(t)

        procArr.put(subprocess.Popen([args.binary, "-r", str(int(reg)), "-t", str(int(train)), "-e", str(int(executions))], shell=True, stdout=subprocess.PIPE))
            
        if (procArr.full()):
            if (first):
                first=False
            else:
                fl.write(b',')
            fl.write(procArr.get().stdout.read())

while (not procArr.empty()):
    if (first):
        first=False
    else:
        fl.write(b',')
    fl.write(procArr.get().stdout.read())

fl.write(b']')
fl.flush()
fl.close()