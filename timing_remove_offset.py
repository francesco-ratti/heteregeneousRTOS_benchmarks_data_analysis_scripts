
import argparse
import csv
import numpy as np
import math

def timerCyclesToNs(cycles):
    cyclestons=2000000000/650000000
    return round(cycles*cyclestons)

def main():
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
    parser.add_argument('filename')           # positional argument
    parser.add_argument('outfilename')           # positional argument
    parser.add_argument('decreaseby')      # option that takes a value
    args=parser.parse_args()

    datalist=np.fromfile(args.filename, dtype=int, sep='\n')
    datalist=datalist-np.array(int(args.decreaseby))
    datalist.tofile(args.outfilename, sep = '\n')

if __name__ == "__main__":
    main()
