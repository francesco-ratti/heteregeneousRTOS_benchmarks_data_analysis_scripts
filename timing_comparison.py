
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
    parser.add_argument('-b', "--base")
    parser.add_argument('-measoffset', "--measurementsoffset")
    args=parser.parse_args()


    datalist=np.fromfile(args.filename, dtype=int, sep='\n')
    if (args.measurementsoffset):
        datalist=datalist-np.array(int(args.measurementsoffset))
    
    mean=datalist.mean()
    std=datalist.std()
    var=datalist.var()
    max=datalist.max()
    min=datalist.min()

    if (args.base is not None):
        datalistbase=np.fromfile(args.base, dtype=int, sep='\n')
        if (args.measurementsoffset):
            datalistbase=datalistbase-np.array(int(args.measurementsoffset))
        basemean=datalistbase.mean()
        basemin=datalistbase.min()
        basemax=datalistbase.max()
        ovh=round((100*(max-basemax)/basemax), 2)

   # print(f"MEAN: {timerCyclesToNs(mean)} STD DEV: {timerCyclesToNs(std)} VAR: {timerCyclesToNs(var)} MAX: {timerCyclesToNs(max)} MIN: {timerCyclesToNs(min)}")
    if (args.base is None):
#        print(f"MEAN: {mean} STD DEV: {std} VAR: {var} MAX: {max} MIN: {min}")
        print(f"{timerCyclesToNs(mean)} & {timerCyclesToNs(max)} & {timerCyclesToNs(min)} & {timerCyclesToNs(var)} & - ")
    else:
#        print(f"MEAN: {mean} STD DEV: {std} VAR: {var} MAX: {max} MIN: {min} BASEMEAN: {basemean}")
        print(f"{timerCyclesToNs(mean)} & {timerCyclesToNs(max)} & {timerCyclesToNs(min)} & {timerCyclesToNs(var)} & {ovh}")

if __name__ == "__main__":
    main()
