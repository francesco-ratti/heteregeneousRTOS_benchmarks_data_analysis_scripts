
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
    parser.add_argument('-ovh', '--overhead')      # option that takes a value
    args=parser.parse_args()


    datalist=np.fromfile(args.filename, dtype=int, sep='\n', count=150000)

    mean=datalist.mean()
    std=datalist.std()
    var=datalist.var()
    max=datalist.max()
    min=datalist.min()
    
    if(args.overhead):
        mean=mean-int(args.overhead)
        max=max-int(args.overhead)
        min=min-int(args.overhead)

    print(f"MEAN: {mean} STD DEV: {std} VAR: {var} MAX: {max} MIN: {min}")
    print(f"MEAN: {timerCyclesToNs(mean)} ns STD DEV: {timerCyclesToNs(std)} ns VAR: {timerCyclesToNs(var)} ns MAX: {timerCyclesToNs(max)} ns MIN: {timerCyclesToNs(min)} ns")


if __name__ == "__main__":
    main()
