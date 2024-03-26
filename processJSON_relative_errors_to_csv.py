import ijson
import argparse
from math import isnan



import numpy as np
import pandas as pd
import math

parser = argparse.ArgumentParser(
                prog = 'ProgramName',
                description = 'What the program does',
                epilog = 'Text at the bottom of help')
parser.add_argument('filename')           # positional argument
parser.add_argument('outfilename')           # positional argument

parser.add_argument('-t', '--trainiterations')
parser.add_argument('-r', '--regions')
parser.add_argument('-a', '--append')
parser.add_argument('-tmin', '--trainmin')      # option that takes a value
parser.add_argument('-tmax', '--trainmax')      # option that takes a value

args=parser.parse_args()


filenames=args.filename.split(",")

columns=[]
if (args.regions is None):
    columns.append('regions')
if (args.trainiterations is None):
    columns.append('trainiterations')
columns.append('relerr')

df = pd.DataFrame(columns=columns)

for flname in filenames:

    with open(flname, "rb") as f:
        for record in ijson.items(f, "item"):
            regions=int(record["regions"])
            trainIterations=int(record["trainIterations"])
            if ((args.regions is None or regions==int(args.regions)) and (args.train is None or trainIterations==int(args.train)) and (args.trainmin is None or trainIterations>=int(args.trainmin)) and (args.trainmax is None or trainIterations<=int(args.trainmax))):

                relErrNum = np.asarray(record["relerr"], dtype=np.uint32)
                relErrNum = relErrNum.view(dtype=np.float32)

                datadict={}
                if (args.regions is None):
                    datadict.put('regions', np.full(len(relErrNum), regions))
                if (args.trainiterations is None):
                    datadict.put('trainIterations', np.full(len(relErrNum), trainIterations))

                datadict.put(relErrNum)

                df.loc[len(df.index)] = [df = pd.concat( 
                        [ df, pd.DataFrame( data = datadict ) ]
                        )]

if (args.append is None):
    df.to_csv(args.outfilename, sep=';')
else:
    df.to_csv(args.outfilename, sep=';', mode='a', index=False, header=False)