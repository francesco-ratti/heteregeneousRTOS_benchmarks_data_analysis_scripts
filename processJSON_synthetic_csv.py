import ijson
import argparse
from math import isnan



import numpy as np
import pandas as pd
import math

def main():
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
    parser.add_argument('filename')           # positional argument
    parser.add_argument('outfilename')           # positional argument

    parser.add_argument('-n', '--name')
    parser.add_argument('-a', '--append')

#    parser.add_argument('-dse2dr', '--designspaceexploration2dregions')
    parser.add_argument('-fftT', '--fftcheckperiod')      # option that takes a value
    parser.add_argument('-fftsize', '--fftsize')      # option that takes a value

    args=parser.parse_args()


    filenames=args.filename.split(",")


    df = pd.DataFrame(columns=['Benchmark', 'regions', 'trainingIterations', 'testIterations', 'fp_rate', 'fn_rate', 'fn_relerr_mean','fn_relerr_var', 'fn_relerr_firstquartile', 'fn_relerr_median', 'fn_relerr_thirdquartile', 'fn_relerr_min', 'fn_relerr_max'])

    for flname in filenames:

        with open(flname, "rb") as f:
            for record in ijson.items(f, "item"):

                regions=int(record["regions"])
                trainIterations=int(record["trainIterations"])
                testIterations=int(record["testIterations"])

                total_pos=int(record["total_pos"])
                fp=int(record["false_pos"])
                tp=int(record["true_pos"])

                total_neg=int(record["true_neg"])+int(record["false_neg"])
                tn=int(record["true_neg"])
                fn=int(record["false_neg"])

                if (args.fftcheckperiod is not None and args.fftsize is not None):
                    fftperiodicity=int(args.fftcheckperiod)
                    fftsize=int(args.fftsize)
                    incr=(192)*fftsize*(1+fftsize/fftperiodicity) #in order to take into account instruction duplication, which timing overhead has been taken into account in timing analysis
                    total_neg=total_neg+incr
                    tn=tn+incr

                fn_rate=100*fn/total_neg
                fp_rate=100*fp/total_pos

                relErrNum = np.asarray(record["relerr"], dtype=np.uint32)
                relErrNum = relErrNum.view(dtype=np.float32)

                relErrVar=np.var(relErrNum)
                relErrMin=np.min(relErrNum)
                relErrMax=np.max(relErrNum)
                relErrMean=np.mean(relErrNum)

                print(f"Progress: regions {regions}, trainIterations {trainIterations}, testIterations {testIterations}\ntot pos {total_pos}, tp {tp}, fp {fp} fp rate {fp_rate} | tot neg {total_neg}, tn {tn}, fn {fn} fn rate {fn_rate} | rel err mean: {relErrMean} max: {relErrMax} min: {relErrMin} var: {relErrVar}\n") #| precision {precision}, recall {recall}, accuracy {accuracy}\n")
                
                df.loc[len(df.index)] = [args.name, regions, trainIterations, testIterations, fp_rate, fn_rate, relErrMean, relErrVar, np.percentile(relErrNum, 25), np.percentile(relErrNum, 50), np.percentile(relErrNum, 75), relErrMin, relErrMax]
    
    if (args.append is None):
        df.to_csv(args.outfilename, sep=';')
    else:
        df.to_csv(args.outfilename, sep=';', mode='a', index=False, header=False)

if __name__ == "__main__":
    main()