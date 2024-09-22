import ijson
import argparse
from math import isnan


import numpy as np
import pandas as pd
import math

#def log_tick_formatter(val, pos=None):
#    return f"$10^{{{int(val)}}}$"  # remove int() if you don't use MaxNLocator


parser = argparse.ArgumentParser(
                prog = 'ProgramName',
                description = 'What the program does',
                epilog = 'Text at the bottom of help')
parser.add_argument('filename')           # positional argument
parser.add_argument('outfilename')           # positional argument

parser.add_argument('-n', '--name')
parser.add_argument('-a', '--append')

parser.add_argument('-r', '--regions')      # option that takes a value
parser.add_argument('-t', '--train')      # option that takes a value
parser.add_argument('-tmin', '--trainmin')      # option that takes a value
parser.add_argument('-tmax', '--trainmax')      # option that takes a value
parser.add_argument('-m', '--multiplier')      # option that takes a value

args=parser.parse_args()

fntresholds=np.array([0.01, 0.10, 0.20, 0.50, 1, 2, 5, 10])
if args.multiplier:
    fntresholds=np.multiply(fntresholds, float(args.multiplier))
fntresholds=np.insert(fntresholds, 0, 0)
#fntresholds=np.geomspace(float(args.falsenegativethresholdmin)/100, float(args.falsenegativethresholdmax)/100, num=int(args.samples), endpoint=True, dtype=float)

filenames=args.filename.split(",")

df = pd.DataFrame()

#print(fntresholds)

for flname in filenames:
    with open(flname, "rb") as f:
        for record in ijson.items(f, "item"):
            fn_withthresh=np.zeros(len(fntresholds), dtype=int)
            regions=int(record["regions"])
            trainIterations=int(record["trainIterations"])
            if ((args.regions is None or regions==int(args.regions)) and (args.train is None or trainIterations==int(args.train)) and (args.trainmin is None or trainIterations>=int(args.trainmin)) and (args.trainmax is None or trainIterations<=int(args.trainmax))):
                # testIterations=int(record["testIterations"])

                # total_pos=int(record["total_pos"])
                # fp=int(record["false_pos"])
                # tp=int(record["true_pos"])

                total_neg=int(record["true_neg"])+int(record["false_neg"])
                tn=int(record["true_neg"])
                fn=int(record["false_neg"])

                # if (args.fftcheckperiod is not None and args.fftsize is not None):
                #     fftperiodicity=int(args.fftcheckperiod)
                #     fftsize=int(args.fftsize)
                #     incr=(192)*fftsize*(1+fftsize/fftperiodicity) #in order to take into account instruction duplication, which timing overhead has been taken into account in timing analysis
                #     total_neg=total_neg+incr
                #     tn=tn+incr

                # fn_rate=100*fn/total_neg
                # fp_rate=100*fp/total_pos

                relErrNum = np.asarray(record["relerr"], dtype=np.uint32)
                relErrNum = relErrNum.view(dtype=np.float32)

                print(f"processing reg {regions}, {trainIterations} trainIteration,  {len(relErrNum)} values")
                # relErrMean=np.mean(relErrNum)*100
                # relErrVar=np.var(relErrNum)*100
                # relErrMin = np.min(relErrNum)*100
                # relErrMax = np.max(relErrNum)*100

                # region_relerrmin.append(relErrMin)
                # region_relerrmax.append(relErrMax)

                # print(f"regions {regions}, trainIterations {trainIterations}, testIterations {testIterations}\ntot pos {total_pos}, tp {tp}, fp {fp} fp rate {fp_rate} | tot neg {total_neg}, tn {tn}, fn {fn} fn rate {fn_rate} | rel err mean: {relErrMean} max: {relErrMax} var: {relErrVar}\n") #| precision {precision}, recall {recall}, accuracy {accuracy}\n")
                # #for charts generation
                # regions_x.append(regions)
                # regions_trainIterations.append(trainIterations)
                # fp_r_not_clamped=fp*100/total_pos
                # fp_r_clamped= 4 if fp_r_not_clamped > 4 else fp_r_not_clamped
                # regions_fp_rate.append(fp_r_clamped)
                # regions_fn_rate.append(fn*100/total_neg)


                # if (args.designspaceexploration2dregions is not None and (int(args.designspaceexploration2dregions)==regions) or args.single is not None):
                #     region_fp_rate.append(fp_r_not_clamped)
                #     region_trainIterations.append(trainIterations)
                #     region_relerrmean.append(relErrMean)
                #     region_relerrvariance.append(relErrVar)
                #     region_fn_rate.append(fn_rate)

                    #relErrNum = np.asarray(record["relerr"], dtype=np.uint32)
                    #relErrNum = relErrNum.view(dtype=np.float32)
                fn_withthresh[0]=fn
                total_neg_tresh=tn+len(relErrNum)
                if (total_neg!=total_neg_tresh):
                    print("ERROR! total_neg!=total_neg_tresh")

                count=0
                for er in relErrNum:
                    for thri in range(1, len(fntresholds)):
                        if (isnan(er)):
                            print("ISNAN")
                        else:
                            if (er>fntresholds[thri]):
                                fn_withthresh[thri]=fn_withthresh[thri]+1
                            else:
                                break
                #                        df.loc[len(df.index)] = [args.name, regions, trainIterations, testIterations, fntresholds, fn_withthresh]                        
                    count=count+1
                df = pd.concat( 
                    [ df, pd.DataFrame( data = { 'name': np.full(len(fntresholds), args.name), 'regions': np.full(len(fntresholds), regions), 'trainingIterations': np.full(len(fntresholds), trainIterations), 'threshold': fntresholds*100, 'fn_rate': (fn_withthresh/total_neg)*100 } ) ]
                    )

if (args.append is None):
    df.to_csv(args.outfilename, sep=';')
else:
    df.to_csv(args.outfilename, sep=';', mode='a', index=False, header=False)