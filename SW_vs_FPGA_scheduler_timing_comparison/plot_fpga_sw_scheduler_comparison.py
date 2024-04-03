import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

'''
def computewcetsw(wcetinswschedulerticks, swscheduleroverhead_reexec, swscheduleroverhead_noreexec, swschedulerjobendoverhead):
    return np.divide((wcetinswschedulerticks+swscheduleroverhead_reexec+(wcetinswschedulerticks-1)*swscheduleroverhead_noreexec+swschedulerjobendoverhead), wcetinswschedulerticks, dtype=float)
def computewcetfpga(wcetinswschedulerticks, fpgascheduleroverhead_reexec, fpgascheduleroverhead_noreexec, fpgaschedulerjobendoverhead):
    return np.divide(wcetinswschedulerticks+fpgascheduleroverhead_reexec+fpgascheduleroverhead_noreexec+fpgaschedulerjobendoverhead, wcetinswschedulerticks, dtype=float)
'''
def computewcetsw(wcetinswschedulerticks, tick, swscheduleroverhead_reexec, swscheduleroverhead_noreexec, swschedulerjobendoverhead):
    return (-1+np.divide(np.ceil(
        np.divide(swscheduleroverhead_reexec-swscheduleroverhead_noreexec+swschedulerjobendoverhead+wcetinswschedulerticks, 1-np.divide(swscheduleroverhead_noreexec, tick) )
        ),
                     wcetinswschedulerticks, dtype=float))*100
def computewcetfpga(wcetinswschedulerticks, tick, graphtick, fpgascheduleroverhead_reexec, fpgascheduleroverhead_noreexec, fpgaschedulerjobendoverhead):
    return (-1+np.divide(np.ceil(np.divide(wcetinswschedulerticks*graphtick
                                       +fpgascheduleroverhead_reexec+fpgascheduleroverhead_noreexec+fpgaschedulerjobendoverhead
                                       , tick))
                     , (np.divide(np.multiply(wcetinswschedulerticks, graphtick), tick))
                     , dtype=float))*100

'''
def computewcetswfromtime(wcet, tick, swscheduleroverhead_reexec, swscheduleroverhead_noreexec, swschedulerjobendoverhead):
    return np.divide((np.ceil(wcet/tick)+swscheduleroverhead_reexec+(np.ceil(wcet/tick)-1)*swscheduleroverhead_noreexec+swschedulerjobendoverhead), wcetinswschedulerticks, dtype=float)
def computewcetfpgafromtime(wcet, tick, fpgascheduleroverhead_reexec, fpgascheduleroverhead_noreexec, fpgaschedulerjobendoverhead):
    return np.divide(np.ceil(wcet/tick)+fpgascheduleroverhead_reexec+fpgascheduleroverhead_noreexec+fpgaschedulerjobendoverhead, wcetinswschedulerticks, dtype=float)
'''
from matplotlib.axis import Axis   
import matplotlib.ticker as tkr

def plot1(swschedulertick, fpgaschedulertick, swscheduleroverhead_reexec, swscheduleroverhead_noreexec, fpgascheduleroverhead_reexec, fpgascheduleroverhead_noreexec, jobendoverheadswscheduler, jobendoverheadfpgascheduler, swlegend, fpgalegend, swschedulercolour, fpgaschedulercolour):
    wcetinswschedulerticks=np.linspace(1.0, 20.0, num=4000)

    #fig= plt.figure(figsize=(10,6))
    fig, ax = plt.subplots() 
    for i in range(len(swscheduleroverhead_noreexec)):
        yswscheduler=computewcetsw(wcetinswschedulerticks, swschedulertick[i], swscheduleroverhead_reexec[i], swscheduleroverhead_noreexec[i], jobendoverheadswscheduler)
        #ax.plot(wcetinswschedulerticks, yswscheduler,":",color=swschedulercolour[i],linewidth=2)
        ax.plot(wcetinswschedulerticks, yswscheduler,color=swschedulercolour[i])
    
    swlegend=np.append(swlegend, fpgalegend)
    swlegend=np.append(swlegend, "prova")
    for i in range(len(fpgascheduleroverhead_noreexec)):
        yfpgascheduler=computewcetfpga(wcetinswschedulerticks, fpgaschedulertick[i], swschedulertick[i], fpgascheduleroverhead_reexec[i], fpgascheduleroverhead_noreexec[i], jobendoverheadfpgascheduler)
        ax.plot(wcetinswschedulerticks, yfpgascheduler,color=fpgaschedulercolour[i],linewidth=1)

    Axis.set_major_formatter(ax.yaxis, tkr.PercentFormatter())
    Axis.set_major_formatter(ax.xaxis, tkr.StrMethodFormatter('{x:,g}*$\sigma$'))
   # ax.set_yscale('log')
    plt.gca().yaxis.grid(True, which='minor')
    plt.gca().yaxis.grid(True, which='major')

    plt.tick_params(axis='y', which='minor')
    #ax.yaxis.set_minor_formatter(tkr.FormatStrFormatter("%.1f"))
    #plt.gca().yaxis.set_minor_locator(tkr.MultipleLocator(10))
    plt.gca().xaxis.set_major_locator(tkr.MultipleLocator(2))
    plt.gca().xaxis.set_minor_locator(tkr.AutoMinorLocator())


    plt.legend(swlegend)
    plt.xlabel(f"WCET $C^i_i$ [$\mu s$], $\sigma={int(swschedulertick[0])}$ [$\mu s$]")
    plt.ylabel("WCET increase [%]")
    plt.show()

'''
def plot2(swscheduleroverhead_reexec, swscheduleroverhead_noreexec, fpgascheduleroverhead_reexec, fpgascheduleroverhead_noreexec):
    #graph 2
    C_T_original_target=0.5

    wcet=np.linspace(1.0, 10.0, num=30)
    tick=np.linspace(1.0, 10.0, num=30).reshape(30,1)
    wcet_ticks=np.ceil(np.divide(wcet,tick))
    print(wcet)
    print(tick)
    print(wcet_ticks)
    T=np.divide(wcet_ticks, C_T_original_target)

    utilisation_sw=np.divide(computewcetsw(wcet_ticks, swscheduleroverhead_reexec, swscheduleroverhead_noreexec), T)
    utilisation_fpga=np.divide(np.ceil(np.divide(computewcetfpga(wcet_ticks, fpgascheduleroverhead_reexec, fpgascheduleroverhead_noreexec), fpgascheduleroverhead_reexec)), np.floor(T/fpgascheduleroverhead_reexec))
'''
#inputfile = np.genfromtxt(csv_fname, delimiter=';', names=True, case_sensitive=True)
file_name = "comparison_data.csv"

current_path = os.path.dirname(os.path.realpath(__file__))  # Get the current script's directory
file_path = os.path.join(current_path, file_name)
    
    
with open(file_path, 'r') as file:
    csv_reader = csv.reader(file, delimiter=";")
        
    # Read the data rows
    for row in csv_reader:
        header=row[0]
        row.pop(0)
        if (header=="swlegend"):
            swlegend=np.array(row, dtype=str)
        elif (header=="fpgalegend"):
            fpgalegend=np.array(row, dtype=str)
        elif (header=="swschedulercolour"):
            swschedulercolour=np.array(row, dtype=str)
        elif (header=="fpgaschedulercolour"):
            fpgaschedulercolour=np.array(row, dtype=str)
        elif (header=="fpgascheduleroverhead_noreexec"):
            fpgascheduleroverhead_noreexec=np.array(row, dtype=float)
        elif (header=="fpgascheduleroverhead_reexec"):
            fpgascheduleroverhead_reexec=np.array(row, dtype=float)
        elif (header=="swscheduleroverhead_noreexec"):
            swscheduleroverhead_noreexec=np.array(row, dtype=float)
        elif (header=="swscheduleroverhead_reexec"):
            swscheduleroverhead_reexec=np.array(row, dtype=float)
        elif (header=="swschedulerjobendoverhead"):
            jobendoverheadswscheduler=float(row[0])
        elif (header=="fpgaschedulerjobendoverhead"):
            jobendoverheadfpgascheduler=float(row[0])
        elif (header=="swschedulertick"):
            swschedulertick=np.array(row, dtype=float)
        elif (header=="fpgaschedulertick"):
            fpgaschedulertick=np.array(row, dtype=float)

plot1(swschedulertick, fpgaschedulertick, swscheduleroverhead_reexec, swscheduleroverhead_noreexec, fpgascheduleroverhead_reexec, fpgascheduleroverhead_noreexec, jobendoverheadswscheduler, jobendoverheadfpgascheduler, swlegend, fpgalegend, swschedulercolour, fpgaschedulercolour)
#plot2(swscheduleroverhead_reexec, swscheduleroverhead_noreexec, fpgascheduleroverhead_reexec, fpgascheduleroverhead_noreexec)
