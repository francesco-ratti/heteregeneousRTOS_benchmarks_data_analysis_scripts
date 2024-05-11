# HeteregeneousRTOS benchmarks data analysis scripts

Scripts to process the output of [HeterogeneousROTS Benchmarks](https://github.com/francesco-ratti/heterogeneousRTOS_benchmarks).

<strong>process_timing.py</strong> to process and convert in ns the measures acquired with the GTC.\
<strong>timing_comparison.py</strong> as above, but comparing two files.\
<strong>timing_remove_offset.py</strong> to remove a measurement offset from a dataset.

<strong>FaultDet_performances</strong> contains the scripts used to generate the JSON files from the binaries output of "faultdet_performance_measurement_standalone" in [Benchmarks source code and experimental evaluation artifacts](https://github.com/francesco-ratti/heterogeneousRTOS_benchmarks), process them and generate the charts used in the paper.\
<strong>SW_vs_FPGA_scheduler_timing_comparison</strong> contains the scripts used to generate the charts for comparing the software scheduler to the FPGA scheduler, based on the measurements acquired (and contained, post-processed, in "measurement.ods" and in the paper).

<h3>Related repositories:</h3>

[HeterogeneousRTOS Source Code](https://github.com/francesco-ratti/heterogeneousRTOS)\
[HeterogeneousRTOS Vivado Platform](https://github.com/francesco-ratti/heterogeneousRTOS_HW)\
[Fault Detector Vitis HLS Project](https://github.com/francesco-ratti/heterogeneousRTOS_faultDetector_HLS)\
[Benchmarks source code and experimental evaluation artifacts](https://github.com/francesco-ratti/heterogeneousRTOS_benchmarks)\
[Data processing and analysis scripts](https://github.com/francesco-ratti/heteregeneousRTOS_benchmarks_data_analysis_scripts)
