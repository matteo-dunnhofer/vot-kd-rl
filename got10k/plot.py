import sys

sys.path.insert(0, '..')

from got10k.experiments import ExperimentGOT10k

report_files = ['/media/TBData2/data/vot/reports/GOT-10k/ablation_a3ct_a3ctd.json']
tracker_names = ['A3CTD', 'A3CT', 'A3CT-SL', 'A3CT-no-curr', 'A3CTD-no-curr']
#report_files = ['/media/TBData2/data/vot/reports/GOT-10k/performance_25_entries.json']
#tracker_names = ['SiamFCv2', 'GOTURN', 'CCOT', 'MDNet', 'EC0', 'KCF']

# setup experiment and plot curves
experiment = ExperimentGOT10k('/media/TBData2/data/vot/GOT-10k', subset='test')
experiment.plot_curves(report_files, tracker_names)