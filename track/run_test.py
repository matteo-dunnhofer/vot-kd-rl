"""
Written by Matteo Dunnhofer - 2020

Script to run experiements with the GOT-10k framework
"""
import sys
sys.path.insert(0, '..')
import argparse
from got10k.experiments import *
from Trackers import Tracker_got10k


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Name of the dataset to test on', type=str)
    parser.add_argument('--tracker', help='Tracker to run, either TRAS, TRAST or TRASFUST')
    parser.add_argument('--visualize', help='Visualize predictions while testing', action="store_true")
    args = parser.parse_args()

    if 'TRAS' in args.tracker:
        from config_track_accv import Configuration
    else:
        from config_track_iccvw import Configuration

    cfg = Configuration()

    base_data_path = cfg.DATA_PATH

    # setup tracker
    tracker = Tracker_got10k(args.tracker, cfg)

    # setup experiments
    if args.dataset == 'GOT10k':
        e = ExperimentGOT10k(base_data_path + 'GOT-10k',
            result_dir=cfg.RESULTS_PATH,
            report_dir=cfg.REPORT_PATH,
            subset='test')
    elif 'OTB' in args.dataset:
        version = int(''.join(list(filter(str.isdigit, args.dataset))))
        e = ExperimentOTB(base_data_path + 'OTB', version=version,
            result_dir=cfg.RESULTS_PATH,
            report_dir=cfg.REPORT_PATH)
    elif 'VOT' in args.dataset:
        version = int(''.join(list(filter(str.isdigit, args.dataset))))
        e = ExperimentVOT(base_data_path + 'VOT/'+str(version), version=version,
            result_dir=cfg.RESULTS_PATH,
            report_dir=cfg.REPORT_PATH)
    elif args.dataset == 'UAV123':
        e = ExperimentUAV123(base_data_path + 'UAV123', version='UAV123',
            result_dir=cfg.RESULTS_PATH,
            report_dir=cfg.REPORT_PATH)
    elif args.dataset == 'LaSOT':
        e = ExperimentLaSOT(base_data_path + 'LaSOTBenchmark',
            result_dir=cfg.RESULTS_PATH,
            report_dir=cfg.REPORT_PATH)


    # run tracking experiments and report performance
    e.run(tracker, visualize=args.visualize)
    e.report([tracker.name])
