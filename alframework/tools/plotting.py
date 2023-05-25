import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
import os

def analysis_plot(meta_dir, master_directory, meta_n_cols=50):
    if not hasattr(analysis_plot, '__plot_number'):
        analysis_plot.__plot_number = 0
    num_models = 0
    metadata_pattern = re.compile(r'metadata-mol-(.{4})-\d{10}.p')

    metadata_record = {'model': [],
                       'realtime': [],
                       'log realtime': [],
                       'simtime': [],
                       'log simtime': [],
                       'Es': [],
                       'Fs': [],
                       'Fsmax': []}

    for pickle_filename in os.listdir(meta_dir):
        match = metadata_pattern.fullmatch(pickle_filename)
        if match.group(1) != 'boot':
            model_number = int(match.group(1))
            if model_number > num_models:
                num_models = model_number
            with open(meta_dir + '/' + pickle_filename, 'rb') as fp:
                model = pickle.load(fp)
                metadata_record['model'].append(model_number)
                metadata_record['realtime'].append(model['realtime_simulation'])
                metadata_record['log realtime'].append(np.log10(model['realtime_simulation']))
                metadata_record['simtime'].append(model['simulationtime'])
                metadata_record['log simtime'].append(np.log10(model['simulationtime']))
                metadata_record['Es'].append(model['Es'])
                metadata_record['Fs'].append(model['Fs'])
                metadata_record['Fsmax'].append(model['Fsmax'])

    # Don't do anything if there are no models
    if num_models > 0:
        df = pd.DataFrame(metadata_record)

        if num_models > meta_n_cols:
            model_filter = np.linspace(1, num_models, num=meta_n_cols, dtype=int)
            df = df[df['model'].isin(model_filter)]

        plot_vars = ['realtime', 'log realtime', 'simtime', 'log simtime' 'Es', 'Fs', 'Fsmax']
        plot_titles = ['Real Time Simulation', 'Log Real Time Simulation', 'Simulation Time', 'Log Simulation Time', 'Es', 'Fs', 'Fsmax Error']
        for var, title in zip(plot_vars, plot_titles):
            plot = df.boxplot(var, by='model', grid=False, rot=45)
            plot.set_title(title)
            plot.set_ylabel(var)
            plt.savefig(master_directory + '/analysis-plot-{:04d}-{:s}.png'.format(analysis_plot.__plot_number, var))

        analysis_plot.__plot_number += 1
