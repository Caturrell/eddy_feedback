import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

import functions.data_wrangling as dw
import functions.eddy_feedback as ef


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Logging is set up.')


def bootstrap_resampling_pamip(ds, sample_size=None, num_samples=1000):
    logging.info('Starting bootstrap resampling...')
    if 'ens_ax' in ds.dims:
        n = ds.sizes['ens_ax']
        sample_size = sample_size or n
        logging.info(f'Using dimension "ens_ax" with size {n}, sample size: {sample_size}')
        bootstrap_indices = np.random.choice(n, size=(num_samples, sample_size), replace=True)
        return [ds.isel(ens_ax=idx) for idx in bootstrap_indices], sample_size
    else:
        n = ds.sizes['time']
        sample_size = sample_size or n
        logging.info(f'Using dimension "time" with size {n}, sample size: {sample_size}')
        bootstrap_indices = np.random.choice(n, size=(num_samples, sample_size), replace=True)
        return [ds.isel(time=idx) for idx in bootstrap_indices], sample_size


def load_datasets(path_dir):
    logging.info(f'Loading datasets from directory: {path_dir}')
    files = os.listdir(path_dir)
    models = sorted(set(os.path.basename(f).split('_')[0] for f in files))
    logging.info(f'Found models: {models}')

    djf_pamip = {}
    jas_pamip = {}

    for model in models:
        logging.info(f'Loading data for model: {model}')
        file_path = os.path.join(path_dir, f'{model}_*.nc')
        ds = xr.open_mfdataset(file_path, parallel=True, chunks={'time': 31})
        logging.info(f'Computing seasonal means for {model}')
        djf_pamip[model] = dw.seasonal_mean(ds, season='djf')
        jas_pamip[model] = dw.seasonal_mean(ds, season='jas')

    logging.info('Finished loading and preprocessing all models.')
    return djf_pamip, jas_pamip


def process_model_bootstrap(model, ds, season, sample_size, output_base, is_southern=False):
    results = []

    logging.info(f'Starting bootstrap resampling for model: {model} ({season.upper()}) with sample_size={sample_size}')
    bootstrap_sets, sample_size = bootstrap_resampling_pamip(ds, sample_size=sample_size)

    for i, item in enumerate(bootstrap_sets):
        if i % 100 == 0:
            logging.info(f'Processing bootstrap sample {i+1}/{len(bootstrap_sets)} for model: {model} ({season.upper()})')

        efp = ef.calculate_efp(item, data_type='pamip', calc_south_hemis=is_southern, bootstrapping=True)
        results.append(efp)

    save_path = os.path.join(output_base, f'{model}_{season}_efp_mixed-freq.npy')
    np.save(save_path, results)
    logging.info(f'Saved EFP values for model: {model} ({season.upper()}) at {save_path}')

    return results


def create_violin_plot(data_df, season, save_dir):
    logging.info(f'Creating violin plot for season: {season.upper()}')
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Model', y='EFP', data=data_df)
    plt.xticks(rotation=90)
    plt.title(f'Bootstrap EFP Values for Each Model ({season.upper()})')
    plt.xlabel('Model')
    plt.ylabel('EFP')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plot_file = os.path.join(save_dir, f'{season}_mixed-freq_bootstrap_efp.png')
    plt.savefig(plot_file)
    logging.info(f'Saved {season.upper()} violin plot at {plot_file}')


def main():
    setup_logging()
    logging.info('Starting main analysis with ensemble-based sample sizes')

    path_dir = '/home/links/ct715/data_storage/PAMIP/processed_monthly/efp_pd_non-regridded'
    output_base = '/home/links/ct715/eddy_feedback/chapter1/daily_efp/bootstrap/data/mixed-freq'
    plot_base = '/home/links/ct715/eddy_feedback/chapter1/daily_efp/bootstrap/plots'

    djf_pamip, jas_pamip = load_datasets(path_dir)

    logging.info('Beginning bootstrap analysis for DJF (Northern Hemisphere)...')
    bootstrap_results_djf = {
        model: process_model_bootstrap(model, ds, season='djf', sample_size=None, output_base=output_base)
        for model, ds in djf_pamip.items()
    }
    bootstrap_df_djf = pd.DataFrame({
        'Model': np.repeat(list(bootstrap_results_djf.keys()), [len(v) for v in bootstrap_results_djf.values()]),
        'EFP': np.concatenate(list(bootstrap_results_djf.values()))
    })
    create_violin_plot(bootstrap_df_djf, 'djf', plot_base)

    logging.info('Beginning bootstrap analysis for JAS (Southern Hemisphere)...')
    bootstrap_results_jas = {
        model: process_model_bootstrap(model, ds, season='jas', sample_size=None, output_base=output_base, is_southern=True)
        for model, ds in jas_pamip.items()
    }
    bootstrap_df_jas = pd.DataFrame({
        'Model': np.repeat(list(bootstrap_results_jas.keys()), [len(v) for v in bootstrap_results_jas.values()]),
        'EFP': np.concatenate(list(bootstrap_results_jas.values()))
    })
    create_violin_plot(bootstrap_df_jas, 'jas', plot_base)

    logging.info('Completed main analysis (ensemble-based sample sizes)')



if __name__ == '__main__':
    setup_logging()
    logging.info('===== Starting bootstrap pipeline with ensemble sizes =====')
    main()
    logging.info('===== All analyses complete =====')

