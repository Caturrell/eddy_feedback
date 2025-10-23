import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Define the path to the data folder
data_folder = '/home/links/ct715/eddy_feedback/chapter1/annular_modes/MSLP_NAO/bootstrap/data'
plot_folder = '/home/links/ct715/eddy_feedback/chapter1/annular_modes/MSLP_NAO/bootstrap/plots'
sample_sizes = [37, 58, 100, 200, 300]

for size in sample_sizes:
    sample_path = os.path.join(data_folder, f'sample_size_{size}')

    # Load DJF EFP values
    bootstrap_results_djf = {}
    for file_name in os.listdir(sample_path):
        if file_name.endswith('_djf_efp_values.npy'):
            model = file_name.replace('_djf_efp_values.npy', '')
            efp_values = np.load(os.path.join(sample_path, file_name))
            bootstrap_results_djf[model] = efp_values

    # Create DataFrame for DJF
    bootstrap_df_djf = pd.DataFrame({
        'Model': np.repeat(list(bootstrap_results_djf.keys()), [len(v) for v in bootstrap_results_djf.values()]),
        'EFP': np.concatenate(list(bootstrap_results_djf.values()))
    })

    # Alphabetical model order with JRA55 last
    model_order = sorted([m for m in bootstrap_results_djf if m != 'JRA55']) + ['JRA55']
    bootstrap_df_djf['Model'] = pd.Categorical(bootstrap_df_djf['Model'], categories=model_order, ordered=True)

    # Load JAS EFP values
    bootstrap_results_jas = {}
    for file_name in os.listdir(sample_path):  # Fixed: previously `sample_size` (int) was used
        if file_name.endswith('_jas_efp_values.npy'):
            model = file_name.replace('_jas_efp_values.npy', '')
            efp_values = np.load(os.path.join(sample_path, file_name))
            bootstrap_results_jas[model] = efp_values

    # Create DataFrame for JAS
    bootstrap_df_jas = pd.DataFrame({
        'Model': np.repeat(list(bootstrap_results_jas.keys()), [len(v) for v in bootstrap_results_jas.values()]),
        'EFP': np.concatenate(list(bootstrap_results_jas.values()))
    })
    model_order = sorted([m for m in bootstrap_results_jas if m != 'JRA55']) + ['JRA55']
    bootstrap_df_jas['Model'] = pd.Categorical(bootstrap_df_jas['Model'], categories=model_order, ordered=True)

    # Compute stats per model
    mean_efp_nh = bootstrap_df_djf.groupby('Model', observed=True)['EFP'].mean()
    std_efp_nh = bootstrap_df_djf.groupby('Model', observed=True)['EFP'].std()
    mean_efp_sh = bootstrap_df_jas.groupby('Model', observed=True)['EFP'].mean()
    std_efp_sh = bootstrap_df_jas.groupby('Model', observed=True)['EFP'].std()

    mean_efp_df = pd.DataFrame({
        'Model': mean_efp_nh.index,
        'EFP_NH': mean_efp_nh.values,
        'EFP_SH': mean_efp_sh.values,
        'EFP_NH_STD': std_efp_nh.values,
        'EFP_SH_STD': std_efp_sh.values
    })

    # Compute Pearson correlation
    pearson_corr, _ = pearsonr(mean_efp_df['EFP_NH'], mean_efp_df['EFP_SH'])

    # Plotting
    plt.figure(figsize=(9, 5))
    sns.scatterplot(x='EFP_NH', y='EFP_SH', hue='Model', style='Model',
                    data=mean_efp_df, s=150)

    plt.errorbar(
        mean_efp_df['EFP_NH'], mean_efp_df['EFP_SH'],
        xerr=mean_efp_df['EFP_NH_STD'], yerr=mean_efp_df['EFP_SH_STD'],
        fmt='none', ecolor='gray', alpha=0.5, capsize=3
    )

    sns.regplot(
        x='EFP_NH', y='EFP_SH', data=mean_efp_df,
        scatter=False, color='black', line_kws={'linewidth': 2}, ci=None
    )

    plt.xlabel('Mean EFP (NH)', fontsize=14)
    plt.ylabel('Mean EFP (SH)', fontsize=14)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.text(
        mean_efp_df['EFP_NH'].min(), mean_efp_df['EFP_SH'].max(),
        f'r = {pearson_corr:.2f}', fontsize=12, verticalalignment='top',
        horizontalalignment='left', bbox=dict(facecolor="white", alpha=0.7)
    )
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(plot_folder, f'pamip_JRA55_efp_regression_plot_{size}.pdf')
    plt.savefig(output_path, bbox_inches='tight')
