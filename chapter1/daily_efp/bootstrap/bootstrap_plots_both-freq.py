import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def load_bootstrap_results(folder, season):
    results = {}
    for file_name in os.listdir(folder):
        if file_name.endswith(f'_{season}_efp_values.npy') or file_name.endswith(f'_{season}_efp_mixed-freq.npy'):
            model = file_name.split('_')[0]
            if model == 'JRA55':
                period = file_name.split('_')[1]
                model = f'JRA55 ({period})'
            efp_values = np.load(os.path.join(folder, file_name))
            results[model] = efp_values
    return results

def create_bootstrap_df(results):
    df = pd.DataFrame({
        'Model': np.repeat(list(results.keys()), [len(v) for v in results.values()]),
        'EFP': np.concatenate(list(results.values()))
    })
    model_order = sorted([m for m in results if m.split(' ')[0] != 'JRA55']) + ['JRA55 (1979-2016)']
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
    return df

def compute_mean_std(df):
    mean = df.groupby('Model', observed=True)['EFP'].mean()
    std = df.groupby('Model', observed=True)['EFP'].std()
    return mean, std

def build_summary_df(mean_nh, std_nh, mean_sh, std_sh):
    return pd.DataFrame({
        'Model': mean_nh.index,
        'EFP_NH': mean_nh.values,
        'EFP_SH': mean_sh.values,
        'EFP_NH_STD': std_nh.values,
        'EFP_SH_STD': std_sh.values
    })

def plot_subplots(summary_dict, output_path):
    """summary_dict: {freq: summary_df}"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    
    for ax, (freq, df) in zip(axes, summary_dict.items()):
        corr_set = df[df['Model'] != 'JRA55 (1979-2016)']
        pearson_corr, p_value = pearsonr(corr_set['EFP_NH'], corr_set['EFP_SH'])
        
        sns.scatterplot(
            x='EFP_NH', y='EFP_SH',
            hue='Model', style='Model',
            data=df, s=120, ax=ax, legend=False
        )
        
        ax.errorbar(
            df['EFP_NH'], df['EFP_SH'],
            xerr=df['EFP_NH_STD'], yerr=df['EFP_SH_STD'],
            fmt='none', ecolor='gray', alpha=0.5, capsize=3
        )
        
        sns.regplot(
            x='EFP_NH', y='EFP_SH', data=df,
            scatter=False, color='black', line_kws={'linewidth':2},
            ci=None, ax=ax
        )
        
        ax.set_title(f"{freq.capitalize()} EFP")
        ax.set_xlabel('Mean EFP (NH)', fontsize=12)
        ax.set_ylabel('Mean EFP (SH)', fontsize=12)
        ax.text(
            0.02, 0.87,
            f'r = {pearson_corr:.2f}, p = {p_value:.4f}',
            transform=ax.transAxes,
            fontsize=11, bbox=dict(facecolor="white", alpha=0.7)
        )
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    data_folder = '/home/links/ct715/eddy_feedback/chapter1/daily_efp/bootstrap/data'
    plot_folder = '/home/links/ct715/eddy_feedback/chapter1/daily_efp/bootstrap/plots'
    frequencies = ['daily', 'mixed-freq']  # change as needed

    summary_dict = {}
    
    for freq in frequencies:
        print('Processing frequency:', freq)
        freq_path = os.path.join(data_folder, freq)
        bootstrap_djf = load_bootstrap_results(freq_path, 'djf')
        bootstrap_jas = load_bootstrap_results(freq_path, 'jas')
        
        df_djf = create_bootstrap_df(bootstrap_djf)
        df_jas = create_bootstrap_df(bootstrap_jas)
        
        mean_djf, std_djf = compute_mean_std(df_djf)
        mean_jas, std_jas = compute_mean_std(df_jas)
        
        summary_df = build_summary_df(mean_djf, std_djf, mean_jas, std_jas)
        summary_dict[freq] = summary_df

    output_path = os.path.join(plot_folder, 'efp_regression_subplots.png')
    plot_subplots(summary_dict, output_path)

if __name__ == "__main__":
    main()
