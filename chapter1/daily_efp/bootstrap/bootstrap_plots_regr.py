import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def load_bootstrap_results(folder, season):
    """Load bootstrap EFP values for a given season (djf or jas)."""
    results = {}
    for file_name in os.listdir(folder):
        if file_name.endswith(f'_{season}_efp_values.npy'):
            model = file_name.split('_')[0]
            if model == 'JRA55':
                period = file_name.split('_')[1]
                model = f'JRA55 ({period})'
            efp_values = np.load(os.path.join(folder, file_name))
            results[model] = efp_values
    return results


def create_bootstrap_df(results):
    """Convert bootstrap results dictionary into a DataFrame with model order set."""
    df = pd.DataFrame({
        'Model': np.repeat(list(results.keys()), [len(v) for v in results.values()]),
        'EFP': np.concatenate(list(results.values()))
    })
    model_order = sorted([m for m in results if m.split(' ')[0] != 'JRA55']) + ['JRA55 (1979-2016)'] # omit 1958: 'JRA55 (1958-2016)', 
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
    return df


def compute_mean_std(df):
    """Compute mean and standard deviation of EFP by model."""
    mean = df.groupby('Model', observed=True)['EFP'].mean()
    std = df.groupby('Model', observed=True)['EFP'].std()
    return mean, std


def build_summary_df(mean_nh, std_nh, mean_sh, std_sh):
    """Combine NH and SH stats into a summary DataFrame."""
    return pd.DataFrame({
        'Model': mean_nh.index,
        'EFP_NH': mean_nh.values,
        'EFP_SH': mean_sh.values,
        'EFP_NH_STD': std_nh.values,
        'EFP_SH_STD': std_sh.values
    })


def plot_regression(df, pearson_corr, p_value, output_path):
    """Plot regression scatter with error bars and save."""

    from pypalettes import load_cmap

    # colours
    cmap = load_cmap("Cross")
    models = df['Model'].unique()
    colors = [cmap(i / (len(models) - 1)) for i in range(len(models))]
    colors.append("black")  # add black at the end
    models_extended = list(models) + ["extra"]  # optional placeholder
    palette = dict(zip(models_extended, colors))

    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        x='EFP_NH', y='EFP_SH',
        hue='Model', style='Model',
        data=df, s=150, palette=palette
    )

    plt.errorbar(
        df['EFP_NH'], df['EFP_SH'],
        xerr=df['EFP_NH_STD'], yerr=df['EFP_SH_STD'],
        fmt='none', ecolor='gray', alpha=0.5, capsize=3
    )

    sns.regplot(
        x='EFP_NH', y='EFP_SH', data=df,
        scatter=False, color='black',
        line_kws={'linewidth': 2}, ci=None
    )

    plt.xlabel('Mean EFP (NH)', fontsize=14)
    plt.ylabel('Mean EFP (SH)', fontsize=14)
    plt.legend(
        title='Model', bbox_to_anchor=(1.05, 1),
        loc='upper left', fontsize=9
    )
    plt.text(
        0.02, 0.47,
        f'r = {pearson_corr:.2f} p = {p_value:.3f}',
        fontsize=12, verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor="white", alpha=0.7)
    )
    plt.xlim(0, 0.5)
    plt.ylim(0, 0.5)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()



def main(freq='daily'):
    data_folder = '/home/links/ct715/eddy_feedback/chapter1/daily_efp/bootstrap/data'
    plot_folder = '/home/links/ct715/eddy_feedback/chapter1/daily_efp/bootstrap/plots'
    daily_path = os.path.join(data_folder, freq)

    # Load bootstrap results
    bootstrap_djf = load_bootstrap_results(daily_path, 'djf')
    bootstrap_jas = load_bootstrap_results(daily_path, 'jas')

    # Create DataFrames
    df_djf = create_bootstrap_df(bootstrap_djf)
    df_jas = create_bootstrap_df(bootstrap_jas)

    # Compute means and stds
    mean_djf, std_djf = compute_mean_std(df_djf)
    mean_jas, std_jas = compute_mean_std(df_jas)

    # Build summary DataFrame
    summary_df = build_summary_df(mean_djf, std_djf, mean_jas, std_jas)

    # Pearson correlation
    corr_set = summary_df[summary_df['Model'] != 'JRA55 (1979-2016)']
    pearson_corr, p_value = pearsonr(corr_set['EFP_NH'], corr_set['EFP_SH'])

    # Plot
    output_path = os.path.join(plot_folder, 'pamip_daily-efp_regression_plot.png')
    plot_regression(summary_df, pearson_corr, p_value, output_path)


if __name__ == "__main__":
    main()
