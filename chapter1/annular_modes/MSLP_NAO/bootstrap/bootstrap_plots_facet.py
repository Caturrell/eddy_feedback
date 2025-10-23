import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sp
from pypalettes import load_cmap

# --- Paths and setup ---
data_folder = '/home/links/ct715/eddy_feedback/chapter1/annular_modes/MSLP_NAO/bootstrap/data'
plot_folder = '/home/links/ct715/eddy_feedback/chapter1/annular_modes/MSLP_NAO/bootstrap/plots'
sample_sizes = [37, 58, 100, 200, 300]

# --- Load original EFP data ---
efp_df = pd.read_csv('/home/links/ct715/eddy_feedback/chapter1/efp_random/data/daily_efp_8models+jra55.csv', index_col=0)
models = efp_df['model'].unique()

# Load colormap and define shared palette
cmap = load_cmap("Cross")
colors = [cmap(i / (len(models) - 1)) for i in range(len(models))]
palette = dict(zip(models, colors))


# Add JRA55 manually
palette['JRA55'] = 'black'

# --- Create combined figure ---
nplots = len(sample_sizes) + 1
ncols = 3
nrows = int(np.ceil(nplots / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), constrained_layout=True)
axes = axes.flatten()

# --- First plot: Original 8-model EFP plot ---
ax = axes[0]
sns.regplot(data=efp_df, x='efp_sh', y='efp_nh', scatter_kws={'color': 'white'}, line_kws={"color": "gray"}, ci=None, ax=ax)
sns.scatterplot(data=efp_df, x='efp_sh', y='efp_nh', hue='model', style='model', s=150, palette=palette, ax=ax)

corr_df_og = efp_df[efp_df['model'] != 'JRA55']
r, _ = sp.pearsonr(x=corr_df_og['efp_nh'], y=corr_df_og['efp_sh'])
ax.text(.05, .92, f"r = {r:.2f}", transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.7), fontsize=12)

ax.set_ylabel('NH EFP', fontsize=14)
ax.set_xlabel('SH EFP', fontsize=14)
ax.set_xlim(0.1, 0.4)
ax.set_ylim(0.1, 0.4)
ax.set_title('Original EFP (daily)', fontsize=14)
ax.legend_.remove()

# --- Bootstrap plots ---
for idx, size in enumerate(sample_sizes):
    ax = axes[idx + 1]
    sample_path = os.path.join(data_folder, f'sample_size_{size}')

    # Load DJF
    bootstrap_results_djf = {
        f.replace('_djf_efp_values.npy', ''): np.load(os.path.join(sample_path, f))
        for f in os.listdir(sample_path) if f.endswith('_djf_efp_values.npy')
    }
    bootstrap_df_djf = pd.DataFrame({
        'Model': np.repeat(list(bootstrap_results_djf), [len(v) for v in bootstrap_results_djf.values()]),
        'EFP': np.concatenate(list(bootstrap_results_djf.values()))
    })

    # Load JAS
    bootstrap_results_jas = {
        f.replace('_jas_efp_values.npy', ''): np.load(os.path.join(sample_path, f))
        for f in os.listdir(sample_path) if f.endswith('_jas_efp_values.npy')
    }
    bootstrap_df_jas = pd.DataFrame({
        'Model': np.repeat(list(bootstrap_results_jas), [len(v) for v in bootstrap_results_jas.values()]),
        'EFP': np.concatenate(list(bootstrap_results_jas.values()))
    })

    # Order models with JRA55 last
    model_order = sorted([m for m in bootstrap_results_djf if m != 'JRA55']) + ['JRA55']
    bootstrap_df_djf['Model'] = pd.Categorical(bootstrap_df_djf['Model'], categories=model_order, ordered=True)
    bootstrap_df_jas['Model'] = pd.Categorical(bootstrap_df_jas['Model'], categories=model_order, ordered=True)

    # Stats
    mean_efp_df = pd.DataFrame({
        'Model': model_order,
        'EFP_NH': bootstrap_df_djf.groupby('Model', observed=True)['EFP'].mean().values,
        'EFP_SH': bootstrap_df_jas.groupby('Model', observed=True)['EFP'].mean().values,
        'EFP_NH_STD': bootstrap_df_djf.groupby('Model', observed=True)['EFP'].std().values,
        'EFP_SH_STD': bootstrap_df_jas.groupby('Model', observed=True)['EFP'].std().values
    })

    # Exclude JRA55 from correlation
    corr_df = mean_efp_df[mean_efp_df['Model'] != 'JRA55']
    r, _ = sp.pearsonr(corr_df['EFP_NH'], corr_df['EFP_SH'])

    # Plot
    sns.regplot(x='EFP_SH', y='EFP_NH', data=mean_efp_df,
                scatter=False, color='gray', line_kws={'linewidth': 2}, ci=None, ax=ax)

    sns.scatterplot(x='EFP_SH', y='EFP_NH', hue='Model', style='Model',
                    data=mean_efp_df, palette=palette, s=150, ax=ax, legend=False)

    ax.errorbar(mean_efp_df['EFP_SH'], mean_efp_df['EFP_NH'],
                xerr=mean_efp_df['EFP_SH_STD'], yerr=mean_efp_df['EFP_NH_STD'],
                fmt='none', ecolor='gray', alpha=0.5, capsize=3)

    ax.set_xlim(0.1, 0.4)
    ax.set_ylim(0.1, 0.4)
    ax.set_xlabel('Mean EFP (SH)', fontsize=12)
    ax.set_ylabel('Mean EFP (NH)', fontsize=12)
    ax.set_title(f'Sample size = {size}', fontsize=14)
    ax.text(0.05, 0.92, f"r = {r:.2f}", transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7), fontsize=11)

# Hide unused axes
for i in range(nplots, len(axes)):
    fig.delaxes(axes[i])

# Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Model', bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=10)

# Save
output_path = os.path.join(plot_folder, 'efp_regression_facetgrid_combined.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()
