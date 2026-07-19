"""
Compares the original method (jra55_850) against the Simpson-style sliding-segment
lag methodology + linear detrending (jra_simpson_lag) - see
/home/links/ct715/.claude/plans/wobbly-prancing-giraffe.md - for the
1979-2014, 6hourly, level_full_100_850 configuration.

Scope: southern hemisphere, all_time / DJF / JJA (separate figures for each time
frame - not combined). Native and va (vertically-averaged) variants are overlaid
on the same panels (va = solid, native = dashed), and the two methods are
overlaid too (color = method) - so each panel has up to 4 lines: 2 methods x 2
variants. fig2 (cospectrum/coherence/phase-diff) drops the imaginary cospectrum
part and the arctan tau-fit line that the single-dataset fig2 shows, since
color+linestyle are already spent on method+variant here - the fit timescale is
folded into the legend label as text instead.

For all_time, lag_method has no effect on the underlying data at all, so any
difference seen there is purely from detrending. For DJF/JJA, the two datasets
are NOT run through the same code path here: jra55_850 predates the "_continuous"
variables entirely (it only has the tightly-packed, season-only PC series), so
its autocorrelation/cross-correlation is computed the "naive" way (statsmodels
ccf on the raw tightly-packed series, matching what that dataset actually
contains). jra_simpson_lag has "_continuous" variables, so its autocorrelation/
cross-correlation is computed the sliding-segment way (index stays season-
restricted, displaced side drawn from the continuous projection), matching what
b_fit_simpson_2013/eof_plots' lag_method='simpson_sliding' path actually
computes. This means each line shows what that dataset's method genuinely
produces, rather than an artificial like-for-like recomputation - so DJF/JJA
differences reflect both the detrending change AND the season-boundary fix
combined, not just one or the other.

Mirrors, in structure, the single-dataset versions of these figures:
    b-parameter/b_methodology/fig2_cospec_coher_pdiff.py
    b-parameter/b_methodology/fig3_pw-spectra_autocorr.py
    b-parameter/b_methodology/fig4_cross-correlation.py
"""

import os
import numpy as np
import xarray as xar
import statsmodels.tsa.stattools as sm
import matplotlib.pyplot as plt

import functions.SIT_functions.SIT_eddy_feedback_functions as eff

script_dir = os.path.dirname(os.path.abspath(__file__))
all_plots_true_dir = os.path.dirname(script_dir)

hemisphere = 's'
lag_len = 40  # matches lag_len used to generate the original plots

va_str_dict = {'native': '', 'va': '_va'}
variant_linestyle = {'va': '-', 'native': '--'}
time_frames = ['all_time', 'DJF', 'JJA']
season_month_dict = {'DJF': [12, 1, 2], 'JJA': [6, 7, 8]}

methods = {
    'original':        {'exp_name': 'jra55_850',      'label': 'Original method',               'color': '#1f77b4'},
    'simpson_sliding': {'exp_name': 'jra_simpson_lag', 'label': 'Simpson sliding-lag + detrend',  'color': '#d62728'},
}

for method_info in methods.values():
    level_dir = os.path.join(
        all_plots_true_dir, f"{method_info['exp_name']}_sit_plots", '1979_2014',
        '6hourly', 'level_full_100_850',
    )
    eof_file = os.path.join(
        all_plots_true_dir, f"{method_info['exp_name']}_sit_plots", '1979_2014',
        '6hourly', 'EOF_full_100_850_prop_nans.nc',
    )
    method_info['power_spec_ds'] = xar.open_dataset(os.path.join(level_dir, 'power_spec.nc'), auto_complex=True)
    method_info['eof_ds'] = xar.open_dataset(eof_file)


def pc_names(va_str, time_frame):
    """Variable-name conventions shared by fig2/fig3/fig4 for a given va_str ('' or '_va')
    and time_frame ('all_time', 'DJF', 'JJA', ...)."""
    return {
        'ucomp_self':   f'ucomp{va_str}_PCs_{hemisphere}_{time_frame}',
        'div1_pseudo':  f'div1_QG{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}',
        'ucomp_pseudo': f'ucomp{va_str}_PCs_from_ucomp{va_str}_{hemisphere}_{time_frame}',
    }


def get_index_and_displaced(eof_ds, index_var_name, continuous_var_name, time_frame):
    """(index_values, displaced_values, used_sliding). index_var_name supplies the
    season-restricted (or all_time raw) index; continuous_var_name is looked up for
    its "_continuous" counterpart to supply the displaced side, mirroring
    b_fit_simpson_2013's ucomp_name (index) vs ucomp_continuous_name (displaced)
    split - the continuous variant only exists under the "..._PCs_from_ucomp..."
    projection naming, even for ucomp projected onto its own solver."""
    da = eof_ds[index_var_name].sel(eof_num=0)

    if time_frame == 'all_time':
        vals = da.values
        return vals, vals, False

    continuous_full_name = f'{continuous_var_name}_continuous'
    if continuous_full_name in eof_ds.variables:
        ntime = eof_ds.coords['time'].shape[0]
        where_hem = np.where(eof_ds['time'].dt.month.isin(season_month_dict[time_frame]))
        index_vals = np.zeros(ntime) + np.nan
        index_vals[where_hem[0]] = da.values
        displaced_vals = eof_ds[continuous_full_name].sel(eof_num=0).values
        return index_vals, displaced_vals, True

    vals = da.values  # tightly-packed, season-only (no continuous data for this dataset)
    return vals, vals, False


def _split_pos_neg(corr, lags):
    corr = np.asarray(corr)
    zero_idx = lag_len
    pos_lags, pos_vals = lags[zero_idx:], corr[zero_idx:]
    neg_lags, neg_vals = lags[:zero_idx + 1][::-1], corr[:zero_idx + 1][::-1]
    return pos_lags, pos_vals, neg_lags, neg_vals


def autocorr_pos_neg(eof_ds, var_name, time_frame):
    idx, disp, sliding = get_index_and_displaced(eof_ds, var_name, var_name, time_frame)
    if sliding:
        corr, lags = eff.cross_correlation(idx, disp, lag_len)
        return _split_pos_neg(corr, lags)
    acf_vals = sm.ccf(idx, idx, nlags=lag_len)
    pos_lags = np.arange(lag_len)
    neg_lags = np.arange(0, -lag_len, -1)
    return pos_lags, acf_vals, neg_lags, acf_vals


def cross_corr_pos_neg(eof_ds, ucomp_self_name, ucomp_pseudo_name, div1_pseudo_name, time_frame):
    ucomp_idx, ucomp_disp, ucomp_sliding = get_index_and_displaced(eof_ds, ucomp_self_name, ucomp_pseudo_name, time_frame)
    div1_idx, div1_disp, div1_sliding = get_index_and_displaced(eof_ds, div1_pseudo_name, div1_pseudo_name, time_frame)

    if ucomp_sliding and div1_sliding:
        # index (ucomp) stays season-restricted; displaced (div1) is continuous - one
        # call covers both lag directions (positive = ucomp leads div1, per
        # cross_correlation's sign convention).
        corr, lags = eff.cross_correlation(ucomp_idx, div1_disp, lag_len)
        return _split_pos_neg(corr, lags)

    x_corr, y_corr = ucomp_idx, div1_idx  # tightly-packed season-only, or all_time raw
    pos_vals = sm.ccf(y_corr, x_corr, nlags=lag_len)
    neg_vals = sm.ccf(x_corr, y_corr, nlags=lag_len)
    pos_lags = np.arange(lag_len)
    neg_lags = np.arange(0, -lag_len, -1)
    return pos_lags, pos_vals, neg_lags, neg_vals


def data_and_plot_dirs(time_frame):
    """.../method_comparison/{data,plots}/{time_frame}/ - created on demand."""
    data_dir = os.path.join(script_dir, 'data', time_frame)
    plot_dir = os.path.join(script_dir, 'plots', time_frame)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return data_dir, plot_dir


# ── Figure 1: cospectrum / coherence-squared / phase-difference ──────────────
#
# Simplified relative to the single-dataset fig2_cospec_coher_pdiff.py: color +
# linestyle are already spent on method + va/native here, so there's no third
# channel free for the imaginary part of the cospectrum or the arctan tau fit
# line fig2 shows. Real part only is plotted (the primary/energy-relevant part);
# the fit timescale is folded into the legend label as text (tau=X.Xd) instead
# of an extra line, so that information isn't lost, just not drawn as a curve.

def make_cospec_coher_pdiff_figure(time_frame):
    time_name = 'time' if time_frame == 'all_time' else f'time_{time_frame}'
    frequency_name = f'frequency_{time_name}'

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    save_data = {}
    ref_freq = None

    for method_key, method_info in methods.items():
        ds = method_info['power_spec_ds']
        color, label = method_info['color'], method_info['label']

        for va_label, va_str in va_str_dict.items():
            names = pc_names(va_str, time_frame)
            ucomp_name, div1_name = names['ucomp_self'], names['div1_pseudo']
            linestyle = variant_linestyle[va_label]

            freq = ds[frequency_name]
            cospec = ds[f'{ucomp_name}_{div1_name}_cospec_stft']
            coher = ds[f'{ucomp_name}_{div1_name}_coher_stft']
            phase_diff = ds[f'{div1_name}_{ucomp_name}_phase_diff']
            tau_fit_3 = float(ds[f'{div1_name}_{ucomp_name}_phase_diff_tau_fit_3'])

            line_label = f'{label} ({va_label}, τ={tau_fit_3:4.1f}d)'

            axes[0].plot(freq, np.real(cospec), color=color, linestyle=linestyle, label=line_label)
            axes[1].plot(freq, coher ** 2., color=color, linestyle=linestyle, label=line_label)
            axes[2].plot(freq, phase_diff, color=color, linestyle=linestyle, label=line_label)

            key = f'{method_key}_{va_label}'
            save_data[f'{key}_freq'] = freq.values
            save_data[f'{key}_cospec_real'] = np.real(cospec.values)
            save_data[f'{key}_coher'] = coher.values
            save_data[f'{key}_phase_diff'] = phase_diff.values
            save_data[f'{key}_tau_fit_3'] = np.array(tau_fit_3)

            if ref_freq is None:
                ref_freq = freq.values

    axes[0].axhline(0, color='k', linewidth=0.5)
    axes[0].plot(ref_freq, 2. * np.pi * ref_freq, color='grey', linestyle=':', label=r'$2\pi f$')
    axes[0].set_xlim(0., 0.25)
    axes[0].legend(fontsize=7)
    axes[0].grid(True)
    axes[0].set_title(r'Cospectrum (real part) of $[\overline{u}]_s$ and $[\overline{m}]_s$')
    axes[0].set_xlabel('frequency (1/days)')

    axes[1].set_xlim(0., 0.25)
    axes[1].set_ylim(0., 1.)
    axes[1].set_yticks(np.arange(0., 1. + 0.2, 0.2))
    axes[1].legend(fontsize=7)
    axes[1].grid(True)
    axes[1].set_title(r'Coherence squared of $[\overline{u}]_s$ and $[\overline{m}]_s$')
    axes[1].set_xlabel('frequency (1/days)')

    axes[2].set_xlim(0., 0.25)
    axes[2].set_ylim(0., 90.)
    axes[2].set_yticks(np.arange(0., 90. + 22.5, 22.5))
    axes[2].legend(fontsize=7)
    axes[2].grid(True)
    axes[2].set_title(r'Phase difference of $[\overline{u}]_s$ and $[\overline{m}]_s$ (observed)')
    axes[2].set_xlabel('frequency (1/days)')

    for ax, label in zip(axes, ['(a)', '(b)', '(c)']):
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=13,
                 fontweight='bold', va='top', ha='left')

    fig.suptitle(f'jra55_850 vs jra_simpson_lag - {hemisphere}H, {time_frame} (solid=va, dashed=native)')
    plt.tight_layout()

    data_dir, plot_dir = data_and_plot_dirs(time_frame)

    npz_file = os.path.join(data_dir, 'fig2_cospec_coher_pdiff.npz')
    np.savez(npz_file, **save_data)
    print(f'Saved data to {npz_file}')

    out_file = os.path.join(plot_dir, 'fig2_cospec_coher_pdiff.png')
    plt.savefig(out_file)
    plt.close(fig)
    print(f'Saved figure to {out_file}')


# ── Figure 2: power spectra + lagged autocorrelation (2x2) ───────────────────

def make_pw_spectra_autocorr_figure(time_frame):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex='col')
    save_data = {}
    time_name = 'time' if time_frame == 'all_time' else f'time_{time_frame}'
    frequency_name = f'frequency_{time_name}'

    for method_key, method_info in methods.items():
        ps_ds, eof_ds = method_info['power_spec_ds'], method_info['eof_ds']
        color, label = method_info['color'], method_info['label']

        for va_label, va_str in va_str_dict.items():
            names = pc_names(va_str, time_frame)
            ucomp_ps_name, div1_ps_name = names['ucomp_self'], names['div1_pseudo']
            ucomp_autocorr_name, div1_autocorr_name = names['ucomp_pseudo'], names['div1_pseudo']
            linestyle = variant_linestyle[va_label]
            line_label = f'{label} ({va_label})'

            freq = ps_ds[frequency_name]
            ucomp_power_spec = 2. * ps_ds[f'{ucomp_ps_name}_power_spec_stft'].values
            div1_power_spec = 2. * ps_ds[f'{div1_ps_name}_power_spec_stft'].values

            pos_lags, ucomp_acf_pos, neg_lags, ucomp_acf_neg = autocorr_pos_neg(eof_ds, ucomp_autocorr_name, time_frame)
            _, div1_acf_pos, _, div1_acf_neg = autocorr_pos_neg(eof_ds, div1_autocorr_name, time_frame)

            axes[0, 0].plot(freq, ucomp_power_spec, color=color, linestyle=linestyle, label=line_label)
            axes[0, 1].plot(pos_lags, ucomp_acf_pos, color=color, linestyle=linestyle, label=line_label)
            axes[0, 1].plot(neg_lags, ucomp_acf_neg, color=color, linestyle=linestyle)
            axes[1, 0].plot(freq, div1_power_spec, color=color, linestyle=linestyle, label=line_label)
            axes[1, 1].plot(pos_lags, div1_acf_pos, color=color, linestyle=linestyle, label=line_label)
            axes[1, 1].plot(neg_lags, div1_acf_neg, color=color, linestyle=linestyle)

            key = f'{method_key}_{va_label}'
            save_data[f'{key}_freq'] = freq.values
            save_data[f'{key}_ucomp_power_spec'] = ucomp_power_spec
            save_data[f'{key}_div1_power_spec'] = div1_power_spec
            save_data[f'{key}_pos_lags'] = pos_lags
            save_data[f'{key}_neg_lags'] = neg_lags
            save_data[f'{key}_ucomp_acf_pos'] = ucomp_acf_pos
            save_data[f'{key}_ucomp_acf_neg'] = ucomp_acf_neg
            save_data[f'{key}_div1_acf_pos'] = div1_acf_pos
            save_data[f'{key}_div1_acf_neg'] = div1_acf_neg

    titles = [
        (axes[0, 0], r'Power spectrum of $[\overline{u}]_s$', 'freq'),
        (axes[0, 1], r'Lagged autocorrelation of $[\overline{u}]_s$', 'lag'),
        (axes[1, 0], r'Power spectrum of $[\overline{m}]_s$', 'freq'),
        (axes[1, 1], r'Lagged autocorrelation of $[\overline{m}]_s$', 'lag'),
    ]
    for ax, title, kind in titles:
        ax.grid(True)
        ax.set_title(title)
        ax.legend(fontsize=7)
        if kind == 'freq':
            ax.set_xlim(0., 0.25)
            ax.set_xlabel('frequency (1/days)')
            ax.set_ylabel('power spectral density')
        else:
            ax.axhline(0, color='k', linewidth=0.5)
            ax.axvline(0, color='k', linewidth=0.5)
            ax.set_xlim(-30., 30.)
            ax.set_ylim(-0.2, 1.0)
            ax.set_yticks(np.arange(-0.2, 1.0 + 0.2, 0.2))
            ax.set_xlabel('lag (days)')
            ax.set_ylabel('lagged correlation')

    for ax, label in zip(axes.flat, ['(a)', '(b)', '(c)', '(d)']):
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=13,
                 fontweight='bold', va='top', ha='left')

    fig.suptitle(f'jra55_850 vs jra_simpson_lag - {hemisphere}H, {time_frame} (solid=va, dashed=native)')
    plt.tight_layout()

    data_dir, plot_dir = data_and_plot_dirs(time_frame)

    npz_file = os.path.join(data_dir, 'fig3_pw-spectra_autocorr.npz')
    np.savez(npz_file, **save_data)
    print(f'Saved data to {npz_file}')

    out_file = os.path.join(plot_dir, 'fig3_pw-spectra_autocorr.png')
    plt.savefig(out_file)
    plt.close(fig)
    print(f'Saved figure to {out_file}')


# ── Figure 3: lagged cross-correlation ────────────────────────────────────────

def make_cross_correlation_figure(time_frame):
    fig, ax = plt.subplots(figsize=(7, 5))
    save_data = {}

    for method_key, method_info in methods.items():
        eof_ds = method_info['eof_ds']
        color, label = method_info['color'], method_info['label']

        for va_label, va_str in va_str_dict.items():
            names = pc_names(va_str, time_frame)
            linestyle = variant_linestyle[va_label]
            line_label = f'{label} ({va_label})'

            pos_lags, cross_corr_pos, neg_lags, cross_corr_neg = cross_corr_pos_neg(
                eof_ds, names['ucomp_self'], names['ucomp_pseudo'], names['div1_pseudo'], time_frame)

            ax.plot(pos_lags, cross_corr_pos, color=color, linestyle=linestyle, label=line_label)
            ax.plot(neg_lags, cross_corr_neg, color=color, linestyle=linestyle)

            key = f'{method_key}_{va_label}'
            save_data[f'{key}_pos_lags'] = pos_lags
            save_data[f'{key}_neg_lags'] = neg_lags
            save_data[f'{key}_cross_corr_pos'] = cross_corr_pos
            save_data[f'{key}_cross_corr_neg'] = cross_corr_neg

    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlim(-30., 30.)
    ax.set_ylim(-0.2, 0.6)
    ax.set_yticks(np.arange(-0.2, 0.6 + 0.2, 0.2))
    ax.set_xlabel('lag (days)')
    ax.set_ylabel('lagged correlation')
    ax.grid(True)
    ax.legend(fontsize=8)
    ax.set_title(f'Lagged cross-correlation of $[\\overline{{u}}]_s$ and $[\\overline{{m}}]_s$ ({time_frame}, solid=va, dashed=native)')

    ax.text(15, 0.5, r'$[\overline{u}]_s$ leads $[\overline{m}]_s$', fontsize=10, ha='center', va='center')
    ax.text(-15, 0.5, r'$[\overline{m}]_s$ leads $[\overline{u}]_s$', fontsize=10, ha='center', va='center')

    plt.tight_layout()

    data_dir, plot_dir = data_and_plot_dirs(time_frame)

    npz_file = os.path.join(data_dir, 'fig4_cross-correlation.npz')
    np.savez(npz_file, **save_data)
    print(f'Saved data to {npz_file}')

    out_file = os.path.join(plot_dir, 'fig4_cross-correlation.png')
    plt.savefig(out_file)
    plt.close(fig)
    print(f'Saved figure to {out_file}')


for time_frame in time_frames:
    make_cospec_coher_pdiff_figure(time_frame)
    make_pw_spectra_autocorr_figure(time_frame)
    make_cross_correlation_figure(time_frame)

print('Done.')
