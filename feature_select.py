import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. 读取 Excel 数据
df = pd.read_excel("整合数据\merged_vad_mad.xlsx", sheet_name="Sheet1")

# 2. 输出 VAD 和 MAD 的统计信息
print("=== VAD 统计特征 ===")
vad_stats = df['Vad'].describe()
print(vad_stats)

print("\n=== MAD 统计特征 ===")
mad_stats = df['Mad'].describe()
print(mad_stats)

# 3. 绘制 VAD 和 MAD 直方图（两个子图）
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(df['Vad'], bins=10, color='skyblue', edgecolor='black')
axes[0].set_xlabel('$V_{ad}$ (%)')
axes[0].set_ylabel('Frequency')
# axes[0].set_title('Distribution of $V_{ad}$')
axes[0].grid(alpha=0.3)

axes[1].hist(df['Mad'], bins=10, color='lightcoral', edgecolor='black')
axes[1].set_xlabel('$M_{ad}$ (%)')
axes[1].set_ylabel('Frequency')
# axes[1].set_title('Distribution of $M_{ad}$')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("imag/histograms_vad_mad.svg", format="svg")
plt.show()

# 4. 光谱数据（假设 Sample、Vad、Mad 在前3列，其余为光谱）
spectral_df = df.drop(columns=['Sample', 'Vad', 'Mad'])
wavelengths = spectral_df.columns.astype(float)

# 5. 分别计算光谱均值和标准差
mean_spectrum_vad = spectral_df[df['Vad'].notnull()].mean(axis=0)
std_spectrum_vad = spectral_df[df['Vad'].notnull()].std(axis=0)

mean_spectrum_mad = spectral_df[df['Mad'].notnull()].mean(axis=0)
std_spectrum_mad = spectral_df[df['Mad'].notnull()].std(axis=0)

# 6. 绘制光谱均值 ± 标准差（VAD & MAD 同图）
plt.figure(figsize=(8,5))

plt.plot(wavelengths, mean_spectrum_vad, color='blue', label='Mean spectrum')
plt.fill_between(wavelengths,
                 mean_spectrum_vad - std_spectrum_vad,
                 mean_spectrum_vad + std_spectrum_vad,
                 color='lightblue', alpha=0.5)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
# plt.title('Mean NIR Spectrum with ±1 Standard Deviation')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("imag/mean_spectrum.svg", format="svg")
plt.show()

# 7. 选取代表性样本（按分位点）并绘制其光谱
def _select_indices_by_quantiles(series, quantiles):
    s = series.dropna().sort_values()
    n = len(s)
    if n == 0:
        return []
    ranks = [int(round(q * (n - 1))) for q in quantiles]
    ranks = [min(max(r, 0), n - 1) for r in ranks]
    return list(s.index[ranks])

def _plot_representative_spectra(df, spectral_df, value_col, quantiles, wavelengths, save_path, cmap_name='viridis'):
    indices = _select_indices_by_quantiles(df[value_col], quantiles)
    if len(indices) == 0:
        print(f"No data for {value_col}, skip.")
        return

    colors = plt.cm.get_cmap(cmap_name)(np.linspace(0, 1, len(indices)))
    plt.figure(figsize=(8, 5))
    for color, idx, q in zip(colors, indices, quantiles):
        spectrum = spectral_df.loc[idx].values.astype(float)
        var_map = {'Mad': r'$M_{ad}$', 'Vad': r'$V_{ad}$'}
        var_disp = var_map.get(value_col, value_col)
        label = f"{df.loc[idx, 'Sample']} - {var_disp} quantile={q:.0%}"
        plt.plot(wavelengths, spectrum, color=color, label=label, linewidth=1.5)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fmt = save_path.split('.')[-1]
    plt.savefig(save_path, format=fmt)
    plt.show()

# 分位点（从左到右选取几个代表性样本）
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]

# 基于 VAD 的代表性样本光谱
_plot_representative_spectra(
    df=df,
    spectral_df=spectral_df,
    value_col='Vad',
    quantiles=quantiles,
    wavelengths=wavelengths,
    save_path="imag/vad_representative_spectra.svg",
)

# 基于 MAD 的代表性样本光谱
_plot_representative_spectra(
    df=df,
    spectral_df=spectral_df,
    value_col='Mad',
    quantiles=quantiles,
    wavelengths=wavelengths,
    save_path="imag/mad_representative_spectra.svg",
)

# 8. 合并输出：左右并排的代表性样本光谱（VAD | MAD）
def _plot_representative_on_ax(ax, df, spectral_df, value_col, quantiles, wavelengths, cmap_name='viridis'):
    indices = _select_indices_by_quantiles(df[value_col], quantiles)
    if len(indices) == 0:
        ax.text(0.5, 0.5, f"No data for {value_col}", ha='center', va='center')
        return
    colors = plt.cm.get_cmap(cmap_name)(np.linspace(0, 1, len(indices)))
    var_map = {'Mad': r'$M_{ad}$', 'Vad': r'$V_{ad}$'}
    var_disp = var_map.get(value_col, value_col)
    for color, idx, q in zip(colors, indices, quantiles):
        spectrum = spectral_df.loc[idx].values.astype(float)
        label = f"{df.loc[idx, 'Sample']} - {var_disp} quantile={q:.0%}"
        ax.plot(wavelengths, spectrum, color=color, label=label, linewidth=1.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorbance')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
_plot_representative_on_ax(axes[0], df, spectral_df, 'Vad', quantiles, wavelengths, cmap_name='viridis')
axes[0].set_title('Representative spectra by $V_{ad}$ quantiles')

_plot_representative_on_ax(axes[1], df, spectral_df, 'Mad', quantiles, wavelengths, cmap_name='plasma')
axes[1].set_title('Representative spectra by $M_{ad}$ quantiles')

plt.tight_layout()
plt.savefig("imag/representative_spectra_side_by_side.svg", format="svg")
plt.show()

# 9. Plot the same sample across different sampling distances (choose Vad or Mad)
def _read_distance_file(target: str, distance_mm: int, root_dir: str) -> pd.DataFrame:
    subdir = "煤炭-挥发分-不同距离" if target.lower() == "vad" else "煤炭-水分-不同距离"
    fname = (f"Vad_{distance_mm}mm.xlsx" if target.lower() == "vad" else f"mad_{distance_mm}mm.xlsx")
    path = os.path.join(root_dir, subdir, fname)
    return pd.read_excel(path)


# 10. Align samples across distances by base ID and plot by index
def _get_base_sample_id(sample_str: str, distances=(5, 10, 15, 20)) -> str:
    s = str(sample_str).strip()
    if s.endswith('.sam'):
        s = s[:-4]
    parts = s.split('-')
    if len(parts) >= 2 and parts[-1].isdigit() and int(parts[-1]) in distances:
        base_id = '-'.join(parts[:-1])
    else:
        base_id = s
    return base_id

def _add_base_id_column(df: pd.DataFrame, distances=(5, 10, 15, 20)) -> pd.DataFrame:
    if 'Sample' in df.columns:
        df = df.copy()
        df['base_id'] = df['Sample'].apply(lambda x: _get_base_sample_id(x, distances))
    else:
        df = df.copy()
        df['base_id'] = None
    return df

def _extract_spectrum_by_row(df: pd.DataFrame, row: pd.Series):
    drop_cols = [c for c in ['Sample', 'Vad', 'Mad', 'base_id'] if c in df.columns]
    spec_df = df.drop(columns=drop_cols)
    # map columns convertible to float -> wavelength
    col_map = {}
    for c in spec_df.columns:
        try:
            col_map[c] = float(c)
        except Exception:
            # skip non-numeric column names
            pass
    if not col_map:
        keep_cols = [c for c in spec_df.columns if np.issubdtype(spec_df[c].dtype, np.number)]
        wavelengths = np.array(keep_cols, dtype=float) if keep_cols else None
        spectrum = row[keep_cols].values.astype(float)
    else:
        keep_cols = [c for c in spec_df.columns if c in col_map]
        wavelengths = np.array([col_map[c] for c in keep_cols], dtype=float)
        spectrum = row[keep_cols].values.astype(float)
    order = np.argsort(wavelengths)
    return wavelengths[order], spectrum[order]

def list_aligned_samples(target='Vad', distances=(5, 10, 15, 20), root_dir=os.getcwd()):
    target = target.capitalize()
    dfs = {}
    for d in distances:
        df_d = _read_distance_file(target, d, root_dir)
        dfs[d] = _add_base_id_column(df_d, distances)
    # intersection of base_ids across all distances
    sets = [set(dfs[d]['base_id'].dropna().tolist()) for d in distances]
    common_ids = set.intersection(*sets) if sets else set()
    return sorted(common_ids)

def plot_sample_spectra_across_distances_by_index(sample_idx: int, target='Vad', distances=(5, 10, 15, 20), root_dir=os.getcwd(), save_path=None):
    target = target.capitalize()
    assert target in ('Vad', 'Mad'), "target must be 'Vad' or 'Mad'"
    # load and add base_id
    dfs = {}
    for d in distances:
        df_d = _read_distance_file(target, d, root_dir)
        dfs[d] = _add_base_id_column(df_d, distances)
    # compute common base_ids
    common_ids = list_aligned_samples(target=target, distances=distances, root_dir=root_dir)
    if not common_ids:
        print('[WARN] No aligned samples across distances.')
        return
    # 1-based index selection
    if sample_idx < 1 or sample_idx > len(common_ids):
        print(f"[WARN] sample_idx out of range. Valid 1..{len(common_ids)}")
        return
    base_id = common_ids[sample_idx - 1]
    # extract curves
    curves = []
    for d in distances:
        df_d = dfs[d]
        row = df_d[df_d['base_id'] == base_id]
        if row.empty:
            print(f"[WARN] base_id '{base_id}' missing at {d} mm, skip.")
            continue
        wl, spec = _extract_spectrum_by_row(df_d, row.iloc[0])
        curves.append((d, wl, spec))
    if not curves:
        print('[WARN] No curves extracted.')
        return
    # plot
    cmap = plt.cm.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(curves)))
    plt.figure(figsize=(8, 5))
    for (color, (d, wl, spec)) in zip(colors, curves):
        plt.plot(wl, spec, label=f"{d} mm", color=color, linewidth=1.8)
    var_map = {'Mad': r'$M_{ad}$', 'Vad': r'$V_{ad}$'}
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.title(f"Sample {base_id}: spectra at different distances ({var_map[target]})")
    plt.grid(alpha=0.3)
    plt.legend(title='Distance')
    plt.tight_layout()
    if save_path is None:
        safe_id = str(base_id).replace(' ', '_')
        save_path = f"imag/{target.lower()}_sample_{safe_id}_by_distance.svg"
    fmt = save_path.split('.')[-1]
    plt.savefig(save_path, format=fmt)
    plt.show()

# Example usage (uncomment):
# ids = list_aligned_samples(target='Vad', distances=(5,10,15,20))
# print(f"Aligned samples (count={len(ids)}):", ids[:10])
# plot_sample_spectra_across_distances_by_index(sample_idx=1, target='Vad')
