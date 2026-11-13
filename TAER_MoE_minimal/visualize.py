"""Write exactly one 1D CSV table and three environmental/feature PNG figures."""
import argparse
import csv
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'tc3-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

OUTPUT_FILES = {'data_1d.csv', 'environment_3d.png', 'features6.png', 'features10.png'}
TRACK = 'bst_divi10_train_val_test_inlcude15_2023_new'


def render(root, out, track=None, start=0, obs_len=8, pred_len=4):
    if track is None:
        tracks = sorted((root / TRACK).glob('*/test/*.txt'))
        if len(tracks) != 1:
            raise ValueError('Choose one test storm with --track relative/path/to/track.txt')
        track = tracks[0]
    else:
        track = root / track
    rows = [line.split() for line in track.read_text().splitlines() if line.strip()]
    rows = rows[start:start + obs_len + pred_len]
    if len(rows) != obs_len + pred_len:
        raise ValueError('The selected window does not contain obs_len + pred_len records')
    area, name = track.name[:2], rows[0][-1]

    def asset(modality, row, suffix='.npy'):
        parent = root / modality / area / row[-2][:4]
        for storm in (name, name.upper(), name.lower()):
            path = parent / storm / (row[-2] + suffix)
            if path.is_file():
                return path
        raise FileNotFoundError(str(parent / name / (row[-2] + suffix)))

    values = np.array([[float(v) for v in row[2:6]] for row in rows], dtype=np.float64)
    series = values * np.array([5., 5., 50., 25.]) + np.array([180., 0., 960., 40.])
    n = obs_len
    with (out / 'data_1d.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['timestamp_utc', 'role', 'longitude_deg_e', 'latitude_deg_n', 'pressure_hpa', 'wind_ms'])
        writer.writerows([row[-2], 'observed' if i < n else 'future truth', *series[i]]
                         for i, row in enumerate(rows))

    gph = np.stack([np.load(asset('all_ocean_gph500_2023', row)) for row in rows[:n]])
    yy, xx = np.mgrid[:gph.shape[1], :gph.shape[2]]
    norm = Normalize(float(gph.min()), float(gph.max()))
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n):
        ax.plot_surface(xx, yy, np.full_like(xx, i * 6), facecolors=plt.cm.viridis(norm(gph[i])),
                        rstride=4, cstride=4, shade=False, alpha=.8)
    ax.set(xlabel='Grid x (pixel)', ylabel='Grid y (pixel)', zlabel='Hours since first observation',
           title=name + ' — GPH500 time × y × x (single pressure level)')
    ax.view_init(elev=25, azim=-60)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, shrink=.6,
                 label='Stored GPH500 value')
    fig.savefig(out / 'environment_3d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Keep stored map values and channel indices; no physical-unit conversion is inferred.
    feature_labels = {
        6: ['msl', 'wind10', 't850', 't500', 't300', 'vorticity850'],
        10: ['t2m proxy', 't850', 't500', 'DIV', 'z850', 'RH1000', 'u850', 'v850', 'shear speed', 'shear direction'],
    }
    for channels in (6, 10):
        modality = 'features' + str(channels)
        maps = np.load(asset(modality, rows[n - 1], '_' + modality + '.npy'))
        if maps.ndim != 3 or maps.shape[0] != channels:
            raise ValueError('Unexpected spatial feature shape: ' + str(maps.shape))
        cols = 3 if channels == 6 else 5
        fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 7))
        for c, ax in enumerate(axes.flat):
            im = ax.imshow(maps[c], origin='upper', cmap='coolwarm')
            ax.set_title('F%d[%d]: %s' % (channels, c, feature_labels[channels][c]))
            ax.set_xlabel('Grid x (pixel)')
            ax.set_ylabel('Grid y (pixel)')
            fig.colorbar(im, ax=ax, shrink=.7)
        fig.suptitle('%s — stored maps at %s; visualization assets' % (name, rows[n - 1][-2]))
        fig.tight_layout()
        fig.savefig(out / (modality + '.png'), dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data_demo')
    parser.add_argument('--track', help='Track file relative to data-dir; omitted for a one-storm demo')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--obs-len', type=int, default=8)
    parser.add_argument('--pred-len', type=int, default=4)
    parser.add_argument('--output', default='visualizations', help='Dedicated visualization output directory')
    args = parser.parse_args()
    root, out = Path(args.data_dir), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.start < 0 or min(args.obs_len, args.pred_len) < 1:
        parser.error('start must be nonnegative; observation/prediction lengths must be positive')
    unknown = [p.name for p in out.iterdir() if p.name not in OUTPUT_FILES]
    if unknown:
        parser.error('Output contains unrelated files; choose a dedicated output directory: ' + ', '.join(sorted(unknown)))
    if any(p.is_symlink() or not p.is_file() for p in out.iterdir()):
        parser.error('Output must contain only regular visualization files')
    # Complete all four renders before replacing existing results.
    with tempfile.TemporaryDirectory(prefix='.visualize-', dir=str(out.parent)) as temporary:
        staging = Path(temporary)
        render(root, staging, args.track, args.start, args.obs_len, args.pred_len)
        for name in sorted(OUTPUT_FILES):
            (staging / name).replace(out / name)
    print('Saved exactly 4 files to ' + str(out))
    for name in sorted(OUTPUT_FILES):
        print('  ' + name)


if __name__ == '__main__':
    main()
