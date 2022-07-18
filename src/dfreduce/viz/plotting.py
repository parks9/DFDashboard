import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from .. import utils


__all__ = ['show_image', 'tile_images']


def show_image(path_or_pixels, percentile=[1, 99], subplots=None,
               cmap='gray_r', rasterized=False, **kwargs):
    pixels = utils.load_path_or_pixels(path_or_pixels)
    if subplots is None:
        figsize = kwargs.pop('figsize', (10, 10))
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw=dict(xticks=[], yticks=[]), 
                               **kwargs)
    else:
        fig, ax = subplots
    if percentile is not None:
        vmin, vmax = np.nanpercentile(pixels, percentile)
    else:
        vmin, vmax = None, None
    ax.imshow(pixels, origin='lower', cmap=cmap, rasterized=rasterized,
              vmin=vmin, vmax=vmax)
    return fig, ax


def tile_images(path_or_pixels, ncols=4, size_scale_factor=3, 
                number_frames=False, percentile=[1, 99], fontsize=25, 
                trim_zeros=False):
    pixels = [utils.load_path_or_pixels(p) for p in path_or_pixels]
    ratio = pixels[0].shape[1] / pixels[0].shape[0]
    nrows = int(np.ceil(len(pixels) / ncols))
    figsize= (ratio * ncols * size_scale_factor, nrows * size_scale_factor)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize,
        subplot_kw=dict(xticks=[], yticks=[], aspect='equal'))
    fig.subplots_adjust(hspace=0.04, wspace=0.03)
    for num, (ax, img) in enumerate(zip(axes.flat, pixels)):
        if trim_zeros:
            not_zero = ~np.all(img == 0.0, axis=1)
            i_low, i_high = np.argwhere(not_zero)[[0, -1]].flatten()
            not_zero = ~np.all(img == 0.0, axis=0)
            j_low, j_high = np.argwhere(not_zero)[[0, -1]].flatten()
            _img = img[i_low:i_high, j_low:j_high]
        else:
            _img = img
        fig, ax = show_image(_img, subplots=(fig, ax), percentile=percentile)
        if number_frames:
            ax.text(0.95, 0.95, num, transform=ax.transAxes, color='c', 
                    fontsize=fontsize, ha='right', va='top', 
                    bbox=dict(boxstyle='Square,pad=0.2', color='k'))
    for i in range(len(axes.flatten()) - len(pixels)):
        axes.flatten()[-(i+1)].set_visible(False)
    return fig, axes


def view_reduction_results(data_path, image_type, save_path=None, show=False,
                           sort_files=True, print_files=False, **kwargs):
    frames = glob(os.path.join(data_path, '*_' + image_type + '*'))
    if sort_files:
        print('sorting frames')
        frames.sort()
    if print_files:
        for fn in frames:
            print(fn)
    pixels = [fits.getdata(fn) for fn in frames]
    fig, axes = tile_image_pixels(pixels, number_images=True, **kwargs)
    label = os.path.basename(frames[0]).split('_')[0]
    suptitle = fig.suptitle('DragonflyReduceRT {} images: {}'.\
        format(image_type, label), fontsize=25, y=1.02)
    if save_path is not None:
        fn = os.path.join(save_path, '{}-{}.png'.format(label, image_type))
        plt.tight_layout()
        fig.savefig(fn, dpi=200, bbox_extra_artists=(suptitle,),
                    bbox_inches='tight')
    else:
        plt.show()


def display_nearest_master_cals(frames_db, date=None, frame_type=None,
        serialno=None, exptime=None, subplots=None, save_as=None,**kwargs):
    """ Visualize the matching between individual frames and master cals.
    """

    titles = ['Matching flats with master darks',
              'Matching lights with master darks',
              'Matching lights with master flats']

    frame_types = ['flat', 'light', 'light']
    mastercal_types = ['dark', 'dark', 'flat']

    fig, axes = plt.subplots(3, 1, figsize=(12,15))
    fig.subplots_adjust(hspace=0.5)

    mask = frames_db.mask_database(**kwargs)

    if date is not None:
        date_mask = frames_db.table.date == date
    else:
        date_mask = np.ones(frames_db.nrows, dtype=bool)

    if frame_type is not None:
        frame_type_mask = frames_db.table.frame_type == frame_type
    else:
        frame_type_mask = frames_db.table.frame_type != 'dark'

    if serialno is not None:
        serialno_mask = frames_db.table.serialno == serialno
    else:
        serialno_mask = np.ones(frames_db.nrows, dtype=bool)

    if exptime is not None:
        exptime_mask = frames_db.table.exptime == exptime
    else:
        exptime_mask = np.ones(frames_db.nrows, dtype=bool)

    subset = frames_db.table[date_mask & frame_type_mask &
                        serialno_mask & exptime_mask]

    for ind, ax in enumerate(axes):
        ax.set_title(titles[ind])
        ax.set_ylabel('Time delay')
        ax.set_xlabel('Frame')

        if mastercal_types[ind] == 'dark':
            mastercal_dates = pd.to_datetime(subset.master_dark_date)
        elif mastercal_types[ind] == 'flat':
            mastercal_dates = pd.to_datetime(subset.master_flat_date)

        datediff = mastercal_dates - subset.date

        datediff = datediff[subset.frame_type == frame_types[ind]]
        datediff_ns = datediff.values.astype(np.float)
        datediff_days = datediff_ns / 8.64e13

        frame_names = subset.frame[subset.frame_type == frame_types[ind]]

        if len(datediff_days) > 0:

            if np.abs(datediff_days).max() == 0:
                ax.set_ylim(-1, 1)

            ax.plot(frame_names, datediff_days, 'o', color='silver', ms=5)

            x_vals = np.array(frame_names)
            x_labels = np.array([s.split('/')[1] for s in frame_names])

            print(x_vals)
            print(x_labels)

            plt.xticks(x_vals, x_labels, rotation=70)

        else:
            ax.set_yticks([])
            ax.set_xticks([])

    return fig, axes
