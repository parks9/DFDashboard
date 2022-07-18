import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from matplotlib.patches import Rectangle
from ..astrometry import pixel_area_map
from ..tasks import check_astrometric_solution
from .. import utils
from .. import package_dir

style = os.path.join(package_dir, 'viz/df.mplstyle')
plt.style.use(style)


__all__ = ['wcs_quality_check', 'plot_distortion']


def wcs_quality_check(check_astrom_out_or_kw, xlim=[-4.5, 4.5],
                      ylim=[-4.5, 4.5], levels=25, subplots=None, 
                      scatter_cmap='RdYlBu_r', fontsize=20, scatter_alpha=0.9, 
                      draw_pam=True, draw_hists=True,  hist_percent_ax=0.2, 
                      scatter_ec='k', pam_cmap='gray', pam_levels=25, 
                      pam_line_color='gray', pam_line_alpha=0.5, **kwargs):
    if subplots is None:
        figsize = kwargs.pop('figsize', (16, 6))
        fig, ax = plt.subplots(1, 2, figsize=figsize, **kwargs)
        fig.subplots_adjust(wspace=0.25)

    if type(check_astrom_out_or_kw) == dict:
        checked_astrom = check_astrometric_solution(**check_astrom_out_or_kw)
    else:
        checked_astrom = check_astrom_out_or_kw

    # plot 1: offset ra & dec compared to reference catalog
    # points are colored by distance from the center of the image
    sc_0 = ax[0].scatter(checked_astrom['delta_ra'],
                         checked_astrom['delta_dec'],
                         c=checked_astrom['dr_image'],
                         s=35, cmap=scatter_cmap, alpha=scatter_alpha, 
                         ec=scatter_ec, zorder=10)
    cbar_0 = fig.colorbar(sc_0, ax=ax[0])
    cbar_0.ax.set_ylabel(r'$\Delta$R [pixels]', fontsize=fontsize)
    cbar_0.solids.set_edgecolor('face')
    ax[0].grid(True)
    ax[0].axhline(y=0, lw=2, color='k')
    ax[0].axvline(x=0, lw=2, color='k')
    
    if draw_hists:
        kw = dict(alpha=0.1, color='k')
        ax_histx = ax[0].twinx()
        counts, bins = np.histogram(checked_astrom['delta_ra'], bins='auto')
        w = np.ones_like(checked_astrom['delta_ra']) / counts.max()
        ax_histx.hist(checked_astrom['delta_ra'], bins=bins, weights=w, **kw)
        ax_histx.hist(checked_astrom['delta_ra'], bins=bins,
                      weights=w, color='k', histtype='step', lw=2)
        ax_histx.set(yticks=[])
        ax_histx.set_ylim(0, 1 / hist_percent_ax)
        
        ax_histy = ax[0].twiny()
        counts, bins = np.histogram(checked_astrom['delta_dec'], bins='auto')
        w = np.ones_like(checked_astrom['delta_dec']) / counts.max()
        ax_histy.hist(checked_astrom['delta_dec'], bins=bins, 
                      weights=w, orientation='horizontal', **kw)
        ax_histy.hist(checked_astrom['delta_dec'], bins=bins, 
                      weights=w, orientation='horizontal', color='k', 
                      histtype='step', lw=2)
        ax_histy.set(xticks=[])
        ax_histy.set_xlim(0, 1 / hist_percent_ax)
    
    ax[0].set_xlabel(r'$\Delta \alpha$ [arcsec]', fontsize=fontsize)
    ax[0].set_ylabel(r'$\Delta \delta$ [arcsec]', fontsize=fontsize)
    ax[0].set_xlim(*xlim)
    ax[0].set_ylim(*ylim)
    ax[0].minorticks_on()

    # plot 2: matched sources
    # points are colored by the magnitude of the coorinate offset
    nx = checked_astrom['header']['NAXIS1']
    ny = checked_astrom['header']['NAXIS2']
    if draw_pam:
        pam = pixel_area_map(checked_astrom['header'])
        yy, xx = np.mgrid[1:ny+1, 1:nx+1]
        ax[1].contourf(xx, yy, pam, cmap=pam_cmap, 
                       levels=pam_levels, zorder=-10)
        ax[1].contour(xx, yy, pam, colors=pam_line_color,
                      alpha=pam_line_alpha, levels=pam_levels, zorder=-5)
    sc_1 = ax[1].scatter(checked_astrom['cat_match']['X_IMAGE'],
                         checked_astrom['cat_match']['Y_IMAGE'],
                         c=checked_astrom['angular_sep'], s=35,
                         cmap=scatter_cmap, alpha=scatter_alpha, ec=scatter_ec)
    cbar_1 = fig.colorbar(sc_1, ax=ax[1])
    cbar_1.ax.set_ylabel(r'$|\Delta\omega|$ [arcsec]', fontsize=fontsize)
    cbar_1.solids.set_edgecolor('face')
    ax[1].set_xlabel(r'$x$ [pixel]', fontsize=fontsize)
    ax[1].set_ylabel(r'$y$ [pixel]', fontsize=fontsize)
    ax[1].add_patch(Rectangle([0, 0], nx+1, ny+1, lw=3, ec='k', fc='none'))
    ax[1].axis('equal')
    ax[1].minorticks_on()

    return fig, ax


def plot_distortion(path_or_header, magnification=100, num_grid_lines=20, 
                    subplots=None, fontsize=26, **kwargs):
    """Draw a cartesian coordinate grid distorted by the WCS."""

    if subplots is None:
        figsize = kwargs.pop('figsize', (8, 6))
        fig, ax = plt.subplots(figsize=figsize, **kwargs)
        ax.set(xticks=[], yticks=[])
    header = utils.load_path_or_header(path_or_header) 
    wcs = WCS(header)

    # Define the grid sampling
    nx = header['NAXIS1']
    ny = header['NAXIS2']

    x = np.linspace(0, header['NAXIS1'], num=num_grid_lines)
    y = np.linspace(0, header['NAXIS2'], num=num_grid_lines)

    # Horizontal lines
    undistorted_lines = []
    distorted_lines = []
    for yo in y:
        u_dummy = []
        d_dummy = []
        for xo in x:
            u_dummy.append([xo,yo])
            # Distortion-corrected
            ra, dec = wcs.all_pix2world([xo], [yo], 1, ra_dec_order=True)       
            # Not distortion-corrected
            [x_u, y_u] = wcs.wcs_world2pix(ra, dec, 1)                          
            x_u = xo + magnification*(x_u - xo)
            y_u = yo + magnification*(y_u - yo)
            d_dummy.append([x_u[0], y_u[0]])
        undistorted_lines.append(u_dummy)
        distorted_lines.append(d_dummy)

    for i in range(0, len(distorted_lines)):
        xl, yl = np.transpose(undistorted_lines[i])
        ax.plot(xl, yl,'k-', lw=1.5)
        xl, yl = np.transpose(distorted_lines[i])
        ax.plot(xl, yl,'r-', lw=2)

    # Vertical lines
    for xo in x:
        u_dummy = []
        d_dummy = []
        for yo in y:
            u_dummy.append([xo,yo])
            ra, dec = wcs.all_pix2world([xo], [yo], 1, ra_dec_order=True)       
            [x_u, y_u] = wcs.wcs_world2pix(ra, dec, 1)                         
            x_u = xo + magnification*(x_u - xo)
            y_u = yo + magnification*(y_u - yo)
            d_dummy.append([x_u[0], y_u[0]])
        undistorted_lines.append(u_dummy)
        distorted_lines.append(d_dummy)

    for i in range(0, len(distorted_lines)):
        xl, yl = np.transpose(undistorted_lines[i])
        ax.plot(xl, yl,'k-', lw=1.5)
        xl, yl = np.transpose(distorted_lines[i])
        ax.plot(xl, yl,'r-', lw=2)

    # Set the plot range to correspond to the image dimensions.
    ax.set_title("Exaggeration: {:.0f}x".format(magnification), 
                 fontsize=fontsize, y=1.015)
    ax.axis('square')
    fig.tight_layout()
    ax.axis([1, header['NAXIS1'], 1, header['NAXIS2']])

    return fig, ax
