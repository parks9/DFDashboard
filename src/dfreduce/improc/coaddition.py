# Standard library 
import os
from multiprocessing import RawArray

# Third-party
import numpy as np
from scipy import ndimage
from  astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from tqdm import tqdm

# Project
from ..utils import load_pixels_and_header
from ..cameras import get_filter_name
from .. import utils, DFStruct, logger
from .core import grid_image


__all__  = [
    'deep_stack',
    'MedianCoadder', 
    'WeightedAverageCoadder']


def deep_stack(coadd_path):
    fn_g = os.path.join(coadd_path, 'coadd_median_g.fits')
    fn_r = os.path.join(coadd_path, 'coadd_median_r.fits')
    if os.path.isfile(fn_g) and os.path.isfile(fn_r):
        gdata = fits.getdata(fn_g)
        rdata, header = fits.getdata(fn_r, header=True)
        coadd_data = np.sum([gdata, rdata], axis=0)
    elif os.path.isfile(fn_g):
        logger.warning('r-band image does not exist! using g-band only.')
        coadd_data, header = fits.getdata(fn_g, header=True)
    elif os.path.isfile(fn_r):
        logger.warning('g-band image does not exist! using r-band only.')
        coadd_data, header = fits.getdata(fn_r, header=True)
    else:
        logger.critical('median images do not exist!')
        coadd_data = None
        header = None
    coadd = DFStruct(pixels=coadd_data, header=header)
    return coadd


def _bad_pixel_mask(path, flux_scale=1, thresh=2.5, med_pix=None,
                    back_val=None, gain=0.37, write=True, bpm_path='/tmp'):
    
    pixels, header = load_pixels_and_header(path, None)
                      
    g = gain
    sky = header['MEDSKY']
    sqrt_term = g * med_pix / flux_scale + g * sky
    sqrt_term[sqrt_term < 0] = np.nan
    noise = flux_scale * np.sqrt(sqrt_term) / g 
    noise[pixels==0] = 0.

    med_bs = med_pix - back_val
    residual = pixels * flux_scale - med_bs 
    residual[pixels==0] = 0.
    stddev = residual.copy()
    stddev[pixels!=0] = residual[pixels!=0] / noise[pixels!=0]
    
    mask = np.greater(np.abs(stddev), thresh, where=np.isfinite(stddev))
    structure = ndimage.generate_binary_structure(2, 2)
    mask = ndimage.morphology.binary_dilation(mask.astype(float), structure)

    unmask_stars = residual / ndimage.median_filter(med_bs, (3, 3))
    unmask_stars = np.less(
        np.abs(unmask_stars), 1.0, where=np.isfinite(unmask_stars)) 

    mask[(mask==1) & unmask_stars] = 0.
    masked_frac = mask[pixels != 0].sum() / len(mask[pixels != 0])
    
    results = DFStruct(mask=mask, masked_frac=masked_frac, 
                       noise=noise, residual=residual)

    if write:
        out_fn = os.path.basename(path).replace('.fits', '_bpm.fits')
        out_fn = os.path.join(bpm_path, out_fn)
        h = header.copy()
        h['BPFRAC'] = masked_frac
        utils.write_pixels(out_fn, mask.astype(np.uint8), header=h)
    
    return results


# The following two functions are are passed to Pool.
# Their purpose is to setup the median coadd to be shared between processors.
var_dict = {}
def _bpm_mp_init(med_raw_array, shape):
    var_dict['pix'] = med_raw_array
    var_dict['shape'] = shape


def _bad_pixel_mask_mp(path, flux_scale, **kwargs):
    _bad_pixel_mask(path, flux_scale, **kwargs)


class BaseCoadder(object):

    _base_header_keys = [
     'SIMPLE',
     'BITPIX',
     'NAXIS',
     'NAXIS1',
     'NAXIS2',
     'PIXSCALE',
     'TARGET',
     'DATE'
    ]
    
    def __init__(self, paths):
        self.paths = np.asarray(paths)
        self.is_good = np.ones_like(self.paths, dtype=bool)
        self.shape = utils.get_image_shape(self.paths[0])
        
    def _get_header_params(self):
        params = dict(zp_space=[], zp_median=[],
                      airmass=[], zp_fid=[], 
                      sky_med=[], filter_name=[], 
                      exptime=[])
        for p in self.paths:
            header = utils.load_path_or_header(p)
            params['zp_fid'].append(header['FIDZP'])
            params['sky_med'].append(header['MEDSKY'])
            params['airmass'].append(header['AIRMASS'])
            params['zp_space'].append(header['ZPSPACE'])
            params['zp_median'].append(header['MEDIANZP'])
            params['exptime'].append(header['EXPTIME'])
            params['filter_name'].append(get_filter_name(header))            
        self.frame_params = Table(params)
        
    def _paths_by_filter(self, bandpass):
        if bandpass is None or bandpass == 'all':
            filter_cut = np.ones_like(self.paths, dtype=bool)
        else:
            filter_cut = self.frame_params['filter_name'] == bandpass
        return filter_cut
    
    def _setup(self, bandpass, coadd_type, nseg):
        self.reset()
        self.coadd_type = coadd_type
        self.bandpass = bandpass.upper()
        self._get_header_params()
        filter_cut = self._paths_by_filter(bandpass.upper())
        self.filt_cut = filter_cut
        params = self.frame_params[filter_cut]
        
        zp_mean = np.mean(params['zp_median'])
        self.flux_scales = 10**((params['zp_median'] - zp_mean) / -2.5)
        self.weights = 1.0 / self.flux_scales / params['sky_med']
        self.filter_paths = self.paths[filter_cut]
        self.filt_is_good = self.is_good[filter_cut]
        self.filter_params = params
            
        # setup base header
        h = utils.load_path_or_header(self.filter_paths[0])
        wcs = WCS(h)
        header = fits.Header()
        for k in self._base_header_keys:
            header.set(*h.cards[k])
        header['FILTER'] = get_filter_name(h)
        for c in wcs.to_header(True).cards:
            header.set(*c)
        ra_str, dec_str = utils.get_header_radec(h)
        header['RADEC'] = f'{ra_str},{dec_str}'
        comment = 'Average of median ZP from all frames'
        header.set('REFZP', zp_mean, comment)
        comment = 'Average of airmass from all frames'
        airmass_mean = np.mean(params['airmass'])
        header.set('AIRMASS', airmass_mean, comment)
        self.header = header

        if not utils.is_list_like(nseg):
            nseg = [nseg, nseg]
        self.grid = grid_image(self.shape, *nseg)

    def reset(self):
        self.exptime_eff = 0
        self.coadd = np.zeros(self.shape)
        self.exptime_map = np.zeros(self.shape)
        self.weight_map = np.zeros(self.shape)
        
    def fetch_section(self, label, has_weights=False):
        images = []
        s_ = self.grid.slices[label - 1]
        for i, p in enumerate(self.filter_paths[self.filt_is_good]):
            _img = utils.load_path_or_pixels(p)[s_]
            exptime = self.filter_params['exptime'][i]
            self.exptime_map[s_][_img != 0] += exptime
            w = self.weights[i] if has_weights else 1.0
            self.weight_map[s_][_img != 0] += w
            images.append(_img)
        images = np.array(images)
        return images, s_
            
    def write(self, out_path, coadd=True, weight=True, 
              exptime=True, prefix=None, suffix=None):
        assert self.exptime_eff > 0, 'no coadd data found'
        prefix = '' if prefix is None else prefix + '_'
        prefix = f'{prefix}coadd_{self.coadd_type}_{self.bandpass.lower()}'
        prefix = os.path.join(out_path, prefix)
        suffix = '' if (suffix is None or suffix == '') else f'_{suffix}'

        # TODO: is there a better place to cast pixel arrays to final dtype than here?
        if coadd:
            # replace empty pixels with NaNs, if possible
            empty_pixel_mask = np.zeros(self.shape)
            if weight:
                empty_pixel_mask = self.weight_map == 0
            elif exptime:
                empty_pixel_mask = self.exptime_map == 0

            coadd_nan = np.copy(self.coadd)
            coadd_nan[empty_pixel_mask] = np.nan

            # cast coadds to 32-bit float
            self.coadd_fn = prefix + f'{suffix}.fits'
            utils.write_pixels(self.coadd_fn, 
                               np.array(coadd_nan, dtype='f4'),
                               self.header)
        if weight:
            # cast weight maps to unsigned 16-bit int, max 65535 frames in a coadd
            self.weight_map_fn = prefix + f'_weight{suffix}.fits' 
            utils.write_pixels(self.weight_map_fn, 
                               np.array(self.weight_map, dtype='u2'),
                               self.header)
        if exptime:
            # cast exptime maps to unsigned 32-bit int, max 4294967295 seconds or 136 years
            self.exptime_map_fn = prefix + f'_exptime{suffix}.fits'
            utils.write_pixels(self.exptime_map_fn, 
                               np.array(self.exptime_map, dtype='u4'),
                               self.header)        

    def stack_images(self, bandpass, **kwargs):
        raise NotImplementedError()



class MedianCoadder(BaseCoadder):
    
    def stack_images(self, bandpass, nseg=[10, None], out_path=None, **kwargs):
        self.reset()
        self._setup(bandpass, 'median', nseg)
        for label in tqdm(self.grid.labels):
            images, s_ = self.fetch_section(label, False)
            images = np.ma.masked_where(images==0, images)
            images = images * self.flux_scales[:, None, None]
            self.coadd[s_] = np.ma.median(images, axis=0).data 
        back_val = self.filter_params['sky_med'] * self.flux_scales
        back_val = round(np.median(back_val), 3)
        self.coadd[self.coadd != 0] += back_val # add sky back  
        self.exptime_eff = self.frame_params[self.filt_cut]['exptime'].sum()
        card = 'EXP_EFF', self.exptime_eff, 'effective exposure time [s]'
        self.header.set(*card)
        card = 'BACKVAL', back_val, 'Background value added back to coadd'
        self.header.set(*card)
        self.header['COMBTYPE'] = 'MEDIAN'
        if out_path is not None:
            self.write(out_path, **kwargs)
            
    
class WeightedAverageCoadder(BaseCoadder):

    
    def _check_median(self, median_coadd):
        if median_coadd is None:
            raise Exception('You need a median image to make an average coadd')
        elif utils.is_list_like(median_coadd):
            pixels, header = load_pixels_and_header(*median_coadd)
        elif type(median_coadd) == str:
            pixels, header = load_pixels_and_header(median_coadd)
        else:
            msg = f'{type(median_coadd)} is not a valid median coadd type'
            raise Exception(msg)
        self.median_coadd = pixels
        self.median_header = header
        
    def _check_bpm_path(self, bpm_path, bpm_thresh, force, 
                        median_coadd, nproc):

        if utils.is_list_like(bpm_path):
            bpm_files = bpm_path
            return np.array(bpm_files)
        
        bpm_files = []
        for p in self.filter_paths:
            fn = os.path.basename(p).replace('.fits', '_bpm.fits')
            fn = os.path.join(bpm_path, fn)
            bpm_files.append(fn)
            
        exists = [os.path.isfile(fn) for fn in bpm_files]
        if np.sum(exists) == len(self.filter_paths):
            logger.warning('All BPMs appear to exist --> will reuse')
            return np.array(bpm_files)
        elif force:
            logger.warning('All BPMs exist but force=True --> making again')

        self._check_median(median_coadd)
        
        logger.info(f'Creating {len(self.filter_paths)} bad pixel masks.')

        if nproc == 1:
            med_pix = self.median_coadd
            loop_func = _bad_pixel_mask
            initializer = None
            initargs = None
        else:
            # setup median coadd to share between processors
            raw_array = RawArray('d', self.median_coadd.size)
            med_pix = np.frombuffer(raw_array).reshape(self.median_coadd.shape)
            np.copyto(med_pix, self.median_coadd)
            loop_func = _bad_pixel_mask_mp
            initializer = _bpm_mp_init
            initargs = (raw_array, self.median_coadd.shape)

        loop_vars = list(zip(self.filter_paths, self.flux_scales))

        utils.multiproc.thread_it(
            loop_func, 
            loop_vars,
            nproc = nproc,
            name = 'BPMs',
            initializer = initializer,
            initargs = initargs,
            thresh = bpm_thresh,
            bpm_path = bpm_path,
            back_val = self.median_header['BACKVAL'],
            med_pix = med_pix
        )
        
        return np.array(bpm_files)

    def check_bpm_masked_fraction(self, bpm_files, frac_min=0.001, 
                                  frac_max=0.1):
        is_good = np.ones_like(bpm_files, dtype=bool)
        for i, fn in enumerate(bpm_files):
            reject_frac = utils.load_path_or_header(fn)['BPFRAC']
            if (reject_frac > frac_max) or (reject_frac < frac_min):
                is_good[i] = False
        return is_good

    def fetch_bpm_section(self, label, paths):
        bpms = []
        s_ = self.grid.slices[label - 1]
        for i, p in enumerate(paths):
            _bpm = utils.load_path_or_pixels(p)[s_]
            bpms.append(_bpm)
        bpms = np.array(bpms)
        return bpms, s_

    def stack_images(self, bandpass, median_coadd=None, nseg=[10, None],
                     nproc=1, bpm_thresh=2.5, bpm_path='/tmp', clean=True, 
                     force=False, out_path=None, add_back_sky=True, **kwargs):        
        
        self._setup(bandpass, 'average', nseg)
        self.bpm_files = self._check_bpm_path(bpm_path, bpm_thresh, 
                                              force, median_coadd, nproc)
        self.filt_is_good = self.check_bpm_masked_fraction(self.bpm_files)
        self.bpm_files = self.bpm_files[self.filt_is_good]

        logger.info('Creating weighted average coadd')
        for label in tqdm(self.grid.labels):
            images, s_ = self.fetch_section(label, True)
            bpms, _ = self.fetch_bpm_section(label, self.bpm_files)
            images = np.ma.masked_where((images==0) | (bpms>0), images)
            w = self.weights[self.filt_is_good]
            self.coadd[s_] += np.ma.average(images, axis=0, weights=w).data 

        params = self.frame_params[self.filt_cut]

        if add_back_sky:
            w = self.weights[self.filt_is_good]
            back_vals = params[self.filt_is_good]['sky_med']
            back_val = np.average(back_vals, weights=w)
            self.coadd = self.coadd + back_val
            card = 'BACKVAL', back_val, 'Background value added back to coadd'
            self.header.set(*card)
        else:
            self.header['BACKVAL'] = 'none'

        t_exp = params[self.filt_is_good]['exptime']
        self.exptime_eff = t_exp.sum()
        card = 'EXP_EFF', self.exptime_eff, 'effective exposure time [s]'
        self.header.set(*card)
        self.header['COMBTYPE'] = 'AVERAGE'
        self.header['NFRAMES'] = self.filt_is_good.sum()
        self.header['NREJECT'] = (~self.filt_is_good).sum()

        if clean:
            for fn in self.bpm_files:
                logger.debug(f'Deleting {len(self.bpm_files)} bpm files')
                os.remove(fn)
        if out_path is not None:
            self.write(out_path, **kwargs)
