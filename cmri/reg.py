"""Registration."""

import numpy as np
import numbers
import time
import SimpleITK as sitk

import cmri.utils as utils

def affine_registration_sitkimage(moving, static,
                        pipeline=None,
                        moving_mask=None,
                        static_mask=None,
                        verbose=False,
                        iterations=None):
    """Register a moving image to a static image using SimpleElastix.
       All parameters must be SimpleITK images.

    Parameters
    ----------
    moving : SimpleITK Image
    static : SimpleITK Image
    pipeline : list, sequence of strings
    moving_mask : SimpleITK Image (sitkUInt8)
    static_mask : SimpleITK Image (sitkUInt8)
    verbose : bool, default False
    iterations : int, default None

    Returns
    -------
    resampled, params: the registered image, parameters for Transformix

    """

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(static)
    elastixImageFilter.SetMovingImage(moving)

    if static_mask is not None:
        elastixImageFilter.SetFixedMask(static_mask)
    if moving_mask is not None:
        elastixImageFilter.SetMovingMask(moving_mask)

    # setup registration pipeline
    if pipeline in [None, []]:
        pipeline = ["rigid"]
    parameterMapVector = sitk.VectorOfParameterMap() 
    for pdx, pl in enumerate(pipeline):
        parameterMapVector.append(sitk.GetDefaultParameterMap(pl))
    elastixImageFilter.SetParameterMap(parameterMapVector)

    # NOTE: should the settings below be applied for each pipeline item?

    # useless for our 2D slice data...
    elastixImageFilter.SetParameter('UseDirectionCosines', 'false')

    # mask is region of interest
    elastixImageFilter.SetParameter('ErodeMask', 'false')

    # do not align by geometrical centers
    elastixImageFilter.SetParameter('AutomaticTransformInitialization', 'false')

    elastixImageFilter.SetParameter('NumberOfResolutions', '3')

    # it always seems to reach this... so not converging
    if iterations is not None:
        elastixImageFilter.SetParameter('MaximumNumberOfIterations', str(int(iterations)))

    if not(verbose):
        elastixImageFilter.LogToConsoleOff()

    elastixImageFilter.Execute()
    resampled = elastixImageFilter.GetResultImage()
    params = elastixImageFilter.GetTransformParameterMap()

    #import IPython
    #IPython.embed()

    return resampled, params


def affine_registration(moving, static,
                        pipeline=None,
                        moving_mask=None,
                        static_mask=None,
                        verbose=False,
                        iterations=None):
    """Register a moving image to a static image using SimpleElastix.

    Parameters
    ----------
    moving : NumPy array
    static : NumPy array
    pipeline : list, sequence of strings
    moving_mask : NumPy array (integer)
    static_mask : NumPy array (integer)
    verbose : bool, default False
    iterations : int, default None

    Returns
    -------
    resampled, params: the registered image, parameters for Transformix

    """

    moving = sitk.GetImageFromArray(moving)
    static = sitk.GetImageFromArray(static)

    if moving_mask is not None:
        moving_mask = sitk.GetImageFromArray(moving_mask)
        moving_mask = sitk.Cast(moving_mask, sitk.sitkUInt8)

    if static_mask is not None:
        static_mask = sitk.GetImageFromArray(static_mask)
        static_mask = sitk.Cast(static_mask, sitk.sitkUInt8)
    
    transformed, params = affine_registration_sitkimage(
        moving, static,
        pipeline=pipeline,
        moving_mask=moving_mask,
        static_mask=static_mask,
        verbose=verbose,
        iterations=iterations)

    transformed = sitk.GetArrayFromImage(transformed)

    return transformed, params


def affine_transform(image, params):
    """Apply transform to image.

    Parameters
    ----------
    image : NumPy array
    params : SimpleITK parameter map

    Returns
    -------
    transformed: the transformed image

    """

    image = sitk.GetImageFromArray(image)
    transformed = sitk.Transformix(image, params)
    transformed = sitk.GetArrayFromImage(transformed)

    return transformed


def register_series(series, ref, pipeline=None, denoise=True, static_mask=None, verbose=False, iterations=None):
    """Register a series to a reference image.

    Parameters
    ----------

    series : NumPy array 
    ref : int, or NumPy array
    pipeline : list
    denoise : bool
    static_mask : NumPy array
    verbose : bool, default False
    iterations : int, default None

    Returns
    -------

    """

    # for collecting results
    xformed = np.zeros(series.shape)

    # convert NumPy array into ITK image series
    series = sitk.GetImageFromArray(series, isVector=True)
    
    if static_mask is not None:
        static_mask = sitk.GetImageFromArray(static_mask)
        static_mask = sitk.Cast(static_mask, sitk.sitkUInt8)

    filt2D = sitk.VectorIndexSelectionCastImageFilter()
    num_acq = series.GetNumberOfComponentsPerPixel()

    if isinstance(ref, numbers.Number):
        ref_as_idx = ref
        filt2D.SetIndex(ref)
        ref = filt2D.Execute(series)
    else:
        ref_as_idx = False
        ref = sitk.GetImageFromArray(ref)

    # denoise ref image
    if denoise:
        denoiseFilter = sitk.PatchBasedDenoisingImageFilter()
        ref = denoiseFilter.Execute(ref)

    time_per_run = 0.0
    time_left = "??:??:??"
    for ii in range(num_acq):
        start_time = time.time()
        utils.progress_bar(ii, num_acq, prefix = '', suffix = ' ' + time_left, decimals = 0, length = 20, fill = '#')

        filt2D.SetIndex(ii)
        this_moving = filt2D.Execute(series)
        if isinstance(ref_as_idx, numbers.Number) and ii == ref_as_idx:
            # This is the reference! No need to move and the xform is I(4):
            xformed[..., ii] = sitk.GetArrayFromImage(this_moving)
        else:

            # denoise moving image
            if denoise:
                this_moving_denoised = denoiseFilter.Execute(this_moving)
            else:
                this_moving_denoised = this_moving

            transformed, params = affine_registration_sitkimage(
                this_moving_denoised, ref,
                pipeline=pipeline,
                static_mask=static_mask,
                verbose=verbose,
                iterations=iterations)

            # apply transformation to non-denoised image
            if denoise:
                transformed = sitk.Transformix(this_moving, params)

            xformed[..., ii] = sitk.GetArrayFromImage(transformed)

            time_per_run = time_per_run + (time.time() - start_time - time_per_run) / (ii + 1) 
            time_left = time.strftime('%H:%M:%S', time.gmtime((num_acq - ii - 1)*time_per_run))

    return xformed


def register_dwi_series(data, gtab, b0_ref=0, pipeline=None, denoise=True,
                        static_mask=None, verbose=False, iterations=None):
    """Register a DWI series to the mean of the B0 images in that series.

    Parameters
    ----------

    Returns
    -------

    """

    # First, register the b0s to one image and average
    if np.sum(gtab.b0s_mask) > 1:
        b0_img = data[..., gtab.b0s_mask]
        trans_b0 = register_series(b0_img, ref=b0_ref,
                                   pipeline=pipeline,
                                   denoise=denoise,
                                   static_mask=static_mask,
                                   verbose=verbose,
                                   iterations=iterations)
        ref_data = np.mean(trans_b0, -1)
    else:
        trans_b0 = data[..., gtab.b0s_mask]
        ref_data = np.squeeze(trans_b0, axis=-1)

    # Second, register all images to ref_data  
    xformed = register_series(data, ref_data,
                              pipeline=pipeline,
                              denoise=denoise,
                              static_mask=static_mask,
                              verbose=verbose,
                              iterations=iterations)

    return xformed 

