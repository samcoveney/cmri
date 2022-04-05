"""Registration."""

import numpy as np
import SimpleITK as sitk


def affine_registration_sitkimage(moving, static,
                        pipeline=None,
                        moving_mask=None,
                        static_mask=None,
                        verbose=False):
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
    for pdx in pipeline:
        parameterMapVector.append(sitk.GetDefaultParameterMap(pdx))
    elastixImageFilter.SetParameterMap(parameterMapVector)

    # useless for our 2D slice data...
    elastixImageFilter.SetParameter('UseDirectionCosines', 'false')

    if not(verbose):
        elastixImageFilter.LogToConsoleOff()

    elastixImageFilter.Execute()
    resampled = elastixImageFilter.GetResultImage()
    params = elastixImageFilter.GetTransformParameterMap()

    return resampled, params


def affine_registration(moving, static,
                        pipeline=None,
                        moving_mask=None,
                        static_mask=None,
                        verbose=False):
    """Register a moving image to a static image using SimpleElastix.
       All parameters must be SimpleITK images.

    Parameters
    ----------
    moving : NumPy array
    static : NumPy array
    pipeline : list, sequence of strings
    moving_mask : NumPy array (integer)
    static_mask : NumPy array (integer)
    verbose : bool, default False

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
        verbose=verbose)

    return transformed
