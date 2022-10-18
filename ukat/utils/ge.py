def scale_b1(data, flip_angle):
    """
    Scale the B1 map produced by GE scanners to be a percentage of the
    nominal flip angle.

    Its advisable to verify the velocity encode scale (dicom tag (0019,
    10E2)) is 299.97.

    Parameters
    ----------
    data : np.ndarray
        The B1 map to be scaled.
    flip_angle : float
        The flip angle of the sequence.

    Returns
    -------
    scaled_data : np.ndarray
        The scaled B1 map.
    """
    norm_factor = 10 / flip_angle
    b1_scaled = data * norm_factor
    return b1_scaled
