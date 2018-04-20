"""
Requires skmm

pip install scikit-fmm
"""


def image_to_signed_distance(image):
    import skfmm
    return skfmm.distance(image*2 - 1) / max(*image.shape)
