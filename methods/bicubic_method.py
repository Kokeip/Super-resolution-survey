import cv2

def bicubic_upscale(lr_img, scale=4):
    """Bicubic插值上采样"""
    return cv2.resize(
        lr_img, 
        (int(lr_img.shape[1]*scale), int(lr_img.shape[0]*scale)),
        interpolation=cv2.INTER_CUBIC
    )