import warnings
import albumentations as A

warnings.simplefilter("ignore")


# --------------------------------------------------------------------
# Helpful functions
# --------------------------------------------------------------------

def post_transform(image, **kwargs):
    if image.ndim == 3:
        return image.transpose(2, 0, 1).astype("float32")
    else:
        return image.astype("float32")


# --------------------------------------------------------------------
# Segmentation transforms
# --------------------------------------------------------------------

post_transform = A.Lambda(name="post_transform", image=post_transform, mask=post_transform)

# crop 512
train_transform_1 = A.Compose([
    A.RandomScale(scale_limit=0.3, p=0.5),
    A.RandomCrop(512, 512, p=1.),
    A.Flip(p=0.75),
])

valid_transform_1 = A.Compose([
])

test_transform_1 = A.Compose([
])


train_transform_102 = A.Compose([

    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=45, border_mode=0, value=0, p=0.7),
    A.PadIfNeeded(512, 512, border_mode=0, value=0, p=1.),
    A.Flip(p=0.5),
    A.RandomCrop(512, 512, p=1.),
    A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1),
])


train_transform_103 = A.Compose([

    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=5, border_mode=0, value=0, p=0.7),
    A.PadIfNeeded(512, 512, border_mode=0, value=0, p=1.),
    A.Flip(p=0.5),
    A.RandomCrop(512, 512, p=1.),
    A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1),
])


train_transform_104 = A.Compose([

    A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=45, border_mode=0, value=0, p=0.7),
    A.PadIfNeeded(512, 512, border_mode=0, value=0, p=1.),
    A.Flip(p=0.5),
    A.RandomCrop(480, 480, p=1.),
    A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1),
])


train_transform_color = A.Compose([

    # color transforms
    A.OneOf(
        [
            # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            # A.RandomGamma(gamma_limit=(70, 130), p=1),
            # A.ChannelShuffle(p=0.2),
            # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=5, val_shift_limit=10, p=1),
            # A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
        ],
        p=0.3,
    ),

    # noise transforms
    A.OneOf(
        [
            # A.GaussNoise(p=1),
            # A.MultiplicativeNoise(p=1),
            # A.IAASharpen(p=1),
            # A.GaussianBlur(p=1),
        ],
        p=0.3,
    ),
])


train_transform_color_1 = A.Compose([

    # color transforms
    A.OneOf(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.RandomGamma(gamma_limit=(70, 130), p=1),
            A.ChannelShuffle(p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=5, val_shift_limit=10, p=1),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
        ],
        p=0.3,
    ),

    # noise transforms
    A.OneOf(
        [
            # A.GaussNoise(p=1),
            # A.MultiplicativeNoise(p=1),
            # A.IAASharpen(p=1),
            # A.GaussianBlur(p=1),
        ],
        p=0.3,
    ),
])


train_transform_color_2 = A.Compose([

    # color transforms
    A.OneOf(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.RandomGamma(gamma_limit=(70, 130), p=1),
            A.ChannelShuffle(p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=5, val_shift_limit=10, p=1),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
        ],
        p=0.3,
    ),

    # noise transforms
    A.OneOf(
        [
            A.GaussNoise(p=1),
            A.MultiplicativeNoise(p=1),
            # A.IAASharpen(p=1),
            A.GaussianBlur(p=1),
        ],
        p=0.3,
    ),
])


train_transform_color_3 = A.Compose([

    # color transforms
    A.OneOf(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.RandomGamma(gamma_limit=(70, 130), p=1),
        ],
        p=0.3,
    ),
])


valid_transform_4 = valid_transform_1
test_transform_4 = test_transform_1