from monai.networks.layers import Norm
from monai.networks.nets import UNet


def UNet3D():
    unet = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    return unet


if __name__ == '__main__':
    model = UNet3D()
    print(model)
