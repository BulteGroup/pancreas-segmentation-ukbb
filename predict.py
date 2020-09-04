''''

This script creates the segmentations from the models, you need to have the "params" coherent to what you used in
model training.

Arguments:

 -f path to the file you want to segment (nifti)
 -m path to the model file you want to use

'''

import nibabel as nib
import time
import torch

from data import Winsorized
from monai.transforms import (
    AddChanneld,
    Orientationd,
    Compose,
    SpatialPadd,
    LoadNiftid,
    NormalizeIntensityd,
    ScaleIntensityd,
    ToTensord,
    KeepLargestConnectedComponent,
    CenterSpatialCrop,
)

from params import params5 as params


def get_seg_transforms(end_seg_axcodes):
    seg_transforms = Compose(
        [
            LoadNiftid(keys=["image"], as_closest_canonical=False),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes=end_seg_axcodes),
        ]
    )
    return seg_transforms


def get_test_transforms(end_image_shape):
    test_transforms = Compose(
        [
            LoadNiftid(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Winsorized(keys=["image"]),
            NormalizeIntensityd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(keys=["image"], spatial_size=end_image_shape),
            ToTensord(keys=["image"]),
        ]
    )
    return test_transforms


def predict(file_name, model_path='', _params=params, output_name=None):
    print('Segmenting ' + file_name + ' ...')
    start = time.time()

    # Create test sample as tensor batch
    test_transforms = get_test_transforms(_params['image_shape'])
    test_file = [{"image": file_name}]
    test_batch_image = test_transforms(test_file)[0]["image"].unsqueeze(0)

    # Load model and inference
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    seg = model(test_batch_image.to(device))

    # Postprocessing: keep largest seg component
    seg = torch.argmax(seg, dim=1, keepdim=True).detach().cpu()
    keeplargest = KeepLargestConnectedComponent(applied_labels=1)
    seg = keeplargest(seg)[0]

    # Resize output to original image size
    # Need to bring to canonical because so will seg be
    img = nib.load(file_name)
    img_canon = nib.as_closest_canonical(img)
    crop = CenterSpatialCrop(img_canon.shape)
    seg = crop(seg)[0]

    # Save output seg in canonical orientation
    seg1 = nib.Nifti1Image(seg.numpy(), img_canon.affine, img_canon.header)
    nib.save(seg1, output_name)

    # Change output seg orientation to original image orientation
    seg_file = [{"image": output_name}]
    seg_transforms = get_seg_transforms(end_seg_axcodes=nib.aff2axcodes(img.affine))
    seg1 = seg_transforms(seg_file)

    # Save output seg with same orientation as original image orientation
    seg1 = nib.Nifti1Image(seg1[0]["image"][0], img.affine, img.header)
    nib.save(seg1, output_name)
    
    print('Segmentation saved to ' + output_name)
    end = time.time()
    print('√ (time taken: ', round(end - start, ndigits=4), 'seconds)')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='''

        3D U-Net segmentation 
        ----------------------

        ''')

    parser.add_argument(
        '--filename', '-f', help='MRI VIBE to segment')
    parser.add_argument(
        '--model', '-m', help='Trained model to use',
        default="./saved_models/{}.h5".format(params['f_name']))
    parser.add_argument(
        '--output', '-o', help='Output directory and filename',
        default=params['seg_name'])

    args = parser.parse_args()

    # Segmentation inference
    predict(args.filename, model_path=args.model, _params=params, output_name=args.output)

    """
    Example
    python predict.py --filename data/imgs/1010616.nii.gz --model saved_models/Pancreas-seg-BB-V-3.0.0-alexbagur_200.pth --output 1010616-seg.nii.gz
    """
