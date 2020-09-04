## Pancreas Segmentation in UK Biobank ##

Pancreas segmentation in UK Biobank [VIBE data](http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20202), using the [PyTorch](https://pytorch.org/) implementation of UNet from [Project MONAI](https://github.com/Project-MONAI/MONAI).

<p align="center">
    <img src="img/individualImage.gif" width="640"> <br />
    <em> Example VIBE (left), manual annotation (middle), model prediction (right) </em>
</p>

## Getting started ##

### Installation
Creating a conda environment is recommended.

`pip install requirements.txt`

### Predictions
You may download a trained model [here](https://unioxfordnexus-my.sharepoint.com/:u:/g/personal/ball5417_ox_ac_uk/ERDEFXPQQz1Bt-ByzAf3j7wBnEYswlTnBIVtyR0ziCIQzg?e=N5gnQM) and make segmentation predictions using `predict.py`. 

From the installed python environment, run:

`python predict.py --filename YYYYYYY.nii.gz --model trained_model.pth --output YYYYYYY-seg.nii.gz`

### Model training

Data needs to be organised in the file structure below. Run `data.py` to see dataset examples.

In order to train a model, optionally change the parameters in `params.py` and then run `train.py`.

```
├── data
│   ├── imgs
│   │   ├── AAAAAAA.nii.gz
│   │   ├── BBBBBBB.nii.gz
│   │   ├── CCCCCCC.nii.gz
│   │   ├── DDDDDDD.nii.gz
│   │   └── EEEEEEE.nii.gz
│   └── masks
│       ├── AAAAAAA-seg.nii.gz
│       ├── BBBBBBB-seg.nii.gz
│       ├── CCCCCCC-seg.nii.gz
│       ├── DDDDDDD-seg.nii.gz
│       └── EEEEEEE-seg.nii.gz
```

### Log training

[TensorBoard](https://www.tensorflow.org/tensorboard) may be used to log training, via

`tensorboard --logdir=runs`

### References
1. A. T. Bagur, G. Ridgway, J. McGonigle, S. M. Brady, and D. Bulte, “Pancreas Segmentation-Derived Biomarkers: Volume and Shape Metrics in the UK Biobank Imaging Study,” in *Communications in Computer and Information Science*, vol. 1248 CCIS, 2020, pp. 131–142. <br />
	[Paper](https://doi.org/10.1007/978-3-030-52791-4_11) <br />
	[Conference Presentation](https://www.youtube.com/watch?v=qqe2FV215ZY&feature=youtu.be)

