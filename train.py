""""

3d-Unet training
-------------

Alex Bagur

NOTES:

data folder must contain images and masks in separate folder names: 'imgs' and 'masks'


"""

import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter

from data import create_datasets, get_filenames
from models import UNet3D

from monai.config import print_config
# from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import compute_meandice

from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image


def save_model_metadata(params, train_ds, val_ds):
    print('Saving model metadata to ' + params["f_name"] + '_meta_data.json' + ' ...')
    model_meta_data = {'model_info': params,
                       'training_data': get_filenames(train_ds),
                       'validation_data': get_filenames(val_ds)}
    with open(f'./saved_models/{params["f_name"]}_meta_data.json', '+w') as f:
        json.dump(model_meta_data, f, indent=4)


def train():
    """

    :return:
    """
    print('Model training started')
    set_determinism(seed=0)

    epoch_num = params['nb_epoch']
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        writer.add_scalar("epoch_loss", epoch_loss, epoch + 1)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = model(val_inputs)
                    value = compute_meandice(
                        y_pred=val_outputs,
                        y=val_labels,
                        include_background=False,
                        to_onehot_y=True,
                        mutually_exclusive=True,
                    )
                    metric_count += len(value)
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model, os.path.join('saved_models', "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last validation subject as GIF in TensorBoard with the corresponding image, label and pred
                val_pred = torch.argmax(val_outputs, dim=1, keepdim=True)
                summary_img = torch.cat((val_inputs, val_labels, val_pred), dim=3)
                plot_2d_or_3d_image(summary_img, epoch + 1, writer, tag='last_val_subject')

        # Model checkpointing
        if (epoch + 1) % 20 == 0:
            torch.save(model, os.path.join('saved_models', params['f_name'] + '_' + str(epoch + 1) + '.pth'))

    print(f"train completed, best_metric: {best_metric:.4f}  at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == '__main__':
    # See README for steps before running
    # 1. change up-ec2 script to use deep learning ami
    # 2. up-ec2 -n liver-train -t p3.2xlarge
    # 3. conda activate tf_gpu
    # 4. run-ec2 ...
    # 5. conda activate tf_gpu
    # 6. pip install -r requirements-cloud.txt
    from params import params5 as params
    print_config()

    # Create model output directory if doesn't exist
    if not os.path.exists('saved_models/'):
        os.makedirs('saved_models/')

    # Create datasets
    train_ds, train_loader, val_ds, val_loader = create_datasets(root_dir=params['data_folder'],
                                                                 end_image_shape=params['image_shape'],
                                                                 batch_size=params['batch_size'],
                                                                 validation_proportion=params['validation_proportion'])

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D().to(device)

    # Create loss function and optimizer
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'])

    # Save model metadata to json file
    save_model_metadata(params, train_ds, val_ds)

    train()

    """
    Go to Terminal tab in PyCharm and run:
    >> tensorboard --logdir=runs
    Click on the output url http://localhost:6006/ 
    """