'''

Configurations for training different models
_______


Change params import statement in 'train.py' to train different models - with different input sizes etc

'''

# biobank pancreas only
params5 = dict()
# params5['num_classes'] = 2
params5['batch_size'] = 2
params5['image_shape'] = (192, 160, 48)
params5['nb_epoch'] = 200
params5['learning_rate'] = 1e-4
params5['augment'] = True
params5['data_folder'] = './data'
params5['seg_name'] = 'seg'
params5['f_name'] = 'Pancreas-seg-BB-V-3.0.1-alexbagur'
params5['validation_proportion'] = 0.1
params5['verbose'] = True

print(params5['f_name'])
