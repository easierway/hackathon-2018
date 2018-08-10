import os
from os.path import join as pjoin
from keras import optimizers
from wukong.computer_vision.TransferLearning import WuKongVisionModel

wukong_dir='/home/ec2-user/src/wukong'

train_data_dir = '/home/ec2-user/training-set/train/enhance/'
test_data_dir = '/home/ec2-user/training-set/train/enhance/'
work_dir = pjoin(wukong_dir, 'tmp')

# train a model with the default configuration
#model = WuKongVisionModel()
#model.train_for_new_task(work_dir, task_name, train_data_dir, test_data_dir)

# predict by the trained model
#ret = model.predict(os.path.join(test_data_dir, "cat", "cat.983.jpg"))
#print ret

# Improvement tuning
#org_model = WuKongVisionModel()
#org_model.load_weights(os.path.join(work_dir, 'cat_dog.combined_model_weightsacc0.84_val_acc0.94.best.hdf5'))
#org_model.train_for_improvement_task(work_dir, task_name, train_data_dir, test_data_dir,
#                             optimizer=optimizers.SGD(lr=1e-5, momentum=0.9))
# You can load the weights to the model
new_model = WuKongVisionModel(224, 224)
new_model.load_weights(os.path.join(work_dir, 'douyin_224.combined_model_weightsacc0.83_val_acc0.92.best.hdf5'))
ret = new_model.predict(os.path.join(test_data_dir, "5789.jpg"))
print ret
ret = new_model.predict('/home/ec2-user/training-set/train/origin/5789.jpg')
print ret
