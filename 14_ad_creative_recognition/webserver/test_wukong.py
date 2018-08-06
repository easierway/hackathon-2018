import os
import sys
from wukong.wukong.computer_vision.TransferLearning import WuKongVisionModel

sys.path.append("./wukong")


train_data_dir = r'wukong/samples/cat_dog/train/'
test_data_dir = r'wukong/samples/cat_dog/test'
work_dir = r'./tmp'
task_name = "cat_dog"

# train a model with the default configuration
model = WuKongVisionModel()
model.train_for_new_task(work_dir, task_name, train_data_dir, test_data_dir)

# predict by the trained model
ret = model.predict(os.path.join(test_data_dir, "cat", "cat.983.jpg"))
print ret
