import torch
import copy
from ultralytics import YOLO

class newModule(torch.nn.Module):
	def __init__(self):
		super(newModule, self).__init__()
		self.f = -1
		self.i = 0
	def forward(self, x):
		x = torch.div(x, 255.).to(dtype=torch.float)
		x = torch.nn.functional.interpolate(x, (640, 640), mode='linear')
		return x

yolo_model = YOLO()

yolo_model_new = copy.deepcopy(yolo_model)
# create a new sequential with the new module added
new_module = newModule()
new_sequential = torch.nn.Sequential(new_module)
for module in yolo_model_new.model.model:
    module.i += 1
    if isinstance(module.f, list):
        new_f = []
        for each_f in module.f:
            if each_f == -1:
                new_f.append(each_f)
            else:
                new_f.append(each_f + 1)
        module.f = new_f

# add that to the original model
yolo_model_new.model.model = new_sequential.extend(yolo_model_new.model.model)
new_save = []
for each_idx in yolo_model_new.model.save:
	new_save.append(each_idx + 1)

# overwrite the original save attribute
yolo_model_new.model.save = new_save

yolo_model.export(format='tflite')
yolo_model_new.export(format='tflite', imgsz=(1280, 1280))
