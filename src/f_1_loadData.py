import fiftyone as fo

# A name for the dataset
name = "coco_f1_1"

# The directory containing the dataset to import
dataset_dir = "/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2/coco_val_images_2017/f_1_data/"

# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset  # for example

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    name=name,
)

"""
Traceback (most recent call last):
  File "/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2/f_1_loadData.py", line 12, in <module>
    dataset = fo.Dataset.from_dir(
  File "/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/fiftyone/core/dataset.py", line 4407, in from_dir
    dataset.add_dir(
  File "/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/fiftyone/core/dataset.py", line 3193, in add_dir
    return self.add_importer(
  File "/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/fiftyone/core/dataset.py", line 3737, in add_importer
    return foud.import_samples(
  File "/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/fiftyone/utils/data/importers.py", line 119, in import_samples
    with dataset_importer:
  File "/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/fiftyone/utils/data/importers.py", line 790, in __enter__
    self.setup()
  File "/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/fiftyone/utils/coco.py", line 513, in setup
    image_paths_map = self._load_data_map(self.data_path, recursive=True)
  File "/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/fiftyone/utils/data/importers.py", line 748, in _load_data_map
    raise ValueError("Data directory '%s' does not exist" % data_path)
ValueError: Data directory '/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2/coco_val_images_2017/val2017/data/' does not exist
(env2_det2) dhankar@dhankar-1:~/.../Obj_Detect_Detectron2$ 

"""