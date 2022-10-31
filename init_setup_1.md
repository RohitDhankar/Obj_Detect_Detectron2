

### Am well aware the below CODE / TEXT is Not Python 

```python

$ wget http://images.cocodataset.org/zips/val2017.zip
--2022-10-29 19:02:02--  http://images.cocodataset.org/zips/val2017.zip
Resolving images.cocodataset.org (images.cocodataset.org)... 52.217.230.121
Connecting to images.cocodataset.org (images.cocodataset.org)|52.217.230.121|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 815585330 (778M) [application/zip]
Saving to: ‘val2017.zip’

val2017.zip                         100%[=================================================================>] 777.80M  4.46MB/s    in 4m 14s  

2022-10-29 19:06:17 (3.06 MB/s) - ‘val2017.zip’ saved [815585330/815585330]

```

```python
$ wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
--2022-10-29 20:09:35--  http://images.cocodataset.org/annotations/annotations_trainval2017.zip
Resolving images.cocodataset.org (images.cocodataset.org)... 54.231.172.169
Connecting to images.cocodataset.org (images.cocodataset.org)|54.231.172.169|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 252907541 (241M) [application/zip]
Saving to: ‘annotations_trainval2017.zip’

annotations_trainval2017.zip        100%[=================================================================>] 241.19M  1.23MB/s    in 1m 57s  

2022-10-29 20:11:33 (2.05 MB/s) - ‘annotations_trainval2017.zip’ saved [252907541/252907541]


```
~~~
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip === 6 GB -- NOT DONE 
wget http://images.cocodataset.org/zips/unlabeled2017.zip
~~~

```python

ls_anno_ids----
 [{'segmentation': [[237.12, 319.95, 233.87, 315.4, 231.27, 315.4, 231.92, 323.85, 235.17, 325.15, 234.52, 332.3, 239.72, 340.09, 242.96, 353.08, 230.62, 378.42, 224.77, 373.87, 232.57, 365.43, 231.27, 356.33, 215.03, 360.88, 202.69, 362.18, 197.49, 364.78, 195.54, 369.33, 200.74, 373.87, 203.99, 377.77, 198.14, 382.32, 194.24, 389.47, 177.35, 391.41, 163.71, 391.41, 159.16, 396.61, 137.72, 412.85, 130.57, 426.5, 124.08, 447.93, 124.08, 454.43, 126.03, 462.23, 127.33, 487.56, 140.32, 501.21, 159.81, 508.35, 171.5, 511.6, 183.85, 525.89, 189.04, 529.79, 186.44, 507.7, 205.93, 494.71, 232.57, 455.73, 232.57, 451.83, 243.61, 459.63, 244.26, 436.89, 242.32, 422.6, 229.97, 390.11, 233.87, 375.82, 244.91, 359.58, 246.86, 342.69, 241.02, 326.45, 242.96, 323.2, 242.96, 321.9, 240.37, 321.9]], 'area': 12526.784249999995, 'iscrowd': 0, 'image_id': 196610, 'bbox': [124.08, 315.4, 122.78, 214.39], 'category_id': 2, 'id': 125062}, {'segmentation': [[122.27, 471.26, 100.74, 456.26, 96.82, 417.11, 87.03, 415.15, 98.78, 405.37, 113.78, 387.1, 135.32, 375.35, 158.8, 373.4, 172.51, 379.27, 175.12, 376.01, 171.85, 365.57, 160.11, 358.39, 158.15, 352.52, 173.81, 351.21, 192.08, 351.21, 181.64, 361.0, 184.9, 377.96, 184.9, 377.96, 190.77, 381.88, 171.2, 383.84, 158.15, 393.62, 135.97, 420.37, 124.22, 445.17, 126.18, 471.26]], 'area': 3996.61315, 'iscrowd': 0, 'image_id': 196610, 'bbox': [87.03, 351.21, 105.05, 120.05], 'category_id': 2, 'id': 125921}, {'segmentation': [[119.64, 340.9, 139.82, 335.14, 138.38, 349.55, 144.14, 366.85, 147.03, 371.17, 131.17, 376.94, 105.23, 392.79, 90.81, 408.65, 90.81, 428.83, 93.69, 434.59, 85.05, 434.59, 76.4, 427.39, 72.07, 405.77, 70.63, 392.79, 80.72, 366.85, 96.58, 362.52, 113.87, 362.52, 129.73, 365.41, 126.85, 358.2, 121.08, 356.76, 116.76, 348.11, 116.76, 346.67], [171.53, 319.28, 194.59, 314.95, 206.13, 313.51, 221.98, 317.84, 224.86, 330.81, 221.98, 340.9, 221.98, 346.67, 230.63, 350.99, 243.6, 350.99, 243.6, 356.76, 223.42, 355.32, 217.66, 352.43, 207.57, 353.87, 198.92, 361.08, 191.71, 371.17, 187.39, 374.05, 184.5, 374.05, 190.27, 365.41, 200.36, 355.32, 198.92, 348.11, 187.39, 342.34, 183.06, 333.69, 178.74, 319.28]], 'area': 4458.7988, 'iscrowd': 0, 'image_id': 196610, 'bbox': [70.63, 313.51, 172.97, 121.08], 'category_id': 2, 'id': 240982}, {'segmentation': [[58.79, 384.31, 46.0, 373.12, 46.8, 349.12, 57.19, 331.52, 76.39, 327.53, 76.39, 318.73, 77.19, 312.33, 81.19, 304.33, 106.78, 296.33, 101.99, 309.93, 117.18, 304.33, 122.78, 296.33, 132.38, 272.34, 151.58, 281.93, 149.18, 300.33, 153.98, 305.13, 175.57, 295.53, 181.97, 305.93, 149.98, 319.53, 141.18, 332.32, 125.18, 337.92, 113.98, 325.93, 125.98, 313.93, 106.78, 315.53, 93.19, 323.53, 90.79, 334.72, 66.79, 350.72]], 'area': 3865.1400500000022, 'iscrowd': 0, 'image_id': 196610, 'bbox': [46.0, 272.34, 135.97, 111.97], 'category_id': 2, 'id': 241702}, {'segmentation': [[67.14, 391.46, 62.18, 378.24, 70.44, 352.63, 75.81, 349.74, 92.75, 340.24, 93.99, 331.57, 99.36, 322.9, 112.99, 316.7, 116.29, 316.7, 118.77, 319.18, 117.53, 325.37, 115.88, 335.29, 120.83, 338.18, 115.46, 348.09, 120.83, 357.59, 127.03, 361.72, 128.27, 365.44, 115.05, 361.72, 99.36, 360.89, 86.97, 367.5, 77.88, 378.24, 72.92, 385.67, 71.27, 392.7], [139.42, 341.48, 152.22, 326.2, 156.35, 317.53, 161.72, 310.92, 156.35, 302.66, 153.46, 296.46, 158.0, 294.4, 141.9, 288.2, 156.76, 291.09, 171.63, 296.88, 180.31, 301.83, 182.78, 306.38, 169.98, 300.18, 162.55, 296.05, 158.0, 297.7, 164.61, 309.27, 168.74, 318.35, 171.63, 321.66, 184.85, 313.4, 186.91, 316.29, 186.5, 325.37, 189.39, 333.22, 195.18, 346.44, 188.15, 350.57, 175.76, 350.57, 165.85, 351.39, 160.9, 350.15, 160.9, 341.07, 164.61, 327.03, 168.74, 322.07, 163.79, 319.59, 146.44, 357.18, 143.14, 360.89]], 'area': 3384.5235000000002, 'iscrowd': 0, 'image_id': 196610, 'bbox': [62.18, 288.2, 133.0, 104.5], 'category_id': 2, 'id': 1334365}]
[124.08, 315.4, 122.78, 214.39]
[87.03, 351.21, 105.05, 120.05]
[70.63, 313.51, 172.97, 121.08]
[46.0, 272.34, 135.97, 111.97]
[62.18, 288.2, 133.0, 104.5]
(env2_det2) dhankar@dhankar-1:~/.../Obj_Detect_Detectron2$ 
```
#


### INSTALLED Fiftyone -- for COCO Data 

```python 


Installing collected packages: sseclient-py, sortedcontainers, rfc3986, pprintpp, patool, ndjson, kaleido, glob2, xmltodict, wrapt, tzdata, toml, tifffile, threadpoolctl, tenacity, sniffio, sentinel, scipy, retrying, PyWavelets, python-multipart, pymongo, psutil, priority, opencv-python-headless, networkx, joblib, jmespath, Jinja2, imageio, hyperframe, hpack, h11, greenlet, graphql-core, fiftyone-db, dnspython, dill, dacite, backports.cached-property, argcomplete, aiofiles, wsproto, strawberry-graphql, scikit-learn, scikit-image, pytz-deprecation-shim, plotly, motor, mongoengine, h2, eventlet, Deprecated, botocore, anyio, tzlocal, starlette, s3transfer, hypercorn, httpcore, fiftyone-brain, voxel51-eta, sse-starlette, httpx, boto3, universal-analytics-python3, fiftyone
Successfully installed Deprecated-1.2.13 Jinja2-3.1.2 PyWavelets-1.4.1 aiofiles-22.1.0 anyio-3.6.2 argcomplete-2.0.0 backports.cached-property-1.0.2 boto3-1.25.4 botocore-1.28.4 dacite-1.6.0 dill-0.3.6 dnspython-2.2.1 eventlet-0.33.1 fiftyone-0.17.2 fiftyone-brain-0.9.1 fiftyone-db-0.3.0 glob2-0.7 graphql-core-3.1.7 greenlet-1.1.3.post0 h11-0.12.0 h2-4.1.0 hpack-4.0.0 httpcore-0.15.0 httpx-0.23.0 hypercorn-0.14.3 hyperframe-6.0.1 imageio-2.22.2 jmespath-1.0.1 joblib-1.2.0 kaleido-0.2.1 mongoengine-0.20.0 motor-2.5.1 ndjson-0.3.1 networkx-2.8.7 opencv-python-headless-4.6.0.66 patool-1.12 plotly-5.11.0 pprintpp-0.4.0 priority-2.0.0 psutil-5.9.3 pymongo-3.12.3 python-multipart-0.0.5 pytz-deprecation-shim-0.1.0.post0 retrying-1.3.3 rfc3986-1.5.0 s3transfer-0.6.0 scikit-image-0.19.3 scikit-learn-1.1.3 scipy-1.9.3 sentinel-0.3.0 sniffio-1.3.0 sortedcontainers-2.4.0 sse-starlette-0.10.3 sseclient-py-1.7.2 starlette-0.16.0 strawberry-graphql-0.96.0 tenacity-8.1.0 threadpoolctl-3.1.0 tifffile-2022.10.10 toml-0.10.2 tzdata-2022.5 tzlocal-4.2 universal-analytics-python3-1.1.1 voxel51-eta-0.8.1 wrapt-1.14.1 wsproto-1.2.0 xmltodict-0.13.0

```


```python
(env2_det2) dhankar@dhankar-1:~/.../Obj_Detect_Detectron2$ pip install setuptools==59.5.0
Collecting setuptools==59.5.0
  Downloading setuptools-59.5.0-py3-none-any.whl (952 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 952.4/952.4 kB 12.9 MB/s eta 0:00:00
Installing collected packages: setuptools
  Attempting uninstall: setuptools
    Found existing installation: setuptools 63.4.1
    Uninstalling setuptools-63.4.1:
      Successfully uninstalled setuptools-63.4.1
Successfully installed setuptools-59.5.0
(env2_det2) dhankar@dhankar-1:~/.../Obj_Detect_Detectron2$ python det2_2.py > term_det2_2__11_1_0150h_TRAIN.log
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 275.62it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 275.52it/s]

100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 272.90it/s]
Traceback (most recent call last):
  File "/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2/det2_2.py", line 498, in <module>
    launch(train_data_custom,num_gpu,num_machines=1, machine_rank=0, dist_url="auto",args=(),)
  File "/home/dhankar/temp/11_22/det2/detectron2/detectron2/engine/launch.py", line 82, in launch
    main_func(*args)
  File "/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2/det2_2.py", line 456, in train_data_custom
    train_coco_data.do_train(cfg_launch, model, resume=False) 
  File "/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2/det2_2.py", line 236, in do_train
    data_loader = build_detection_train_loader(cfg)
  File "/home/dhankar/temp/11_22/det2/detectron2/detectron2/config/config.py", line 207, in wrapped
    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
  File "/home/dhankar/temp/11_22/det2/detectron2/detectron2/config/config.py", line 245, in _get_args_from_config
    ret = from_config_func(*args, **kwargs)
  File "/home/dhankar/temp/11_22/det2/detectron2/detectron2/data/build.py", line 344, in _train_loader_from_config
    dataset = get_detection_dataset_dicts(
  File "/home/dhankar/temp/11_22/det2/detectron2/detectron2/data/build.py", line 274, in get_detection_dataset_dicts
    print_instances_class_histogram(dataset_dicts, class_names)
  File "/home/dhankar/temp/11_22/det2/detectron2/detectron2/data/build.py", line 180, in print_instances_class_histogram
    assert (
AssertionError: Got an invalid category_id=18 for a dataset of 4 classes
(env2_det2) dhankar@dhankar-1:~/.../Obj_Detect_Detectron2$ 

```

### I remember this ONE ---- 

```python

https://github.com/facebookresearch/detectron2/blob/c54429b60a64736c8b62002c5729eb818835f745/detectron2/data/build.py#L69

        "Removed {} images with no usable annotations. {} images left.".format(

```

### ERROR -- CUDA Out of memory 

```python

(env2_det2) dhankar@dhankar-1:~/.../Obj_Detect_Detectron2$ 
(env2_det2) dhankar@dhankar-1:~/.../Obj_Detect_Detectron2$ python det2_2.py > term_det2_2__11_1_0330h_TRAIN.log
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 271.78it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 274.46it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 273.69it/s]
Traceback (most recent call last):
  File "/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2/det2_2.py", line 522, in <module>
    launch(train_data_custom,num_gpu,num_machines=1, machine_rank=0, dist_url="auto",args=(),)
  File "/home/dhankar/temp/11_22/det2/detectron2/detectron2/engine/launch.py", line 82, in launch
    main_func(*args)
  File "/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2/det2_2.py", line 480, in train_data_custom
    train_coco_data.do_train(cfg_launch, model, resume=False) 
  File "/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2/det2_2.py", line 251, in do_train
    loss_dict = model(data_from_loader)  
  File "/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/dhankar/temp/11_22/det2/detectron2/detectron2/modeling/meta_arch/rcnn.py", line 158, in forward
    features = self.backbone(images.tensor)
  File "/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/dhankar/temp/11_22/det2/detectron2/detectron2/modeling/backbone/fpn.py", line 155, in forward
    prev_features = lateral_features + top_down_features
RuntimeError: CUDA out of memory. Tried to allocate 194.00 MiB (GPU 0; 3.82 GiB total capacity; 2.15 GiB already allocated; 109.88 MiB free; 2.22 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
(env2_det2) dhankar@dhankar-1:~/.../Obj_Detect_Detectron2$ 

```