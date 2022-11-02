


# python det2_2.py > ./output_dir/terminal_logs/term_det2_2__11_2_0010h_TRAIN_100.log



## Local Path this file -- /Obj_Detect_Detectron2/det2_2.py
## Register a Data Set in the Detectron2 format 
# SOURCE -- https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html?highlight=bbox_mode#standard-dataset-dicts

"""
Chosen Option -->> Detectron2’s standard dataset dict
Detectron2’s standard dataset dict, described below. 
This will make it work with many other builtin features in detectron2, so it’s recommended 
to use it when it’s sufficient.
"""

# SOURCE for the LAUNCH method-- https://detectron2.readthedocs.io/en/latest/modules/engine.html#detectron2.engine.launch


import torch
torch.cuda.empty_cache()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg as get_config
from detectron2.utils.visualizer import Visualizer , ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch #for MULTIPLE GPU training 
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
import detectron2.utils.comm as comm
from detectron2.modeling import build_model
from torch.nn.parallel import DistributedDataParallel
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


from datetime import datetime
dt_time_now = datetime.now()
dt_time_save = dt_time_now.strftime("_%m_%d_%Y_%H_%M_")
detectron2_output_dir = './output_dir/_'+str(dt_time_save)+'_/' #saving the --model_final.pth 
coco_eval_output_dir = detectron2_output_dir+"_coco_eval_output_dir_/"


import pandas as pd
import json , cv2 , os
from tqdm import tqdm

class std_data_dicts():
    def __init__(self):
        self.get_config = get_config()
        print("---type-get_config-",type(get_config)) ##---type-get_config- <class 'function'>
        ## SOURCE -- https://detectron2.readthedocs.io/en/latest/modules/config.html
        
        #model_final_280758.pkl: 167MB [00:14, 11.3MB/s]   ##"COCO-Detection/faster_rcnn_R_50_FPN_3x": "137849458/model_final_280758.pkl",

        self.get_config.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.get_config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        ## https://detectron2.readthedocs.io/en/latest/modules/checkpoint.html
        self.get_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.get_config.MODEL.DEVICE = "cuda"
        self.predictor = DefaultPredictor(self.get_config)
        
    def register_data(self):
        """
        Args: ls_coco_std_data_dicts

        Returns:

        """
        DatasetCatalog.register("coco_eval_data_1",get_std_data_dict) 
        coco_std_data_dict_Registered = DatasetCatalog.get("coco_eval_data_1")
        print("--[INFO-register_data]-coco_std_data_dict_Registered--",len(coco_std_data_dict_Registered))
        
        coco_eval_data_Metadata = MetadataCatalog.get("coco_eval_data_1")
        print("--[INFO_register_data]-BEFORE_SET>> thing_classes--",coco_eval_data_Metadata)
        
        coco_classes = ["person","bicycle","car","motorcycle"]

        MetadataCatalog.get("coco_eval_data_1").set(thing_classes=coco_classes) 
        classes = MetadataCatalog.get("coco_eval_data_1").thing_classes
        print("--[INFO_register_data]---MetadataCatalog___len_Classes:",len(classes))
        print("--[INFO_register_data]---MetadataCatalog___NAME_Classes:",classes)

        coco_eval_data_Metadata = MetadataCatalog.get("coco_eval_data_1")
        print("--[INFO_register_data]-NOW_SET>> thing_classes--",coco_eval_data_Metadata)



    def get_className_from_val(self):
        """ 
        Args:

        Returns:

        """
        from pycocotools.coco import COCO
        path_validate = "./coco_val_images_2017/coco_train_2017/annotations/instances_val2017.json"
        coco_obj=COCO(path_validate)
        ls_classNames = ["person","bicycle","car","motorcycle"]
        for iter_n in range(len(ls_classNames)):
            coco_class_id = coco_obj.getCatIds(ls_classNames[iter_n])
            print("----COCO_CLASS--ID , for the COCO_CLASS_NAME--->> ",int(coco_class_id[0]),"__",ls_classNames[iter_n])


    def get_images_for_test(self,init_anno_path,coco_image_forViz,coco_data_metaData,detectron2_output_dir,str_score_thresh_for_viz):
        """
        Args:

        Returns:
        
        """
        df_annos_coco = pd.read_csv(init_anno_path)
        for iter_k in tqdm(range(len(df_annos_coco["image_local_path"]))):
            testing_image_path = df_annos_coco["image_local_path"][iter_k]
            if iter_k >=20:
                return
            else:
                self.get_gt_bbox_viz(testing_image_path,coco_image_forViz,coco_data_metaData,detectron2_output_dir,str_score_thresh_for_viz)


    def get_gt_bbox_viz(self,testing_image_path,coco_image_forViz,coco_data_metaData,detectron2_output_dir,str_score_thresh_for_viz):
        """
        Actually the PRED BBOX and not the GT BBOX -- But the PRED BBOX with the -- default_predictor
        This -- default_predictor -- is Trained on COCO Data and is being INPUT from the - MODEL ZOO - CONFIG FILE 
        ## self.get_config.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
          
        ##"./coco_val_images_2017/val2017/000000078823.jpg" 
        #./coco_val_images_2017/val2017/000000494869.jpg
        #./coco_val_images_2017/val2017/000000554002.jpg 

        Args:

        Returns:
        
        """
        test_image = cv2.imread(testing_image_path)
        image_file_local_1 = testing_image_path.split(".jpg")[0]
        image_file_local_2 = image_file_local_1.rsplit("/",1)[1]

        print("-[INFO-get_gt_bbox_viz]---image_file_local_2--",str(image_file_local_2))
        default_predictor = self.predictor(test_image)
        
        visualizer = Visualizer(test_image,metadata = MetadataCatalog.get(self.get_config.DATASETS.TRAIN[0]))#,
        #visualizer = Visualizer(coco_image_forViz,metadata = coco_data_metaData)#,instance_mode=ColorMode.IMAGE_BW)#, scale=2.5)
        #coco_img_viz = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        print("--[INFO-get_gt_bbox_viz]--type(default_predictor---\n",type(default_predictor))
        print("--[INFO-get_gt_bbox_viz]---type(default_predictor[instances] ---\n",type(default_predictor["instances"]))
        ## <class 'detectron2.structures.instances.Instances'>
        instance_obj = default_predictor["instances"]
        print("--[INFO-get_gt_bbox_viz]---instance_obj--->pred_classes ",instance_obj.pred_classes)

        if instance_obj[instance_obj.pred_classes >= 0]:

            print("--[INFO-get_gt_bbox_viz]--instance_obj--->pred_classes >=1 ",instance_obj.pred_classes)
            print("--[INFO-get_gt_bbox_viz]--instance_obj---> scores ",instance_obj.scores)

            ls_pred_classes = instance_obj.pred_classes.tolist()
            ls_pred_scores = instance_obj.scores.tolist()
            print("--[INFO-get_gt_bbox_viz]--instance_obj--->ls_pred_classes ",ls_pred_classes)
            print("--[INFO-get_gt_bbox_viz]--instance_obj--->ls_pred_scores ",ls_pred_scores)

            first_obj_pred_class = str(ls_pred_classes[0]) #
            first_obj_pred_score = str(ls_pred_scores[0]) 
            print("--[INFO-get_gt_bbox_viz]--instance_obj---> first_obj_pred_class ",first_obj_pred_class)
            # pred_coco_className = get_className_from_val(first_obj_pred_class)
            #print("---instance_obj--->ls_pred_classes[0]---pred_coco_className-- ",pred_coco_className)
            print("---instance_obj--->str(ls_pred_classes[0])--",str(ls_pred_classes[0]))


        result_image = visualizer.draw_instance_predictions(default_predictor["instances"].to("cpu"))
        save_coco_img = str(image_file_local_2)+"_"+str(str_score_thresh_for_viz)
        save_coco_img_path = detectron2_output_dir + "/_dir_img_gt_bbox_/"+str(dt_time_save)+"/"
        if not os.path.exists(save_coco_img_path):
            os.makedirs(save_coco_img_path)
        cv2.imwrite(save_coco_img_path+str(save_coco_img)+"_.png",result_image.get_image())
        #cv2.imwrite(save_coco_img_path+"test_gt_bbox_.png",result_image.get_image())


class train_coco_data():
    def __init__(self):
        pass

    def setup(custom_dataset_name,args):
        """
        Args: custom_dataset_name :- STR
              args :- dict

        Returns: Create configs and perform basic setups.
                 Code taken mostly- AS-IS from Detectron2 Documentation - for the LAUNCH method 

        """
        #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        print("--[INFO-setup]--args.opts--",args.opts)
        cfg = get_config()
        print("--[INFO-setup]--setup(custom_dataset_name,args-----\n",cfg)
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = (str(custom_dataset_name),)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 1 #4 # here INT Means INT COUNT of IMAGES in each BATCH 
        cfg.SOLVER.BASE_LR = 0.00025 # pick a good LR
        cfg.SOLVER.MAX_ITER = 1000   # 2k , 4 k etc . 
        cfg.SOLVER.STEPS = []        # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 #256 #128   == 256 X 4 = 1024
        ## TODO --[INFO_ORIGINAL_CODE]---NUM_ELE---pred_classes.numel()-- 1024 -- Calc Value 
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.70
        cfg.OUTPUT_DIR = './output_dir/_'+str(dt_time_save)+'_/' #saving the --model_final.pth 
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        print("--[INFO-setup]--detect2_output_dir---\n",cfg.OUTPUT_DIR)
        print("--[INFO-setup]--CONFIG going to build_model(cfg)__\n",cfg)
        cfg.freeze()
        default_setup(cfg, args) # ORIGINAL COMMENT--det2_GIT REPO-->> if you don't like any of the default setup, write your own setup code
        return cfg

    def do_train(cfg, model,resume=False):
        """ 
        """
        from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer

        print("--[INFO-Cls:train_coco_data--Meth:do_train]--INIT--CONFIG--",cfg)
        print("--[INFO-Cls:train_coco_data--Meth:do_train]---SUMMARY--model-",model) ##GeneralizedRCNN( (backbone): FPN(

        model.train()  
        optimizer = build_optimizer(cfg, model)
        scheduler = build_lr_scheduler(cfg, optimizer)

        checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler) 
        start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
        ## TODO -- RESUME_RELOAD -- https://github.com/facebookresearch/detectron2/issues/148
        max_iter = cfg.SOLVER.MAX_ITER
        periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
        writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
        # compared to "train_net.py", we do not support accurate timing and
        # precise BN here, because they are not trivial to implement in a small training loop
        data_loader = build_detection_train_loader(cfg)
        print("--[INFO-Cls:train_coco_data--Meth:do_train]--Type(data_loader----",type(data_loader))
        print("--[INFO-Cls:train_coco_data--Meth:do_train]-Starting training from iteration__",start_iter)

        with EventStorage(start_iter) as storage:
            for data_from_loader, iteration in zip(data_loader, range(start_iter, max_iter)):
            
                #type(data_from_loader)) ## <class 'list'>
                print("--[INFO-Cls:train_coco_data--Meth:do_train]--len(data_from_loader-",len(data_from_loader)) ## <class 'list'>
                for img_iter in range(len(data_from_loader)):
                    print("--[INFO-Cls:train_coco_data--Meth:do_train]--data_from_loader--IMAGE Names in Batch-",data_from_loader[img_iter]['file_name'])

                storage.iter = iteration
                print("-[INFO-Cls:train_coco_data--Meth:do_train]---iteration-",iteration)

                loss_dict = model(data_from_loader)  
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
                ):
                    #do_test(cfg, model)
                    # ORIGINAL COMMENT >> Compared to "train_net.py", the test results are not dumped to EventStorage
                    comm.synchronize()

                if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)

    
def get_std_data_dict():
    """
    This method - can Not be a Class Method --All ARGS - hardcoded below ...
    The -- DatasetCatalog.register -- Expects the --> get_std_data_dict -- to have no INPUT ARGS or INPUT PARAMS 
    Args:

        init_anno_path - Path to CSV File with COCO Style Annotations

        #  Data columns (total 12 columns):
        # #   Column               Non-Null Count  Dtype  
        # ---  ------               --------------  -----  
        # 0   Unnamed: 0           30 non-null     int64  
        # 1   ls_bbox_area         30 non-null     float64
        # 2   ls_other_id          30 non-null     int64  
        # 3   image_INT_id         30 non-null     int64  
        # 4   category_id          30 non-null     int64  
        # 5   X_TOP_LEFT_x1        30 non-null     float64
        # 6   ls_x2                30 non-null     float64
        # 7   ls_y1                30 non-null     float64
        # 8   Y_BOTTOM_RIGHT_y2    30 non-null     float64
        # 9   coco_remote_url      30 non-null     object 
        # 10  image_jpg_file_name  30 non-null     object 
        # 11  image_local_path     30 non-null     object 


    Returns:
        List of DICTS -- the std_data_dict's
    """

    """
    TODO -- 
      File "/home/dhankar/temp/11_22/det2/detectron2/detectron2/data/build.py", line 180, in print_instances_class_histogram
      AssertionError: Got an invalid category_id=18 for a dataset of 4 classes

    ----COCO_CLASS--ID , for the COCO_CLASS_NAME--->>  18 __ dog
    ----COCO_CLASS--ID , for the COCO_CLASS_NAME--->>  1 __ person
    ----COCO_CLASS--ID , for the COCO_CLASS_NAME--->>  2 __ bicycle
    ----COCO_CLASS--ID , for the COCO_CLASS_NAME--->>  4 __ motorcycle
    """


    init_anno_path = "./input_dir/df_out_coco_urls.csv" ## dt_time_save
    read_file_rows = 100 #30000 # 30K
    chunk_idx = 0

    try:

        dict_coco_classes = {
            #"dog":18,
            "person":0,
            "bicycle":1,
            "car":2,
            "motorcycle":3
        }

        ls_chunks = []
        ls_coco_std_data_dicts = []
        image_id = 0 
        annotation_id = 1 #TODO -- Not required by OFFICIAL DOCS - https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html?highlight=bbox_mode#standard-dataset-dicts
        
        for chunk in pd.read_csv(init_anno_path,chunksize=read_file_rows,iterator=True, low_memory=False):
            ls_chunks.append(chunk)
        df_annos_coco_chunks = ls_chunks[chunk_idx]
        print("--[INFO-get_std_data_dict]--df_annos_coco_chunks.info()-",df_annos_coco_chunks.info())

        for iter_k in tqdm(range(len(df_annos_coco_chunks["image_jpg_file_name"]))):
            annotations = [] 
            coco_std_data_dict = {}
            coco_img_name = str(df_annos_coco_chunks["image_jpg_file_name"][iter_k]) 
            print("-[INFO--get_std_data_dict]-coco_img_name->>\n",coco_img_name)
            image_local_path = str(df_annos_coco_chunks["image_local_path"][iter_k]) 
            print("-[INFO--get_std_data_dict]-image_local_path->>\n",image_local_path)
            bbox_gt_label = str(df_annos_coco_chunks["category_id"][iter_k]) 

            Y1 = float(df_annos_coco_chunks["ls_y1"][iter_k])
            Y2 = float(df_annos_coco_chunks["Y_BOTTOM_RIGHT_y2"][iter_k])  #Y_BOTTOM_RIGHT_y2  Y_BOTTOM_RIGHT -- lower-right y -- Y2
            X1 = float(df_annos_coco_chunks["X_TOP_LEFT_x1"][iter_k])  
            X2 = float(df_annos_coco_chunks["ls_x2"][iter_k])  

            image_id += 1 
            str_image_id = image_local_path + "_" + str(image_id) + "_" +str(bbox_gt_label) 
            img_init = cv2.imread(image_local_path) 
            print("-img_init.shape---",img_init.shape)
            img_height, img_width, channels = img_init.shape
            bbox_height = Y2 -Y1
            print("--[INFO]----bbox_height--",bbox_height) 
            bbox_width = X2 - X1 
            print("--[INFO]----bbox_width--",bbox_width) 
            xcenter = (X2 - X1 )/2  
            ycenter = (Y2 - Y1 )/2  
            print("--[INFO]----bbox_xcenter,---ycenter-",xcenter,ycenter) 
            yolo_width = bbox_width / img_width
            yolo_height = bbox_height / img_height
            print("--[INFO]----yolo_height--",yolo_height)

            # COCO Format Starts------------>>
            float_x_center = img_width * xcenter
            float_y_center = img_height * ycenter
            print("--[INFO]----float_x_center,float_y_center--",float_x_center,float_y_center)
            float_width = img_width * yolo_width
            float_height = img_height * yolo_height
            print("--[INFO]---float_width--,--float_height-",float_width,float_height)

            min_x = X1 #int(float_x_center - float_width / 2)
            min_y = Y1 #int(float_y_center - float_height / 2)
            print("--[INFO]---min_x,min_y-",min_x,min_y)
            coco_width = int(float_width)
            coco_height = int(float_height)

            coco_std_data_dict["file_name"] = image_local_path 
            coco_std_data_dict["height"] = img_height
            coco_std_data_dict["width"] = img_width
            coco_std_data_dict["image_id"] = str_image_id #Unique - str or int)
            coco_std_data_dict["categories"] = str_image_id #Unique - str or int)

            if int(bbox_gt_label) in dict_coco_classes.values():
                print("--[INFO]---bbox_gt_label--",bbox_gt_label)
                #class_coco_intIdx = bbox_gt_label 
                category_id = bbox_gt_label #EARLIER >> class_coco_intIdx += 1 ...#COCO Format category_id -- starts with INDEX ==1 

                coco_bbox = (float(min_x), float(min_y), float(coco_width), float(coco_height)) ##BoxMode.XYWH_ABS
                ## https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.BoxMode.XYWH_ABS
                coco_area = coco_width * coco_height
                # max_x = min_x + coco_width
                # max_y = min_y + coco_height

                annotation_id += 1 
                annotation = {
                    "id": annotation_id,
                    "image_id": str_image_id,
                    "bbox": coco_bbox,
                    "bbox_mode":1,
                    "area": coco_area,
                    # "iscrowd": 0,
                    "category_id": category_id
                    # "segmentation": seg,
                }
                annotations.append(annotation)
            coco_std_data_dict["annotations"] = annotations 
            print('-[INFO]---coco_std_data_dict--\n',coco_std_data_dict)
            ls_coco_std_data_dicts.append(coco_std_data_dict)
        print('---[INFO-coco_std_data_dict]--len(ls_dataset_dicts---\n',len(ls_coco_std_data_dicts))
        return ls_coco_std_data_dicts 

    except Exception as err_read_csv_eval_data:
        print('---err_read_csv_eval_data---\n',err_read_csv_eval_data)
        pass


def train_data_custom():

    """ 
    Args: None , No ARGS or Input PARAMS 
    
    Returns:
            CAUTION -->> for PARALELL GPU COMPUTE we will need to REGISTER Datasets every GPU Instance. 
    """
    from detectron2.engine import DefaultTrainer
    from torch.nn.parallel import DistributedDataParallel
    from detectron2.data import DatasetCatalog

    DatasetCatalog.register("coco_eval_data_2",get_std_data_dict)  
    coco_std_data_dict_Registered = DatasetCatalog.get("coco_eval_data_2") 
    print("--[INFO-train_data_custom]-coco_std_data_dict_Registered--",len(coco_std_data_dict_Registered))

    coco_classes = ["person","bicycle","car","motorcycle"]

    MetadataCatalog.get("coco_eval_data_2").set(thing_classes=coco_classes) 
    classes = MetadataCatalog.get("coco_eval_data_2").thing_classes
    print("--[INFO-train_data_custom]---MetadataCatalog___len_Classes:",len(classes))
    print("--[INFO-train_data_custom]---MetadataCatalog__names_of COCO Classes:",classes)

    coco_eval_data_Metadata = MetadataCatalog.get("coco_eval_data_2")
    print("--[INFO-train_data_custom]--NOW_SET>> thing_classes---:",coco_eval_data_Metadata)

    args_default_init = default_argument_parser().parse_args() #
    print("[INFO-train_data_custom]--args_default_init---:", type(args_default_init)) 
    print("[INFO-train_data_custom]--args_default_init--1-:", args_default_init) #
    #Namespace(config_file='', dist_url='tcp://127.0.0.1:50152', eval_only=False, machine_rank=0, num_gpus=1, num_machines=1, opts=[], resume=False)
    args_default_init.num_gpus = 1 # 4 if you have 4 or 6 or 8 GPU's 
    print("[INFO-train_data_custom]--args_default_init--2:", args_default_init)
    
    args= args_default_init

    custom_dataset_name = "coco_eval_data_2"
    cfg_launch = train_coco_data.setup(custom_dataset_name,args)
    model = build_model(cfg_launch)

    

    # distributed = comm.get_world_size() > 1
    # if distributed:
    #     model = DistributedDataParallel(
    #         model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
    #     )

    print("--[INFO-train_data_custom]--type(model\n",type(model)) #<class 'torch.nn.parallel.distributed.DistributedDataParallel'>
    print("--[INFO-train_data_custom]---SUMMARY__model\n",model)
    train_coco_data.do_train(cfg_launch, model, resume=False) 
    print("--[INFO-train_data_custom]----do_train(cfg, model, resume=False)_ENDS")   

    """
    https://detectron2.readthedocs.io/en/latest/modules/modeling.html#detectron2.modeling.build_model
    #https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/build.py
    ORIGINAL COMMENTS IN -->> build_model(
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    
    # <class 'detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN'>
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/rcnn.py
    # https://github.com/facebookresearch/detectron2/blob/5aeb252b194b93dc2879b4ac34bc51a31b5aee13/detectron2/modeling/meta_arch/rcnn.py#L25

    """


def get_eval_data_dict():
    """ """


def eval_validation_set(custom_dataset_name):
    """
    Args:

    Returns:

   
    # ORIGINAL COMMENT Detectron2 -- Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    
    For Running the EVALUATION / VALIDATION on the VALIDATION SET we are required to again - REGISTER the VALIDATION Dataset separately , from the TRAIN data. 
    As an aside - for TEST DATA runs - we dont need to REGISTER the data again .

    """

    dict_coco_classes = {
            #"dog":18,
            "person":0,
            "bicycle":1,
            "car":2,
            "motorcycle":3
        }

    eval_dataset_name = "coco_eval_data_3" 
    
    DatasetCatalog.register(eval_dataset_name,get_eval_data_dict) #eval dataset registry -- get_eval_data_dict
    coco_eval_data_dict_Registered = DatasetCatalog.get(eval_dataset_name) 
    len_registered_eval_dict = len(coco_eval_data_dict_Registered)
    print("--[INFO-eval_validation_set]--coco_eval_data_dict_Registered--",len_registered_eval_dict)

    coco_classes = ["person","bicycle","car","motorcycle"]
    MetadataCatalog.get(eval_dataset_name).set(thing_classes=coco_classes) 
    classes = MetadataCatalog.get(eval_dataset_name).thing_classes
    print("--[INFO-eval_validation_set]---MetadataCatalog___len_Classes:",len(classes))
    print("--[INFO-eval_validation_set]---MetadataCatalog___NAME_Classes:",classes)

    coco_eval_data_evalMetadata = MetadataCatalog.get(eval_dataset_name)
    print("--[INFO-eval_validation_set]--NOW_SET>> thing_classes---MetadataCatalog__coco_eval_data_evalMetadata:",coco_eval_data_evalMetadata)
    #os.chdir(detectron2_output_dir)
    eval_appch_metaData = MetadataCatalog.get(eval_dataset_name)
    print("-[INFO-eval_validation_set]-eval_appch_vids--MetadataCatalog__eval_appch_metaData:",eval_appch_metaData)

    cfg = get_config()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    #cfg.merge_from_file(model_zoo.get_config("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",trained=True))
    print("-[INFO-eval_validation_set]-cfg--initial merged from file___>>\n",cfg)
    
    cfg.OUTPUT_DIR = detectron2_output_dir # Global Var with HouR DIR Suffix
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,"model_final.pth")  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.70
    cfg.DATASETS.TEST = (str(eval_dataset_name),)   

    print("-[INFO-eval_validation_set]-cfg--APPENDED_TO___>>\n",cfg)
    predictor = DefaultPredictor(cfg)
 
    evaluator = COCOEvaluator(eval_dataset_name, output_dir=coco_eval_output_dir)
    eval_dataset_loader = build_detection_test_loader(cfg, eval_dataset_name)
    print("--[INFO-eval_validation_set]---COCOEvaluator_output--\n")
    print(inference_on_dataset(predictor.model,eval_dataset_loader,evaluator)) 
  
    if torch.cuda.is_available():
        predictor.model = predictor.model.to('cuda') ## cuda:2 >> RuntimeError: CUDA error: invalid device ordinal
        predictor.cfg.MODEL.DEVICE = 'cuda' #TODO -- Need to do the MULTIPLE GPU Eval 

    print("--[INFO-eval_validation_set]---type(predictor----",type(predictor))
    print("--[INFO-eval_validation_set]--type(predictor----",predictor)
    str_score_thresh_for_viz = str(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    print("--[INFO-eval_validation_set]--str_score_thresh_for_viz---",str_score_thresh_for_viz)
    
    image_temp_counter = 1 

    for iter_k in range(len_registered_eval_dict):
        eval_image = coco_eval_data_dict_Registered[iter_k]["file_name"]




if __name__ == "__main__":
    detectron2_output_dir = './output_dir/_'+str(dt_time_save)+'_/' #saving the --model_final.pth 
    str_score_thresh_for_viz = 0.70 ## ANYYTHING BELOW 70% AND IS IDENTIFIED AS A CATEGORY OF THE OBJECTS --- THAT IS ACTUALLY A FALSE POSITIVE 
    coco_data_metaData = "TODO"
    coco_image_forViz = "TODO"

    obj_std_dicts = std_data_dicts()
    init_anno_path = "./input_dir/df_out_coco_urls.csv" ## dt_time_save
    # read_file_rows = 100 #30000 # 30K
    # chunk_idx = 0

    ## Below called within -- register_data
    #ls_coco_std_data_dicts = obj_std_dicts.get_std_data_dict(init_anno_path,read_file_rows,chunk_idx)
    #ls_testing_image_paths = obj_std_dicts.get_images_for_test(init_anno_path,coco_image_forViz,coco_data_metaData,detectron2_output_dir,str_score_thresh_for_viz)
    
    obj_std_dicts.register_data()
    obj_std_dicts.get_className_from_val()

    
    #TODO -[11/1/22] --> Implement_LAUNCH_Method 
    # SOURCE for the LAUNCH method-- https://detectron2.readthedocs.io/en/latest/modules/engine.html#detectron2.engine.launch
    # https://github.com/facebookresearch/detectron2/blob/main/tools/plain_train_net.py
    
    num_gpu = 1
    launch(train_data_custom,num_gpu,num_machines=1, machine_rank=0, dist_url="auto",args=(),)
