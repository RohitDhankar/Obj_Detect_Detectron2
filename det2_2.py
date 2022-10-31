## Register a Data Set in the Detectron2 format 
# SOURCE -- https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html?highlight=bbox_mode#standard-dataset-dicts

"""
Chosen Option -->> Detectron2’s standard dataset dict
Detectron2’s standard dataset dict, described below. 
This will make it work with many other builtin features in detectron2, so it’s recommended 
to use it when it’s sufficient.
"""

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg as get_config
from detectron2.utils.visualizer import Visualizer , ColorMode
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

from datetime import datetime
dt_time_now = datetime.now()
dt_time_save = dt_time_now.strftime("_%m_%d_%Y_%H_%M_")

import pandas as pd
import json , cv2 , os
from tqdm import tqdm



class std_data_dicts():
    def __init__(self):
        self.get_config = get_config()
        print("---type-get_config-",type(get_config)) ##---type-get_config- <class 'function'>
        ## SOURCE -- https://detectron2.readthedocs.io/en/latest/modules/config.html

        self.get_config.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.get_config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        ## https://detectron2.readthedocs.io/en/latest/modules/checkpoint.html
        self.get_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.get_config.MODEL.DEVICE = "cuda"
        self.predictor = DefaultPredictor(self.get_config)
        
    
    def get_std_data_dict(self,init_anno_path,read_file_rows,chunk_idx):
        """
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
        try:

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
                print("----[INFO]---min_x,min_y-",min_x,min_y)
                coco_width = int(float_width)
                coco_height = int(float_height)
                coco_std_data_dict["file_name"] = image_local_path 
                coco_std_data_dict["height"] = img_height
                coco_std_data_dict["width"] = img_width
                coco_std_data_dict["image_id"] = str_image_id #Unique - str or int)
                coco_std_data_dict["categories"] = str_image_id #Unique - str or int)

                category_id = bbox_gt_label #EARLIER >> bbox_gt_label += 1 ...#COCO Format category_id -- starts with INDEX ==1 

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

    def register_data(self):
        """
        Args:

        Returns:

        """

    def get_gt_bbox_viz(self,testing_image_path,coco_image_forViz,coco_data_metaData,detectron2_output_dir,str_score_thresh_for_viz):
        """
        Args:

        Returns:
        
        """
        test_image = cv2.imread(testing_image_path)
        print("-[INFO-get_gt_bbox_viz]---Type(test_image",type(test_image))
        default_predictor = self.predictor(test_image)
        
        visualizer = Visualizer(test_image,metadata = MetadataCatalog.get(self.get_config.DATASETS.TRAIN[0]))#,
        #visualizer = Visualizer(coco_image_forViz,metadata = coco_data_metaData)#,instance_mode=ColorMode.IMAGE_BW)#, scale=2.5)
        #coco_img_viz = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        print("--[INFO-get_gt_bbox_viz]--type(default_predictor---\n",type(default_predictor))
        print("--[INFO-get_gt_bbox_viz]---type(default_predictor[instances] ---\n",type(default_predictor["instances"]))
        ## <class 'detectron2.structures.instances.Instances'>
        instance_obj = default_predictor["instances"]
        #print("---instance_obj---> 1 ",instance_obj.num_instances)
        #print("---instance_obj--->pred_classes ",type(instance_obj.pred_classes)) ##<class 'torch.Tensor'>
        print("--[INFO-get_gt_bbox_viz]---instance_obj--->pred_classes ",instance_obj.pred_classes)

        result_image = visualizer.draw_instance_predictions(default_predictor["instances"].to("cpu"))
        #save_coco_img = str(str_image_id)+"_"+str(str_score_thresh_for_viz)
        save_coco_img_path = detectron2_output_dir + "/_dir_img_gt_bbox_/"+str(dt_time_save)+"/"
        if not os.path.exists(save_coco_img_path):
            os.makedirs(save_coco_img_path)
        #cv2.imwrite(save_coco_img_path+str(save_coco_img)+"_.png",result_image.get_image())
        cv2.imwrite(save_coco_img_path+"test_gt_bbox_.png",result_image.get_image())


if __name__ == "__main__":
    detectron2_output_dir = "./output_dir/" #
    str_score_thresh_for_viz = 0.70
    coco_data_metaData = "TODO"
    coco_image_forViz = "TODO"
    testing_image_path = "./coco_val_images_2017/val2017/000000078823.jpg" 
    #./coco_val_images_2017/val2017/000000494869.jpg
    #./coco_val_images_2017/val2017/000000554002.jpg 


    obj_std_dicts = std_data_dicts()
    init_anno_path = "df_out_coco_urls.csv" ## dt_time_save
    read_file_rows = 100 #30000 # 30K
    chunk_idx = 0
    #ls_coco_std_data_dicts = obj_std_dicts.get_std_data_dict(init_anno_path,read_file_rows,chunk_idx)

    obj_std_dicts.get_gt_bbox_viz(testing_image_path,coco_image_forViz,coco_data_metaData,detectron2_output_dir,str_score_thresh_for_viz)


