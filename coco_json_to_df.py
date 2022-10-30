
## TODO -- https://github.com/laclouis5/globox
## SO CODE USED -- https://stackoverflow.com/questions/62251509/coco-json-file-to-csv-format-path-to-image-jpg-x1-y1-x2-y2-class-name

import pandas as pd
import json

path_validate_json = "./coco_val_images_2017/coco_train_2017/annotations/instances_val2017.json" ##/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2



def convert_coco_json_to_csv(filename):
    """
    SO CODE -- https://stackoverflow.com/questions/62251509/coco-json-file-to-csv-format-path-to-image-jpg-x1-y1-x2-y2-class-name 
    """
    try:
        path_validate_images = "./coco_val_images_2017/val2017/"

        df_no_annotations = pd.DataFrame()
        df_coco_urls = pd.DataFrame()
        df_coco_gt_bbox = pd.DataFrame()
        
        # COCO2017/annotations/instances_val2017.json
        json_val_input = json.load(open(filename, 'r'))
        #out_file = filename[:-5] + '.csv'
        out_file = "TEST_ANNOS_3.csv"
        out = open(out_file, 'w')
        out.write('id,x1,y1,x2,y2,label,other_id,bbox_area\n')

        ls_all_ids = []
        ls_coco_urls = []
        ls_images_file_name = []
        ls_images_local_path = []


        ls_image_id = []#str(ann['image_id'])
        ls_other_id = []#str(ann['id'])
        ls_bbox_area = []#str(ann['area'])
        ls_category_id = []#ann['category_id']

        ls_x1 = []
        ls_x2 = []
        ls_y1 = []
        ls_y2 = []


        for key_img in json_val_input['images']:
            ls_all_ids.append(str(key_img['id']))
            ls_coco_urls.append(str(key_img['coco_url']))
            ls_images_file_name.append(str(key_img['file_name']))
            images_local_path = str(path_validate_images) + str(key_img['file_name'])
            ls_images_local_path.append(images_local_path)

        print("---len(all_ids-,all_coco_urls--",len(ls_all_ids ),len(ls_coco_urls),len(ls_images_file_name)) #5k
        df_coco_urls["image_INT_id"] = ls_all_ids
        df_coco_urls["coco_remote_url"] = ls_coco_urls
        df_coco_urls["image_jpg_file_name"] = ls_images_file_name
        df_coco_urls["image_local_path"] = ls_images_local_path

        print("----df_coco_urls.info()----",df_coco_urls.info())
        df_coco_urls.to_csv("df_coco_urls_.csv")


        all_ids_ann = []
        for ann in json_val_input['annotations']:
            #image_id = str(ann['image_id'])
            ls_image_id.append(str(ann['image_id']))
            #other_id = str(ann['id'])
            ls_other_id.append(str(ann['id']))
            #bbox_area = str(ann['area'])
            ls_bbox_area.append(str(ann['area']))
            all_ids_ann.append(str(ann['image_id']))
            x1 = ann['bbox'][0]
            ls_x1.append(ann['bbox'][0]) ##X_TOP_LEFT
            x2 = ann['bbox'][0] + ann['bbox'][2]
            ls_x2.append(ann['bbox'][0] + ann['bbox'][2])
            y1 = ann['bbox'][1]
            ls_y1.append(ann['bbox'][1])
            y2 = ann['bbox'][1] + ann['bbox'][3]
            ls_y2.append(ann['bbox'][1] + ann['bbox'][3]) ## Y_BOTTOM_RIGHT
            label = ann['category_id']
            ls_category_id.append(str(ann['category_id']))
            out.write('{},{},{}, {},{}, {},{},{}\n'.format(str(ann['image_id']),x1, y1,  x2, y2, label, str(ann['id']),str(ann['area'])))
        out.close()
        df_coco_gt_bbox["ls_bbox_area"] = ls_bbox_area
        df_coco_gt_bbox["ls_other_id"] = ls_other_id
        df_coco_gt_bbox["image_INT_id"] = ls_image_id ## Merge DF on this -- image_INT_id
        df_coco_gt_bbox["category_id"] = ls_category_id

        df_coco_gt_bbox["X_TOP_LEFT_x1"] = ls_x1 
        df_coco_gt_bbox["ls_x2"] = ls_x2
        df_coco_gt_bbox["ls_y1"] = ls_y1
        df_coco_gt_bbox["Y_BOTTOM_RIGHT_y2"] = ls_y2

        print("----df_coco_gt_bbox.info()----",df_coco_gt_bbox.info())
        df_coco_gt_bbox.to_csv("df_coco_gt_bbox.csv")

        df_out_coco_urls = df_coco_gt_bbox.merge(df_coco_urls,left_on='image_INT_id',right_on="image_INT_id")
        print("----df_out_coco_urls.info()----",df_out_coco_urls.info())
        df_out_coco_urls.to_csv("df_out_coco_urls.csv")


        all_ids = set(ls_all_ids)
        all_ids_ann = set(all_ids_ann)
        no_annotations = list(all_ids - all_ids_ann)
        df_no_annotations["no_annotations"] = no_annotations

        print("----df_no_annotations.info()----",df_no_annotations.info())

        # Output images without any annotations
        # for image_id in no_annotations:
        #     out.write('{},{},{}, {},{}, {},{},{}\n'.format(image_id, -1, -1, -1, -1, -1,-1,-1))
        # out.close()

        # Sort file by image id
        # s1 = pd.read_csv(out_file)
        # print("----s1.info----",s1.info())
        # s1.sort_values('id', inplace=True)
        # s1.to_csv(out_file, index=False)

    except Exception as err_convert_coco_json_to_csv:
            print('---err_convert_coco_json_to_csv---\n',err_convert_coco_json_to_csv)
            pass

convert_coco_json_to_csv(path_validate_json)
