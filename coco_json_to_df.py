
## TODO -- https://github.com/laclouis5/globox
## SO CODE USED -- https://stackoverflow.com/questions/62251509/coco-json-file-to-csv-format-path-to-image-jpg-x1-y1-x2-y2-class-name

import pandas as pd
import json

path_validate_json = "/home/dhankar/temp/11_22/a___own_git_up/detect2/Obj_Detect_Detectron2/coco_val_images_2017/coco_train_2017/annotations/instances_val2017.json"


def convert_coco_json_to_csv(filename):
    """
    SO CODE -- https://stackoverflow.com/questions/62251509/coco-json-file-to-csv-format-path-to-image-jpg-x1-y1-x2-y2-class-name 
    """
    try:
        
        # COCO2017/annotations/instances_val2017.json
        s = json.load(open(filename, 'r'))
        #out_file = filename[:-5] + '.csv'
        out_file = "TEST_ANNOS_1.csv"
        out = open(out_file, 'w')
        out.write('id,x1,y1,x2,y2,label\n')

        all_ids = []
        for im in s['images']:
            all_ids.append(im['id'])

        print("---len(all_ids---",len(all_ids)) #5k

        all_ids_ann = []
        for ann in s['annotations']:
            image_id = ann['image_id']
            all_ids_ann.append(image_id)
            x1 = ann['bbox'][0]
            x2 = ann['bbox'][0] + ann['bbox'][2]
            y1 = ann['bbox'][1]
            y2 = ann['bbox'][1] + ann['bbox'][3]
            label = ann['category_id']
            out.write('{},{},{},{},{},{}\n'.format(image_id, x1, y1, x2, y2, label))

        all_ids = set(all_ids)
        all_ids_ann = set(all_ids_ann)
        no_annotations = list(all_ids - all_ids_ann)
        # Output images without any annotations
        for image_id in no_annotations:
            out.write('{},{},{},{},{},{}\n'.format(image_id, -1, -1, -1, -1, -1))
        out.close()

        # Sort file by image id
        s1 = pd.read_csv(out_file)
        print("----s1.info----",s1.info())
        s1.sort_values('id', inplace=True)
        s1.to_csv(out_file, index=False)

    except Exception as err_convert_coco_json_to_csv:
            print('---err_convert_coco_json_to_csv---\n',err_convert_coco_json_to_csv)
            pass

convert_coco_json_to_csv(path_validate_json)
