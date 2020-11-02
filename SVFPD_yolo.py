# from yolov4.tf import YOLOv4
from myyolov4 import myyolov4
from FilesAccumulator import FilesAccumulator
from video_cv import SceneDatasetVideo, get_dataset
from tqdm import tqdm
import numpy as np
import cv2, json, os

def get_video_list(folder_name, varaint):
    folder = FilesAccumulator(folder_name)
    if varaint == 'c':
        return folder.find([".avi", 'mkv'], excludes=['variants_exceed_bw', 'variants_random'])
    elif varaint == 'r':
        return folder.find([".avi"], excludes=['variants_exceed_bw', 'variants'])
    elif varaint == 'e':
        return folder.find([".avi"], excludes=['variants_random',  'variants'])
    else:
        return []

def get_bbs(yolo, candidates, size):
    iou_threshold = 0.3,
    score_threshold = 0.25
    pred_bboxes = yolo.candidates_to_pred_bboxes(
        candidates[0].numpy(),
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    pred_bboxes = yolo.fit_pred_bboxes_to_original(pred_bboxes, size)
    return pred_bboxes

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()


    parser.add_argument("-f", "--file", dest="file",
                        help="specify name of the file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output",
                        help="specify name of the output", metavar="OUTPUT")



    args = parser.parse_args()

    input_file = args.file

    output_folder = input_file.rsplit("/", 1)[0]

    if args.output:
        output_file = args.output
    else:
        output_file = output_folder + "/yolo.json"

    video = input_file


    yolo = myyolov4()
    yolo.classes = "coco.names"
    result_detections = []

    video_sdv = SceneDatasetVideo(video)
    actual_size = video_sdv.get_frame_size()

    size = (416, 416)
    yolo.input_size = size

    dataloader_test = get_dataset(video, resizeTo=size, batch_size=32)

    yolo.make_model()
    yolo.load_weights("yolov4.weights", weights_type="yolo")
    ids = []
    predictions = []
    try:
        for data in tqdm(dataloader_test):
            if data == None:
                break
            inputs, img_ids = data
            inputs = np.swapaxes(np.swapaxes(inputs.numpy(), 1, 3), 1, 2)
            prediction = yolo.predict(inputs)

            for bbs, id in zip(predictions, img_ids):
                for bb in bbs:
                    k = bb
                    tmp = {}
                    tmp['image_id'] = int(id.numpy() + 1)
                    tmp['category_id'] = int(bb[4])
                    tl_x = k[0] - k[2] / 2 if k[0] - k[2] / 2 >= 0 else 0
                    tl_y = k[1] - k[3] / 2 if k[1] - k[3] / 2 >= 0 else 0

                    br_x = k[0] + k[2] / 2 if k[0] + k[2] / 2 <= 1 else 1
                    br_y = k[1] + k[3] / 2 if k[1] + k[3] / 2 <= 1 else 1
                    # tmp['bbox'] = [(int((k[0]-k[2]/2)*actual_size[0]), int((k[1]-k[3]/2)*actual_size[1])), (int((k[0]+k[2]/2)*actual_size[0]), int((k[1]+k[3]/2)*actual_size[1]))]
                    tmp['bbox'] = [int(tl_x * actual_size[0]), int(tl_y * actual_size[1]),
                                   int(br_x * actual_size[0]), int(br_y * actual_size[1])]
                    tmp['score'] = bb[5]
                    result_detections.append(tmp)

    except:
        pass

    try:
        os.makedirs(output_folder)
    except:
        print("Folder exists!")



    with open(output_file, 'w') as f:
        json.dump(result_detections, f)


