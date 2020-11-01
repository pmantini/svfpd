from yolov4.tf import YOLOv4
from FilesAccumulator import FilesAccumulator
from video_cv import SceneDatasetVideo, get_dataset
from tqdm import tqdm
import numpy as np
import cv2, json, os

def get_video_list(folder_name):
    folder = FilesAccumulator(folder_name)
    return folder.find([".avi", "mkv"], excludes=[])

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

    parser.add_argument("-d", "--dir", dest="dir",
                        help="specify the name of the input directory", metavar="DIRECTORY")
    parser.add_argument("-o", "--output", dest="output",
                        help="specify the name of the output directory", metavar="OUTPUT")
    parser.add_argument("-m", "--model", dest="model",
                        help="specify model", metavar="MODEL")

    args = parser.parse_args()

    if args.dir:
        input_folder = args.dir
    else:
        input_folder = "/home/pmantini/Downloads/SVFPD"

    if args.output:
        output_folder = args.output
    else:
        output_folder = input_folder + "_results"

    model = args.model

    result_file = model + ".json"

    file_names = get_video_list(input_folder)

    if model == "yolo":
        counter = 0
        for video in file_names:
            print("%s/%s -- Processing %s" % (counter, len(file_names), video))
            yolo = YOLOv4()

            yolo.classes = "coco.names"
            result_detections = []
            video_folder = video.rsplit(input_folder, 1)[1].rsplit("/", 1)[0]

            video_sdv = SceneDatasetVideo(video)
            actual_size = video_sdv.get_frame_size()

            size = (32, 32)
            # size = (240 // 32 * 32, 320)
            # size = (416, 416)
            yolo.input_size = size

            dataloader_test = get_dataset(video, resizeTo=size, batch_size=32)

            yolo.make_model()
            yolo.load_weights("yolov4.weights", weights_type="yolo")

            try:
                for data in tqdm(dataloader_test):
                    if data == None:
                        break
                    inputs, img_ids = data
                    inputs = np.swapaxes(np.swapaxes(inputs.numpy(), 1, 3), 1, 2)

                    for inst, id in zip(inputs, img_ids):
                        bbs = yolo.predict(inst*255)
                        # image = np.uint8(inst*255)
                        # image = cv2.resize(image, actual_size)
                        for bb in bbs:
                            k = bb
                            tmp = {}
                            tmp['image_id'] = int(id.numpy() + 1)
                            tmp['category_id'] = int(bb[4])
                            tl_x = k[0]-k[2]/2 if k[0]-k[2]/2 >= 0 else 0
                            tl_y = k[1]-k[3]/2 if k[1]-k[3]/2 >= 0 else 0

                            br_x = k[0]+k[2]/2 if k[0]+k[2]/2 <= 1 else 1
                            br_y = k[1] + k[3]/2 if k[1]+k[3]/2 <= 1 else 1
                            # tmp['bbox'] = [(int((k[0]-k[2]/2)*actual_size[0]), int((k[1]-k[3]/2)*actual_size[1])), (int((k[0]+k[2]/2)*actual_size[0]), int((k[1]+k[3]/2)*actual_size[1]))]
                            tmp['bbox'] = [int(tl_x*actual_size[0]), int(tl_y*actual_size[1]), int(br_x*actual_size[0]), int(br_y*actual_size[1])]
                            tmp['score'] = bb[5]
                            result_detections.append(tmp)
                            # image = cv2.rectangle(cv2.UMat(image), (tmp['bbox'][0], tmp['bbox'][1]), (tmp['bbox'][2], tmp['bbox'][3]), (255, 0, 0), 1)

                        # cv2.imshow("Image", image)
                        # cv2.waitKey(0)
            except:
                pass


            results_folder = output_folder+ video_folder
            try:
                os.makedirs(results_folder)
            except:
                print("Folder exists!")
            output_file = os.path.join(results_folder, result_file)


            with open(output_file, 'w') as f:
                json.dump(result_detections, f)

            counter += 1
            # print('Detection complete')
                # image = np.uint8(inputs[0]*255)
                # for k in bbs:
                #     print(k)
                #     rectangle = [(int((k[0]-k[2]/2)*640), int((k[1]-k[3]/2)*480)), (int((k[0]+k[2]/2)*640), int((k[1]+k[3]/2)*480))]
                #     print(rectangle)
                #     image = cv2.rectangle(cv2.UMat(image), rectangle[0], rectangle[1], (255,0,0), 1)
                # cv2.imshow("Img", image)
                # cv2.waitKey(10)







