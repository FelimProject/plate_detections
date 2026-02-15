import cv2 as cv
from ultralytics import YOLO
import sys
sys.path.append('..')
from utils.stub_utils import save_stub_file , read_stub_path

class CarDetection:
    def __init__(self , model_path):
        self.model = YOLO(model_path)

    def detect_frames(self , frames , read_from_stub=False , stub_path=None):
        car_detections = []

        if read_from_stub and stub_path is not None:
            stub_data = read_stub_path(stub_path)
            if stub_data is not None:
                return stub_data
        
        for frame in frames:
            car_detections.append(self.detect_frame(frame))

        if stub_path is not None:
            save_stub_file(stub_path , car_detections)

        return car_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame , iou= 0.1 , conf= 0.30)[0]
        id_name_dict = results.names

        car_list = []
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            cls_id = int(box.cls[0].item())
            cls_name = id_name_dict[cls_id]

            if cls_name == 'car':
                car_list.append(result)

        return car_list
    
    def draw_bboxes(self , video_frames , car_detections):
        output_video_frames = []
        for frame , car_list in zip(video_frames , car_detections):
            for _ , bbox in enumerate(car_list):
                x1 , y1 , x2 , y2 = map(int ,bbox)
                cv.putText(frame , f'Car' , (x1 , y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv.rectangle(frame , (x1 , y1) , (x2 , y2) , (255,255,0) , 2)
            
            output_video_frames.append(frame)

        return output_video_frames