import cv2 as cv
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

import sys
sys.path.append('..')
from utils.stub_utils import save_stub_file, read_stub_path

class LicensePlateDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    def detect_frames(self, frames, read_from_stub=False, read_text_stub=False, stub_path=None, text_stub_path=None):
        if read_from_stub and stub_path and read_text_stub and text_stub_path:
            pass

        all_bboxes = []
        all_ocr_raw = []
        all_ids = []

        for frame in frames:
            bboxes, ocr_info, ids = self.detect_frame(frame)
            all_bboxes.append(bboxes)
            all_ocr_raw.append(ocr_info)
            all_ids.append(ids)
       
        best_ocr_per_id = {} 

        for frame_ocr, frame_ids in zip(all_ocr_raw, all_ids):
            for ocr_item, t_id in zip(frame_ocr, frame_ids):
                if t_id is None: continue
                
                if t_id not in best_ocr_per_id or ocr_item['conf'] > best_ocr_per_id[t_id]['conf']:
                    best_ocr_per_id[t_id] = ocr_item

        final_plate_texts = []
        for frame_ocr, frame_ids in zip(all_ocr_raw, all_ids):
            frame_final_texts = []
            for ocr_item, t_id in zip(frame_ocr, frame_ids):
                if t_id in best_ocr_per_id and best_ocr_per_id[t_id]['conf'] > 0:
                    frame_final_texts.append(best_ocr_per_id[t_id]['text'])
                else:
                    frame_final_texts.append(ocr_item['text'])
            final_plate_texts.append(frame_final_texts)

        if stub_path: save_stub_file(stub_path, all_bboxes)
        if text_stub_path: save_stub_file(text_stub_path, final_plate_texts)

        return all_bboxes, final_plate_texts
        
    def detect_frame(self, frame):
        results = self.model.track(frame, iou=0.1, conf=0.30, persist=True)[0]
        id_name_dict = results.names

        license_list = []
        license_plate_list = []
        track_ids = [] 

        for box in results.boxes:
            track_id = int(box.id.item()) if box.id is not None else -1
            
            result = box.xyxy.tolist()[0]
            cls_id = int(box.cls[0].item())
            cls_name = id_name_dict[cls_id]

            if cls_name == 'License_Plate':
                license_list.append(result)
                track_ids.append(track_id) 

                x1, y1, x2, y2 = map(int, result)
                y1, y2 = max(0, y1), min(frame.shape[0], y2)
                x1, x2 = max(0, x1), min(frame.shape[1], x2)
                cropped_plate = frame[y1:y2, x1:x2]

                if cropped_plate.size == 0:
                    license_plate_list.append({'text': 'N/A', 'conf': 0.0})
                    continue

                resized = cv.resize(cropped_plate, (0,0), fx=2, fy=2, interpolation=cv.INTER_CUBIC)
                gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
                processed_img = cv.merge([gray, gray, gray])
                
                ocr_result = self.ocr.ocr(processed_img)

                final_data = {'text': 'N/A', 'conf': 0.0}
                if ocr_result and isinstance(ocr_result, list):
                    res_dict = ocr_result[0]
                    texts = res_dict.get('rec_texts', [])
                    scores = res_dict.get('rec_scores', [])
                    
                    if len(texts) > 0:
                        final_data = {'text': str(texts[0]).strip(), 'conf': float(scores[0])}
                
                license_plate_list.append(final_data)

        return license_list, license_plate_list, track_ids

    def draw_bboxes(self, video_frames, license_plate_detections, license_plate_text_list):
        output_video_frames = []

        for frame, plate_list, text_list in zip(video_frames, license_plate_detections, license_plate_text_list):
            for bbox, text in zip(plate_list, text_list):
                x1, y1, x2, y2 = map(int, bbox)
                cv.putText(frame, f'{text}', (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)

            output_video_frames.append(frame)

        return output_video_frames