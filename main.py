from utils.video_utils import save_video, read_video
from detections.car_detection import CarDetection
from detections.license_plate_detection import LicensePlateDetection
import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
def main():
    input_video_path = 'input_videos/video3.mp4'
    output_video_path = 'output_videos/video3.avi'
    
    car_stub_path = 'tracker_stubs/car_detection.pkl'
    license_stub_path = 'tracker_stubs/license_plate_detection.pkl'
    license_text_stub_path = 'tracker_stubs/license_plate_text.pkl'

    car_detector_model = 'models/car_detection.pt'
    license_detector_model = 'models/license_plate_detection.pt'
    
    video_frames = read_video(input_video_path)
    
    car_detector = CarDetection(car_detector_model)
    license_detector = LicensePlateDetection(license_detector_model)
    
    car_detections = car_detector.detect_frames(
        video_frames, 
        read_from_stub=True, 
        stub_path=car_stub_path
    )
    
    license_detections , license_plate_texts = license_detector.detect_frames(
        video_frames, 
        read_from_stub=True, 
        read_text_stub=True,
        text_stub_path=license_text_stub_path,
        stub_path=license_stub_path
    )
    
    output_video_frames = car_detector.draw_bboxes(
        video_frames, 
        car_detections
    )

    output_video_frames = license_detector.draw_bboxes(
        output_video_frames, 
        license_detections , 
        license_plate_texts
    )
    
    save_video(output_video_frames, output_video_path)

if __name__ == '__main__':
    main()
