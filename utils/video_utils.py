import cv2 as cv

def read_video(video_path):
    cap = cv.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    cap.release()

    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    
    h, w, _ = output_video_frames[0].shape
    
    out = cv.VideoWriter(output_video_path, fourcc, 24, (w, h))

    print(f'Saving {len(output_video_frames)} frames to {output_video_path}...')
    
    for frame in output_video_frames:
        out.write(frame)
    
    out.release()
    print('Video saved successfully!')