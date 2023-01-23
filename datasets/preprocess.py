import numpy as np
import cv2


def load_video_cv2(video, fps_ratio=1, all_frames=False):
    cv2.setNumThreads(3)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 144 or fps is None: 
        fps = 25
    
    fps *= fps_ratio
    frames = []
    count = 0
    skip_first = True
    while cap.isOpened():
        ret = cap.grab()
        if int(count % round(fps)) == 0 or all_frames:
            # sometimes the first frame is empty
            if skip_first and ret == False:
                skip_first=False
                continue
                
            ret, frame = cap.retrieve()
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        count += 1
    cap.release()
    return np.array(frames)


def resize_short(imgs, short_size):
    h, w = imgs[0].shape[:2]
    if w <= h:
        new_w = short_size
        new_h = h  * new_w/ float(w)
    else:
        new_h = short_size
        new_w = w * new_h / float(h)
    new_h = int(new_h)
    new_w = int(new_w)
    new_imgs = [cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for img in imgs]
    return new_imgs


def crop_imgs(imgs, out_size):
    h, w = imgs[0].shape[:2]
    if out_size >= h:
        y0, y1 = 0, h
    else:
        y0 = (h - out_size)//2
        y1 = y0 + out_size
    if out_size >= w:
        w0, w1 = 0, w
    else:
        w0 = (w - out_size)//2
        w1 = w0 + out_size

    new_imgs = [img[y0:y1, w0:w1, :].copy() for img in imgs]
    return new_imgs
    

class Preprocess():
    def __init__(self, clip_len=8, out_size=224, frame_interval=1 ):
        self.clip_len = clip_len
        self.fps_interval = frame_interval
        self.out_size = out_size

    def __call__(self, video_path):
        frames = self.load_video(video_path)
        frames = resize_short(frames, self.out_size)

        frames = crop_imgs(frames, self.out_size)
        frames = np.array(frames).reshape((-1, self.clip_len, self.out_size, self.out_size,3))
        frames = (frames / 255.0 - 0.5) / 0.5
        return frames
       
    def load_video(self, video_path):
        frames = load_video_cv2(video_path, self.fps_interval)
        n_batch = int( len(frames) / float(self.clip_len)+0.5 ) 
        n_batch = max(n_batch, 1)
        n_frames_expect = self.clip_len * n_batch
        frames = frames[:n_frames_expect]
        if len(frames) > 0 and  len(frames) < n_frames_expect:
            pad_frames = [np.zeros(frames[-1].shape).astype(np.uint8) for _ in range(n_frames_expect - len(frames))]
            frames = np.concatenate([frames, pad_frames], axis=0)
        if len(frames) == 0:
            print(f"error: {video_path}")
            raise ValueError
        return frames
