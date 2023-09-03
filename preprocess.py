import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import cv2
import os

class preprocess:
    def __init__(self,root_path,save_root,detector):
        self.root_folder=root_path
        self.save_path=save_root
        self.detector=detector
        
    def process_frame(self,image,i,save_path1):
        # saves the file
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(mp_image)
        bbox=detection_result.detections[0].bounding_box
        x,y = bbox.origin_x, bbox.origin_y
        x1,y1 = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        c_image=image[y:y1+15, x:x1]
        save_location=os.path.join(save_path1,f"{i}.jpg")
        cv2.imwrite(save_location, c_image)
    
    def process_video(self,video_path,savepath1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        i=0    
        while cap.isOpened():
            i+=1
            ret, frame = cap.read()         
            if not ret:
                break
            #self.process_frame(frame,i,savepath1)
            try:
                self.process_frame(frame,i,savepath1)
            except Exception as e:
                print(f"error in {savepath1} -- {e}")
                continue
            
        cap.release()
    
    
    def createdata(self):    
        speakers=os.listdir(self.root_folder)
        for s in tqdm(speakers):
            s_path=os.path.join(self.root_folder,s)
            for videos in tqdm(os.listdir(s_path)):
                faces_save_path=os.path.join(self.save_path,s,videos.split(".")[0])
                os.makedirs(faces_save_path,exist_ok=True)
                self.process_video(os.path.join(s_path,videos),faces_save_path)
            print("------------------------------------------------------")
            print(f"--------------------Done for Speaker - {s} -- Saved in path -{faces_save_path}------------------- ")
            print("------------------------------------------------------")
# ------------------------  Define the model--------------------

model_path = 'detector.tflite'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options,min_detection_confidence=0.5)
detector = vision.FaceDetector.create_from_options(options)

# ------------------------- Initialize paths ---------------------

root_folder="/home2/souvikg544/gridcorpus/video"
save_root="/ssd_scratch/cvit/souvikg544/gridcorpus/faces"

# ---------------------------- Call the functions ------------------

mklipnet=preprocess(root_folder,save_root,detector)
mklipnet.createdata()
