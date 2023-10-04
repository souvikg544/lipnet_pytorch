import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp1
import warnings
warnings.filterwarnings('ignore', message='Face blendshape model *')


class preprocess:
    def __init__(self,root_path,save_path_faces,save_path_landmarks):
        self.root_folder=root_path  # Path where the actul video is saved
        self.save_path_faces=save_path_faces    # path where faces images will get saved
        self.save_path_landmarks=save_path_landmarks #path where landmarks need to be saved

        
        self.landmark_model_path="face_landmarker.task"
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        self.options = self.FaceLandmarkerOptions(
        base_options=self.BaseOptions(model_asset_path=self.landmark_model_path),
        running_mode=self.VisionRunningMode.IMAGE,
        output_face_blendshapes=True
        )

    def get_landmark(self,c_image):
        with self.FaceLandmarker.create_from_options(self.options) as landmarker:
            face_landmarker_result = landmarker.detect(c_image)
        return face_landmarker_result

    def process_frames(self,im,k,save_path1):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=im)
        landmarker=self.get_landmark(mp_image)
        res=np.empty((0, 3), dtype=float)
        land=landmarker.face_landmarks[0]
            
        for j in range(0,len(land),1):
            i=land[j]            
            row=np.array([i.x,i.y,i.z])
            res = np.vstack((res, row))

        blend=np.array([])
        for fb in landmarker.face_blendshapes[0]:
            blend=np.append(blend,fb.score)
        
        landmark_save_path=os.path.join(save_path1,f"{k}_landmark.npy")
        blend_save_path=os.path.join(save_path1,f"{k}_blend.npy")
        #print(landmark_save_path)
        
        np.save(landmark_save_path,res)
        np.save(blend_save_path,blend)
            
    def process_video(self,video_path,savepath_landmarks,savepath_faces):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video ------{video_path}")
            return
        i=0    
        while cap.isOpened():
            i+=1
            ret, frame = cap.read()
            if not ret:
                break

            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            try:
                self.process_frames(frame,i,savepath_landmarks)
            except Exception as e:
                print(f"error in {savepath_landmarks} -- {e}")
                continue

            save_location=os.path.join(savepath_faces,f"{i}.jpg")
            #print(save_location)
            try:
                cv2.imwrite(save_location, frame)
            except Exception as e:
                print(f"error in {save_location} -- {e}")
                continue
           
        cap.release()

    def mp_handler(self,job):           
        vpath, land_save_path, faces_save_path = job
        os.makedirs(faces_save_path,exist_ok=True)
        os.makedirs(land_save_path,exist_ok=True)
        try:
            self.process_video(vpath,land_save_path,faces_save_path)                     
        except Exception as e:
            print(e)
    
    def createdata(self):
        jobs=[]
        speakers=os.listdir(self.root_folder)
        for s in tqdm(speakers):
            s_path=os.path.join(self.root_folder,s)
            for videos in tqdm(os.listdir(s_path)):
                faces_save_path= os.path.join(self.save_path_faces,s,videos.split(".")[0])
                land_save_path= os.path.join(self.save_path_landmarks,s,videos.split(".")[0])
                if(os.path.exists(land_save_path)):
                    continue
                jobs.append((os.path.join(s_path,videos),land_save_path,faces_save_path))
                
            # self.process_video(os.path.join(s_path,videos),land_save_path,faces_save_path)
            #print("------------------------------------------------------")
            #print(f"--------------------Done for Speaker - {s} -- Saved in path -{faces_save_path}------------------- ")
            #print("------------------------------------------------------")

        p = ThreadPoolExecutor(18)
        futures = [p.submit(self.mp_handler, j) for j in jobs]
        rand_res = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]
    	


# ------------------------- Initialize paths ---------------------

root_folder="/ssd_scratch/cvit/souvikg544/gridcorpus/video"
save_root_faces="/ssd_scratch/cvit/souvikg544/gridcorpus/faces"
save_root_landmarks="/ssd_scratch/cvit/souvikg544/gridcorpus/landmarks"

# ---------------------------- Call the functions ------------------

mklipnet=preprocess(root_folder,save_root_faces,save_root_landmarks)
mklipnet.createdata()
