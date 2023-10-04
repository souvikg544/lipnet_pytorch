import cv2
import os
import shutil

for j in range(1,6,1):

    print(f"----------------------------------- Speaker {j} --------------------------------------")
    for files in os.listdir(f"/ssd_scratch/cvit/souvikg544/gridcorpus/landmarks/s{j}"):
        #video_path=os.path.join(f"/home2/souvikg544/gridcorpus/video/s{j}/",files)
       
        folder_path=os.path.join(f"/ssd_scratch/cvit/souvikg544/gridcorpus/faces/s{j}",files)
        align_path=os.path.join(f"/ssd_scratch/cvit/souvikg544/gridcorpus/transcription/s{j}",f"{files}.align")
            
        if not os.path.exists(align_path):
            shutil.rmtree(folder_path)
            continue
        align_data = []
        with open(align_path, 'r') as align_file:
            for line in align_file:
                _,timestamp, label = line.strip().split()
                label=label.lower()
                # if(label=="sil"):
                #     continue
                align_data.append(label)
        
        i=75
        file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        if ((file_count != i) or (len(align_data)!=8)):
            shutil.rmtree(folder_path)
            print(f"{folder_path} ---- file count ---{file_count} ----- opencv count {i}  ---len{len(align_data)}")