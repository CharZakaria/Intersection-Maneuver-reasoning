
import cv2
import pandas as pd

from matplotlib import pyplot as plt
import torch
import numpy as np
from mean_average_precision import MetricBuilder




def visualize(img , boxes, color):
    
    print(len(boxes))
    for box in boxes:
        
        cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), color, 2)
        cv2.putText(img,str(box[-1]), (box[0],box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 2, color)
    cv2.imshow('img',img)        



def bbox_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = xB - xA + 1
    interH = yB - yA + 1

    if interW <=0 or interH <=0 :
        return -1.0

    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# In[85]:


def overlapp(pred_boxs, gt_boxs,self_overlap=0):
    
    if self_overlap==1:
        our_iou=.8
    else:
        our_iou=.2
        
    overlapped =[] 
    
    for i in range(len(pred_boxs)):
        if self_overlap==1:
            j_start=i+1
        else:
            j_start=0
            
        for j in range(j_start,len(gt_boxs)):   
#             if(i!=j):


            iou = bbox_iou(pred_boxs[i],gt_boxs[j])
            if iou > our_iou:
#                 if pred_boxs[i][-1] == gt_boxs[j][-1]:
                if 1==1:
            
                    if (pred_boxs[i][2]-pred_boxs[i][0])*(pred_boxs[i][3]-pred_boxs[i][1])> (gt_boxs[j][2]-gt_boxs[j][0])*(gt_boxs[j][3]-gt_boxs[j][1]):

                        overlapped.append((i,j,iou,1,j))
                    else:

                        overlapped.append((i,j,iou,1,i))
#                 else: overlapped.append((i,j,iou,0,-1))
                
    return overlapped
                    

def gt_box(label_file, img):
    labels= pd.read_csv(path_labels + label_file, sep=' ', header =None)
    gt_boxs = []
    h,w,ccc=img.shape
    for i in range(len(labels)):
        xmin,ymin,xmax,ymax= labels.iloc[i][1], labels.iloc[i][2], labels.iloc[i][3], labels.iloc[i][4]


        gt_boxs.append([xmin,ymin,xmax,ymax,0,0,0])
    return gt_boxs



def Background_subtraction(frame):
    
    
    # backround subtraction
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)                                          
    
    dilation = cv2.dilate(fgmask,np.ones((5,5),np.uint8),iterations = 5)
    edged = cv2.Canny(dilation, 30, 200)  
    
    return edged

def relative_to_absolute(relative_detections,bb,h_padding=0,w_padding=0): # return absolute_dets
    
    abs_dets=[]

    for rel_det in relative_detections:
        xmin,ymin,xmax,ymax,_,class_id = rel_det   

        abs_xmin = xmin + w_padding + bb[0]
        abs_xmax = xmax + w_padding + bb[0]

        abs_ymin = ymin + h_padding + bb[1]
        abs_ymax = ymax + h_padding + bb[1]

        abs_det = [abs_xmin,abs_ymin,abs_xmax,abs_ymax,class_id]
        abs_dets.append(abs_det)
        
    return abs_dets

def detect_moving_objects(frame,edged):
                                        
    moving_objects=[]

    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    try: hierarchy = hierarchy[0]
    except: hierarchy = []
    height, width = edged.shape
    min_x, min_y = width, height
    max_x = max_y = 0  
    
    pre_process,  inference,  NMS , nbr_predections_ours=[],[],[],[]
   
    for contour, hier in zip(contours, hierarchy):
        
        (x,y,w,h) = cv2.boundingRect(contour)

        if w > 10 and h > 10:
            roi= frame[y:y+h, x:x+w] 
            roi = roi.astype(np.uint8)
            h_roi,w_roi,_=roi.shape
            results = model(roi,max(h_roi,w_roi))
            results=results.__dict__['pred'][0].type(torch.int32).tolist()
            if results:
                abs_dets= relative_to_absolute(results,(x,y))
                moving_objects.extend(abs_dets)

    return moving_objects
            

def pred_box(pred_boxs_copy):

    pred_boxs = pred_boxs_copy
    for box in pred_boxs:
            
        if(classes[box[-1]] not in ['car','bus','truck']):
            pred_boxs.remove(box)
    for box in pred_boxs:
        box[4]=0 #remove
        box.extend([0.9]) #remove
        
    overlapped = overlapp(pred_boxs, pred_boxs,self_overlap=1)

    temp=[]
    for i in range(len(overlapped)):
#         if overlapped[i][0] !=  overlapped[i][1] and pred_boxs[overlapped[i][-1]] not in temp:      
        if pred_boxs[overlapped[i][-1]] not in temp:        
            temp.append(pred_boxs[overlapped[i][-1]])

    for box in temp:     
        pred_boxs.remove(box)
    
    return pred_boxs



kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=False)

for model_name in ['yolov5s','yolov5m', 'yolov5l', 'yolov5x']:
    
    model = torch.hub.load('ultralytics/yolov5', model_name)
    model.conf=0.30

    #extract classes names
    frame_name = str(1).zfill(8) + '.jpg'    
    frame = cv2.imread(path_frames+ frame_name)[:,:,::-1] 
    results = model(frame)
    classes= results.__dict__['names']

    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    for i in range(7200,8199):
           
        frame_name = str(i).zfill(8) + '.jpg'    
        frame = cv2.imread(path_frames+ frame_name)[:,:,::-1]  
        frame=frame[:,:780] 
        label_file = str(i).zfill(8) + '.txt'
        edged = Background_subtraction(frame)

        moving_objects_d = detect_moving_objects(frame,edged)        
        
        gt_boxs= gt_box(label_file,frame)
        pred_boxs = pred_box(moving_objects_d)
        if pred_boxs:
            # print('pred_boxs' ,pred_boxs,'\n')
            # print('gt_boxs' ,gt_boxs,'\n')
            metric_fn.add(np.array(pred_boxs), np.array(gt_boxs))

def Background_subtraction_proposed(frame,size):
    
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)                                          
    
    #dilatation
    dilation = cv2.dilate(fgmask,np.ones((5,5),np.uint8),iterations = 5)
    dilation = cv2.resize(dilation,size)
    # cv2.imshow('dil',dilation)
    # cv2.waitKey(10)
    edged = cv2.Canny(dilation, 30, 200)  
    
    return edged

def detect_moving_objects_proposed(frame,edged):
                                        
    
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    try: hierarchy = hierarchy[0]
    except: hierarchy = []
    height, width = edged.shape
    min_x, min_y = width, height
    max_x = max_y = 0  
    # computes the bounding box for the contour, and draws it on the frame,
    
    pre_process,  inference,  NMS , nbr_predections_ours=[],[],[],[]
    
    for contour, hier in zip(contours, hierarchy):
        
        (x,y,w,h) = cv2.boundingRect(contour)
#         min_x, max_x = min(x, min_x), max(x+w, max_x)
#         min_y, max_y = min(y, min_y), max(y+h, max_y)
        
        if w > 10 and h > 10:
            
#             cv2.rectangle(img2, (x,y), (x+w,y+h), (255, 0, 0), 2)
            roi= frame[y:y+h, x:x+w] 
            roi = roi.astype(np.uint8)
            h_roi,w_roi,_=roi.shape
    #         print(h_roi,w_roi)
            results = model(roi,max(h_roi,w_roi))
            if results:
                pre_process.append(results.__dict__['t'][0])
                inference.append(results.__dict__['t'][1])
                NMS.append(results.__dict__['t'][2])
                nbr_predections_ours.append(len(results.__dict__['pred'][0]))
    
    return [len(NMS),sum(nbr_predections_ours),sum(pre_process), sum(inference), sum(NMS)]


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=False)

for model_name in ['yolov5s','yolov5m', 'yolov5l', 'yolov5x']:
    
    model = torch.hub.load('ultralytics/yolov5', model_name)
    #set confidence to 0.30
    model.conf=0.30

    #extract classes names
    frame_name = str(1).zfill(8) + '.jpg'    
    frame = cv2.imread(path_frames+ frame_name) 
    results = model(frame)
    classes= results.__dict__['names']
    h,w,_=  frame.shape
    
    ## normalize shape to 640
    if w>h and w>640:
        new_w = 640
        new_h = h * 640//w
    if w<h and h>640:
        new_h = 640
        new_w = w * 640//h



    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
    results_detections=[]
    for i in range(7200,8199): 
        result=[]  
        frame_name = str(i).zfill(8) + '.jpg'    
        frame = cv2.imread(path_frames+ frame_name)
        

        edged=Background_subtraction_proposed(frame,(new_w,new_h))


        frame = cv2.resize(frame,(new_w,new_h))

        results = model(frame,max(new_h,new_w))
        result.extend([frame_name,len(results.__dict__['pred'][0]),results.__dict__['t'][0],results.__dict__['t'][1], results.__dict__['t'][2]])
           
       
        result.extend(detect_moving_objects_proposed(frame,edged))
        results_detections.append(result)

    df = pd.DataFrame(data=results_detections)
    
    df.columns =['frame_name', 'C_nbr_predections', 'C_pre_process',  'C_inference',  'C_NMS','P_condidate_nbr', 'P_nbr_predections', 'P_sum(pre_process)', 'P_sum(inference)', 'P_sum(NMS)']
    df['C_time']=df.apply(lambda row: row['C_pre_process']+row['C_inference']+row['C_NMS'], axis=1)
    df['P_time']=df.apply(lambda row: row['P_sum(pre_process)']+row['P_sum(inference)']+row['P_sum(NMS)'], axis=1)
    print('conventional sum time',df.C_time.sum(), 'conventional mean time',df.C_time.mean())
    print('proposed sum time',df.P_time.sum(), 'proposed mean time',df.P_time.mean())
    df.to_csv('results/proposed_'+model_name+'.csv')



