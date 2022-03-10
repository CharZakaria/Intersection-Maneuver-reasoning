
import cv2
import copy
import numpy as np
from detectors import Detectors
# a new detection method is available at new_detector.py (that uses YOLOv5 in a Ressource-efficient framework)
# call the new detection method using :  optimized_detection_method(frame, model_name = 'yolov5s')
from tracker import Tracker

import os

def trans_pixel(x,y,matrix_h):

	tran_x = (matrix_h[0][0]*int(x) + matrix_h[0][1]*int(y) + matrix_h[0][2])// (matrix_h[2][0]*int(x) + matrix_h[2][1]*int(y) + matrix_h[2][2])
	tran_y = (matrix_h[1][0]*x + matrix_h[1][1]*y + matrix_h[1][2])// (matrix_h[2][0]*x + matrix_h[2][1]*y + matrix_h[2][2])
	
	return int(tran_x),int(tran_y)

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def format_img_size(img,img_min_side=500):
	""" formats the image size based on config """
	img_min_side = img_min_side
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		# ratio = 1
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img,new_width,new_height


def record_video(folder_source):

	fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
	video=cv2.VideoWriter(folder_source+'.avi', fourcc, 250,(888,500))
	print("start video generation")

	sorted_files = [s for s in os.listdir(folder_source)
	     if os.path.isfile(os.path.join(folder_source, s))]
	sorted_files.sort(key=lambda s: os.path.getmtime(os.path.join(folder_source, s)))

	for j in sorted_files:
		img = cv2.imread('./'+folder_source+'/'+j)
		video.write(img)
	
	print("video generated")
	cv2.destroyAllWindows()
	video.release()


def main():

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	frame_count = 1
	# Create Object Detector
	detector = Detectors()

	# Create Object Tracker
	tracker = Tracker(50, 8, 5)
	

	track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
					(0, 255, 255), (255, 0, 255), (255, 127, 255),
					(127, 0, 255), (127, 0, 127)]


	pts_src_np = np.float32([					[385, 108],[478, 108],				[984, 100],[1080,100],
							[0,260],            [468,200]
							])

	pts_dst = np.float32([				
												[500,500 ],[650, 500],				[1350, 500],[1500, 500],
							[0,1100],            [500,1100]
							])
	
	# # Calculate Homography
	matrix_h, status = cv2.findHomography(pts_src_np, pts_dst)
	
	path_video = '/path_to_video/' 
	icon_dims = 75
	logo_dims = 114.7
	non_stop_icon= cv2.resize(cv2.imread("./icons/non_stop.png"), (icon_dims, icon_dims), interpolation=cv2.INTER_CUBIC)

	stop_icon = cv2.resize(cv2.imread("./icons/stop.png"), (icon_dims, icon_dims), interpolation=cv2.INTER_CUBIC)
	non_row_icon = cv2.resize(cv2.imread("./icons/non_row.png"), (icon_dims, icon_dims), interpolation=cv2.INTER_CUBIC)
	row_icon = cv2.resize(cv2.imread("./icons/row.png"), (icon_dims, icon_dims), interpolation=cv2.INTER_CUBIC)

	up_left,up_left_width,up_left_height = format_img_size(cv2.imread("./icons/up_left.png"),icon_dims)
	up_right,up_right_width,up_right_height = format_img_size(cv2.imread("./icons/up_right.png"),icon_dims)
	left_up,left_up_width,left_up_height = format_img_size(cv2.imread("./icons/left_up.png"),icon_dims)
	right_up,right_up_width,right_up_height = format_img_size(cv2.imread("./icons/right_up.png"),icon_dims)
	right_left,right_left_width,right_left_height = format_img_size(cv2.imread("./icons/right_left.png"),icon_dims)
	left_right,left_right_width,left_right_height = format_img_size(cv2.imread("./icons/left_right.png"),icon_dims)
	none_icon,none_icon_width,none_icon_height = format_img_size(cv2.imread("./icons/none.png"),icon_dims)

	ticlablogo,ticlab_width,ticlab_height = format_img_size(cv2.imread("./icons/uir.png"),logo_dims)
	
	

	path_video = './dataset/' 
	frame_rate = 30

	for f in os.listdir(path_video):
		cap = cv2.VideoCapture(path_video+f)
		ret, orig_frame0 = cap.read()
		

		len_trajectory_history,len_stopped_history,len_non_stopped_history,len_respect_row_history,len_non_respect_row_history = 0,0,0,0,0
		
		while(ret):
			# Capture frame-by-frame
			
			
				
			frame_count += 1
			
			if frame_count>=0:
				
				# if frame_count%1000==0:
				# 	print(frame_count)

				frame = orig_frame0[380:800,100:].copy()
				
				ppts = np.array([[800,0],[1000,50],[1150,70],[1400,100],[1820,100],[1820,0]], np.int32)
				cv2.fillConvexPoly(frame,ppts,0)
				ppts = np.array([[0,0],[0,150],[400,130],[300,0]], np.int32)
				cv2.fillConvexPoly(frame,ppts,0)
				ppts = np.array([[0,420],[700,300],[1820,300],[1820,420]], np.int32)
				cv2.fillConvexPoly(frame,ppts,0)
				
				centers,vehicle_types = detector.detect(frame)

				tracker.Update(centers,vehicle_types,frame_count,matrix_h)
				

				
				## Reasoning System

				tracker.calculate_histories()



				len_trajectory_history =	len(tracker.trajectory_history)			
				len_non_respect_row_history =	len(tracker.non_respect_row_history)
				len_respect_row_history =	len(tracker.respect_row_history)
				len_non_stopped_history =	len(tracker.non_stopped_history)
				len_stopped_history =	len(tracker.stopped_history)
				
				if len_trajectory_history > 0:
					resulted_trajectory = tracker.trajectory_history[-1]
				
				else:
					resulted_trajectory = "none"
				
				if str(resulted_trajectory) == "[1, 11, 0, 2]":
					img_traject,traject_width,traject_height = up_left,up_left_width,up_left_height
				if str(resulted_trajectory) == "[1, 11, 0, 3]":
					img_traject,traject_width,traject_height = up_right,up_right_width,up_right_height
				if str(resulted_trajectory) == "[2, 0, 11, 1]":
					img_traject,traject_width,traject_height = left_up,left_up_width,left_up_height
				if str(resulted_trajectory) == "[3, 0, 11, 1]":
					img_traject,traject_width,traject_height = right_up,right_up_width,right_up_height
				if str(resulted_trajectory) == "[3, 0, 2]":
					img_traject,traject_width,traject_height = right_left,right_left_width,right_left_height
				if str(resulted_trajectory) == "[2, 0, 3]":
					img_traject,traject_width,traject_height = left_right,left_right_width,left_right_height

				if str(resulted_trajectory) == "none":
					img_traject,traject_width,traject_height = none_icon,none_icon_width,none_icon_height

				resized_frame,new_width,new_height = format_img_size(orig_frame0)
				resized_frame[new_height-icon_dims:,:]=0				
				resized_frame[:ticlab_height,:ticlab_width]=ticlablogo
				resized_frame[new_height-icon_dims:,0:icon_dims]=non_row_icon
				
				cv2.putText(resized_frame,str(len_non_respect_row_history), (icon_dims+25,new_height-13), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4,cv2.LINE_AA)

				resized_frame[new_height-icon_dims:,new_width//5:new_width//5+icon_dims]=row_icon
				cv2.putText(resized_frame,str(len_respect_row_history), (new_width//5+icon_dims+25,new_height-13),cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0),4,cv2.LINE_AA)

				resized_frame[new_height-icon_dims:,2*new_width//5:2*new_width//5+icon_dims]=non_stop_icon
				cv2.putText(resized_frame,str(len_non_stopped_history), (2*new_width//5+icon_dims+25,new_height-13), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),4,cv2.LINE_AA)

				resized_frame[new_height-icon_dims:,3*new_width//5:3*new_width//5+icon_dims]=stop_icon
				cv2.putText(resized_frame,str(len_stopped_history), (3*new_width//5+icon_dims+25,new_height-13), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0),4,cv2.LINE_AA)
				
				resized_frame[new_height-traject_height:, 4*new_width//5: 4*new_width//5+traject_width]=img_traject

				#cv2.imwrite("correct_demos/"+str(frame_count)+".png",resized_frame)
				
				cv2.imshow('Intersection Analysis using AI Reasoning', resized_frame)
				cv2.waitKey(1)


				# cv2.imshow('warpPerspective', im_out)
				# cv2.waitKey(1)

				# Check for key strokes
			ret, orig_frame0 = cap.read()
			# orig_frame0 = rotate(orig_frame0,-50)

			k = cv2.waitKey(1) & 0xff
			if k == 27:  # 'esc' key has been pressed, exit program.
				break

		# When everything done, release the capture
		cap.release()
		cv2.destroyAllWindows()

	record_video("correct_demos")


if __name__ == "__main__":
	# execute main
	main()
