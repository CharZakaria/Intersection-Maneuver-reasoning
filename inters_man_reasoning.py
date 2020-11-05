
import cv2
import copy
import numpy as np
from detectors import Detectors
from tracker import Tracker
# from preprocessor import Preprocessor
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

def format_img_size(img):
	""" formats the image size based on config """
	img_min_side = 600
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
	return img

def main():

	frame_count = 1
	frame_row_thresh = 100

	# Create Object Detector
	detector = Detectors()

	# Create Object Tracker
	tracker = Tracker(80, 8, 5)
	



	# Variables initialization
	# skip_frame_count = 26800


	track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
					(0, 255, 255), (255, 0, 255), (255, 127, 255),
					(127, 0, 255), (127, 0, 127)]
	# Infinite loop to process video frames
	
	
	pts_src_np = np.float32([[1080, 35],[1186, 32],[1293, 27],
							[1073, 51],[1186, 50],[1301, 43],

							[1058, 95],[1189, 90],[1326, 88],
							[1052, 117],[1190, 112],[1333, 107],

							[1021, 195],[1191, 185],[1372, 182],
							[1009, 231],[1193, 223],[1386, 216],

							[951, 377],[1198, 368],[1456, 362],
							[926, 455],[1201, 451],[1481, 437],

							[792, 831],[1208, 832],[1630, 811],
							[717, 1047],[1208, 1063],[1700, 1026]])

	pts_dst = np.float32([				
							[40, 270],[70, 270],[100, 270],
							[40, 300],[70, 300],[100, 300],
							
							[40, 400],[70, 400],[100, 400],
							[40, 430],[70, 430],[100, 430],
							
							[40, 530],[70, 530],[100, 530],
							[40, 560],[70, 560],[100, 560],
							
							[40, 660],[70, 660],[100,660],
							[40, 690],[70, 690],[100,690],

							[40, 790],[70, 790],[100, 790],
							[40, 820],[70, 820],[100, 820]])

	# pts_src_np = np.float32([[1080, 35],[1186, 32],[1293, 27],
	# 						[1073, 51],[1186, 50],[1301, 43],

	# 						[1058, 95],[1189, 90],[1326, 88],
	# 						[1052, 117],[1190, 112],[1333, 107],

	# 						[1021, 195],[1191, 185],[1372, 182],
	# 						[1009, 231],[1193, 223],[1386, 216],

	# 						[951, 377],[1198, 368],[1456, 362],
	# 						[926, 455],[1201, 451],[1481, 437],

	# 						[792, 831],[1208, 832],[1630, 811],
	# 						[717, 1047],[1208, 1063],[1700, 1026]])

	# pts_dst = np.float32([				
	# 							 	  [100, ],[200, ],
	# 						[40, 300],[100, ],[200, ],
							
	# 						[40, 400],[70, 400],[100, 400],
	# 						[40, 430],[70, 430],[100, 430],
							
	# 						[40, 530],[70, 530],[100, 530],
	# 						[40, 560],[70, 560],[100, 560],
							
	# 						[40, 660],[70, 660],[100,660],
	# 						[40, 690],[70, 690],[100,690],

	# 						[40, 790],[70, 790],[100, 790],
	# 						[40, 820],[70, 820],[100, 820]])
	
	# # Calculate Homography
	matrix_h, status = cv2.findHomography(pts_src_np, pts_dst)
	
	path_video = '/home/howdrive/Videos/intersection_videos/'  
	
	for f in os.listdir(path_video):
		frame_zero = 0
		cap = cv2.VideoCapture(path_video+f)
		ret, orig_frame0 = cap.read()
		

		while(ret):
			# Capture frame-by-frame
			

			frame_count += 1
			frame_zero +=1
			
			
			# ppts = np.array([[0,0],[0,1439],[120,1439],[1150,0]], np.int32)
			# cv2.fillConvexPoly(orig_frame0,ppts,0)

			ppts = np.array([[470,0],[770,192],[1082,0],[1082,192]], np.int32)
			cv2.fillConvexPoly(orig_frame0,ppts,0)
			# frame = orig_frame0[380:800,100:]
			frame = orig_frame0
			# frame = orig_frame0[frame_row_thresh: , 150:]



			# Detect and return centeroids of the objects in the frame
			centers,vehicle_types = detector.detect(frame)
			# print(vehicle_types)
			# centers,res_frame = preprocessor.preprocess(frame)
			

			# Track object using Kalman Filter
			tracker.Update(centers,vehicle_types,frame_count,matrix_h)

			# For identified object tracks draw tracking line
			# Use various colors to indicate different track_id
			
			

			for i in range(len(tracker.tracks)):
				# if tracker.tracks[i].skipped_frames==0 and tracker.tracks[i].stat=="vehicle":                    
				if tracker.tracks[i].stat=="vehicle":                    
					# print("vehicle : ",tracker.tracks[i].track_id," ",(int(tracker.tracks[i].prediction[0]),int(tracker.tracks[i].prediction[1])),"frame : ", frame_count)
					clr = tracker.tracks[i].track_id % 9
					# calc_velocity = "{:.2f} Km/h ".format(tracker.tracks[i].vehicle_velocity) 
					if tracker.tracks[i].nbr_speed>0:
						
						mean_speed = (tracker.tracks[i].sum_speed/tracker.tracks[i].nbr_speed)
						cv2.putText(frame, "{:.2f}".format(mean_speed),(tracker.tracks[i].trace[0][0]+10,tracker.tracks[i].trace[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2,track_colors[clr],2,cv2.LINE_AA)
					
					if (len(tracker.tracks[i].trace) > 1):
						
						for j in range(len(tracker.tracks[i].trace)-1):
							# Draw trace line

							x1 = tracker.tracks[i].trace[j][0]
							y1 = tracker.tracks[i].trace[j][1]
							x2 = tracker.tracks[i].trace[j+1][0]
							y2 = tracker.tracks[i].trace[j+1][1]
							
							
							cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), track_colors[clr], 2)
							#cv2.putText(orig_frame,tracker.tracks[i].KF.vehicle_velocity,(x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 4,clr,2,cv2.LINE_AA)
							# velocity = str(tracker.tracks[i].KF.vehicle_velocity) + "Km/h"
							
							# cv2.circle(im_out,(trans_pixel(x1,y1,matrix_h)), 5, (0,0,255), -1)
				# Display the resulting tracking frame
				# cv2.imshow('Tracking', frame)
				# cv2.release('Tracking')
			# Display the original frame
			# orig_frame = cv2.resize(orig_frame,(frame_dim1[1],frame_dim1[0]))


			
			#cv2.circle(im_out,(trans_pixel(x,y,matrix_h)), 63, (0,0,255), -1)
			# for i in range(len(tracker.tracks)):
			# 	orig_x = tracker.tracks[i].KF.u[0]
			# 	orig_y = tracker.tracks[i].KF.u[1]

			# 	# pixel_trans = trans_pixel(orig_x,orig_y,matrix_h)
			# 	# pixel_trans = cv2.warpPerspective(orig_pixel, matrix_h, (orig_frame.shape[1],orig_frame.shape[0]))
			# 	# pixel_trans = cv2.perspectiveTransform(orig_pixel, matrix_h) 
			# 	# print((orig_x,orig_y) , " -> ", pixel_trans)


			
			
			
			
			# im_out = cv2.warpPerspective(frame, matrix_h, (frame.shape[1],frame.shape[0]))
			# im_out = im_out[700:,:200]
			# cv2.line(frame, (1, 625), (2420, 625), (0,0,0), 5)
			# cv2.line(frame, (1, 640), (2420, 640), (0,0,0), 5)
			# frame = orig_frame0
			cv2.putText(frame,tracker.result_text , (250,1200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
			# orig_frame = format_img_size(frame)
			orig_frame = frame
			# orig_frame = cv2.resize(frame,(648,380))	
			display_frame_count = frame_zero/25
			
			cv2.putText(orig_frame,f+' / : %r' %display_frame_count, (300,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
			

			
			cv2.imshow('Tracking', orig_frame)
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


if __name__ == "__main__":
	# execute main
	main()
