
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
	
	path_video = '/home/howdrive/Videos/intersection_videos/'  
	frame_rate = 30

	for f in os.listdir(path_video):
		frame_zero = 0
		cap = cv2.VideoCapture(path_video+f)
		ret, orig_frame0 = cap.read()
		

		while(ret):
			# Capture frame-by-frame
			
			
				
			frame_count += 1
			frame_zero +=1
			
			if frame_count%1==0:
				

				frame = orig_frame0[380:800,100:]
				
				ppts = np.array([[850,0],[1150,70],[1400,100],[1820,100],[1820,0]], np.int32)
				cv2.fillConvexPoly(frame,ppts,0)
				ppts = np.array([[0,0],[0,150],[400,130],[300,0]], np.int32)
				cv2.fillConvexPoly(frame,ppts,0)
				ppts = np.array([[0,420],[730,330],[1820,300],[1820,420]], np.int32)
				cv2.fillConvexPoly(frame,ppts,0)
				
				centers,vehicle_types = detector.detect(frame)

				tracker.Update(centers,vehicle_types,frame_count,matrix_h)
				

				for i in range(len(tracker.tracks)):
					# if tracker.tracks[i].skipped_frames==0 and tracker.tracks[i].stat=="vehicle":                    
					if tracker.tracks[i].stat=="vehicle":                    
						# print("vehicle : ",tracker.tracks[i].track_id," ",(int(tracker.tracks[i].prediction[0]),int(tracker.tracks[i].prediction[1])),"frame : ", frame_count)
						clr = tracker.tracks[i].track_id % 9
						# calc_velocity = "{:.2f} Km/h ".format(tracker.tracks[i].vehicle_velocity) 
						if tracker.tracks[i].nbr_speed>0:
							
							mean_speed = (tracker.tracks[i].sum_speed/tracker.tracks[i].nbr_speed)
							if mean_speed>0:
								cv2.putText(frame, "{:.0f}".format(mean_speed),(tracker.tracks[i].trace[0][0]+10,tracker.tracks[i].trace[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,track_colors[clr],2,cv2.LINE_AA)
						
						if (len(tracker.tracks[i].trace) > 1):
							
							for j in range(len(tracker.tracks[i].trace)-1):
								# Draw trace line

								x1 = tracker.tracks[i].trace[j][0]
								y1 = tracker.tracks[i].trace[j][1]
								x2 = tracker.tracks[i].trace[j+1][0]
								y2 = tracker.tracks[i].trace[j+1][1]
								
								
								cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), track_colors[clr], 2)

				cv2.putText(frame,tracker.result_text , (250,1200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
				orig_frame = frame
				display_frame_count = frame_zero/frame_rate
				
				cv2.putText(orig_frame,f+' / : ' "{:.2f}".format(display_frame_count), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
				

				
				cv2.imshow('Analyzing', orig_frame)
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
