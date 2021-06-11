import cv2,os
import numpy as np

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
logo_dims = 121.175

ticlablogo,ticlab_width,ticlab_height = format_img_size(cv2.imread("./icons/ticlab_new.png"),logo_dims)
# choose codec according to format needed

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
folder_source = "correct_demos"
# sorted([os.listdir('./'+folder_source)].)
def getfiles(dirpath):
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    return a
filesss = getfiles(folder_source)


video=cv2.VideoWriter('whole_clip_newlogo.avi', fourcc, 180,(888,500))
for j in range(1410,2221):
	# print(j)
	img = cv2.imread('./correct_demos/'+str(j)+'.png')
	img[:ticlab_height,:ticlab_width]=ticlablogo
	video.write(img)
for j in range(2360,2801):
	# print(j)
	img = cv2.imread('./correct_demos/'+str(j)+'.png')
	img[:ticlab_height,:ticlab_width]=ticlablogo
	video.write(img)
for j in range(3585,4751):
	# print(j)
	img = cv2.imread('./correct_demos/'+str(j)+'.png')
	img[:ticlab_height,:ticlab_width]=ticlablogo
	video.write(img)
print("video generated")
cv2.destroyAllWindows()
video.release()