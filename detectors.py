import cv2
import numpy as np

# Load Yolo

class Detectors(object):

	def __init__(self):

		self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
		self.classes = []
		self.considered_classes = ['car','bus','train','truck','bicycle','motorbike']

		with open("coco.names", "r") as f:
			self.classes = [line.strip() for line in f.readlines()]
		layer_names = self.net.getLayerNames()
		self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
	# colors = np.random.uniform(0, 255, size=(len(classes), 3))

	def detect(self,img):
		
		all_dets = []
		vehicle_types = []
		# img = all_img[110:300,600:1023]

		# img = cv2.resize(img, None, fx=0.4, fy=0.4)
		height, width, channels = img.shape

		# Detecting objects
		blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

		self.net.setInput(blob)
		outs = self.net.forward(self.output_layers)

		# Showing informations on the screen
		class_ids = []
		confidences = []
		boxes = []
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.5 and self.classes[class_id] in self.considered_classes:
					# Object detected
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)

					# Rectangle coordinates
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)

					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					class_ids.append(class_id)

		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
		# print(indexes)

		for i in range(len(boxes)):
			if i in indexes:
				x, y, w, h = boxes[i]
				label = str(self.classes[class_ids[i]])

				all_dets.append([x+w/2 , y+h/2])
				vehicle_types.append(label)
		
		return all_dets,vehicle_types          




# cap = cv2.VideoCapture('/home/howdrive/Videos/20201101_142132.mp4') 
# while(True):
# 	ret,all_img=cap.read()

# 	# Loading image
# 	# cv2.imwrite("frame_shape.jpg",img)
# 	# img = all_img
# 	img = all_img[380:800,:]

# 	all_dets , vehicle_types = detect_yolo_intersec(img,self.classes,self.considered_classes,self.net,output_layers)
	


# 	cv2.imshow("Image", img)
# 	k = cv2.waitKey(50) & 0xff
# 	if k == 27:  # 'esc' key has been pressed, exit program.
# 		break

# cv2.destroyAllWindows()