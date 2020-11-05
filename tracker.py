import numpy as np
# from common import dprint
from scipy.optimize import linear_sum_assignment


class KalmanFilter(object):

	def __init__(self,b):
		
		# self.u = np.ones((4, 1))  
		self.u = np.array([[int(b[0])], [int(b[1])], [1], [1]])
		self.b = np.ones((4, 1))  
		self.F = np.array([[1,0, 1,0], [0, 1,0,1], [0, 0,1,0], [0, 0,0,1]])
		self.lastResult = np.ones((4, 1))
		self.trans_lastResult = (0, 0)
		

	def predict(self):

		self.lastResult = self.u 
		# self.u = np.round(np.dot(self.F, self.u))
		
		# if self.u[3]>0:
		if 0==0:
			self.u[3] = 0
			self.u[2] = 0
			# print("neg velocity to 0")
		# else:
		# 	self.u = np.array([int(b[0]),int(b[1]),int(b[0]) - int(self.lastResult[0]),int(b[1]) - int(self.lastResult[1])])
		
		return self.u[0], self.u[1],self.u[2], self.u[3]

	def correct(self, b,history):

		self.u = np.array([int(b[0]),int(b[1]),int(b[0]) - int(self.lastResult[0]),int(b[1]) - int(self.lastResult[1])])		

		return self.u[0], self.u[1]


def trans_pixel(x,y,matrix_h):
	
	tran_x = (matrix_h[0][0]*int(x) + matrix_h[0][1]*int(y) + matrix_h[0][2])// (matrix_h[2][0]*int(x) + matrix_h[2][1]*int(y) + matrix_h[2][2])
	tran_y = (matrix_h[1][0]*x + matrix_h[1][1]*y + matrix_h[1][2])// (matrix_h[2][0]*x + matrix_h[2][1]*y + matrix_h[2][2])
	
	return tran_x,tran_y

class Track(object):

	def __init__(self, prediction,tmp_trackIdCount):

		self.track_id = 172  # identification of each track object
		self.tmp_track_id = tmp_trackIdCount
		self.KF = KalmanFilter(prediction)  # KF instance to track this object
		self.prediction = np.array([[int(prediction[0])], [int(prediction[1])], [1], [1]])  # predicted centroids (x,y)
		self.skipped_frames = 0  # number of frames skipped undetected
		self.trace = []  # trace path
		self.age = -3
		self.stat = "tmp"
		self.tmp_follow_time = 0
		self.vehicle_velocity = 1500

		self.time_stamp = 0
		self.sum_speed = 0
		self.nbr_speed = 0
		self.vehicle_type = ""
		self.used_lane = 0
		self.headway = 0

		# print("tmp track created: ", np.asarray(prediction))

class Tracker(object):

	def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length):

		self.dist_thresh = dist_thresh
		self.max_frames_to_skip = max_frames_to_skip
		self.max_trace_length = max_trace_length
		self.tracks = []
		self.trackIdCount = 1599
		self.tmp_trackIdCount = 1
		self.dt = 1/30
		
		self.pixel_metre = 10
		
		self.counting_step = 15
		
		self.last_rec_time_l1 = 0
		self.last_rec_time_l2 = 0
		self.last_rec_time_l3 = 0
		self.last_rec_time_l4 = 0

		self.result_text = "Class | Lane | Headway | Mean speed"
	
	def Update(self, detections,vehicle_types,count_frame,matrix_h):

		# Create tracks if no tracks vector found
		# print(len(detections) , len(self.tracks) , count_frame)
		for i in range(len(self.tracks)):
			# print("update loop i : ", i , len(assignment), len(self.tracks) )
			# self.tracks[i].prediction = self.tracks[i].KF.predict()
			

			# Velocity 
			if self.tracks[i].stat=="vehicle":
				
				orig_x = int(self.tracks[i].KF.u[0])
				orig_y = int(self.tracks[i].KF.u[1])
				
				trans_x_y = trans_pixel(orig_x,orig_y,matrix_h)


				if count_frame%self.counting_step==0:
					
					if self.tracks[i].KF.trans_lastResult!=(0, 0):
						
						vehicle_velocity = np.sqrt(((trans_x_y[0]- self.tracks[i].KF.trans_lastResult[0])/(self.pixel_metre*self.counting_step*self.dt)) **2 + ((trans_x_y[1]- self.tracks[i].KF.trans_lastResult[1])/(self.pixel_metre*self.counting_step*self.dt)) **2)				
						# print("Vehicle {} frame {} Velocity {:.2f} Km/h".format(self.tracks[i].track_id ,count_frame,vehicle_velocity*3.6) )
						self.tracks[i].vehicle_velocity = vehicle_velocity
						
						if vehicle_velocity>0.0:
							
							self.tracks[i].sum_speed += vehicle_velocity*3.6
							self.tracks[i].nbr_speed += 1 
					
					else:
						self.tracks[i].time_stamp = count_frame



					self.tracks[i].KF.trans_lastResult = trans_x_y
					#print("lastResult updated : ", trans_x_y, " frame ", count_frame , "id track ",i)
			
				#Following time / distance 

				if trans_x_y[1] < 640 and trans_x_y[1] > 590 and self.tracks[i].tmp_follow_time == 0:
					self.tracks[i].tmp_follow_time = 1
					
					## Lane 4 
					if trans_x_y[0]<40 and self.last_rec_time_l4 != 0:
						follow_time = (count_frame - self.last_rec_time_l4) * self.dt
						self.tracks[i].used_lane = 4
						if follow_time< 5.0:							
							# print('Lane 4 ## follow time  is {:.2f}'.format(follow_time))
							self.tracks[i].headway = follow_time
							
						# else: 
						# 	# print('Lane 4 ##  new follownig situation')
						self.last_rec_time_l4 = count_frame

						if self.tracks[i].vehicle_velocity !=1500:
							
							follow_distance = follow_time * self.tracks[i].vehicle_velocity
							# print('Lane 4 ## follow distance is {:.2f} Vehicle {}'.format(follow_distance,self.tracks[i].track_id))
					
					if trans_x_y[0]<40 and self.last_rec_time_l4 == 0:
						# print('Lane 4 ## recorded at {} Vehicle {}'.format(count_frame,self.tracks[i].track_id))
						self.last_rec_time_l4 = count_frame
						self.tracks[i].used_lane = 4

					## Lane 3 
					if trans_x_y[0]>=40 and trans_x_y[0]<70 and self.last_rec_time_l3 != 0:
						follow_time = (count_frame - self.last_rec_time_l3) * self.dt
						self.tracks[i].used_lane = 3
						if follow_time< 5.0:							
							# print('Lane 3 ## follow time  is {:.2f}'.format(follow_time))
							self.tracks[i].headway = follow_time
							
						# else: 
						# 	print('Lane 3 ##  new follownig situation')
						self.last_rec_time_l3 = count_frame

						if self.tracks[i].vehicle_velocity !=1500:
							
							follow_distance = follow_time * self.tracks[i].vehicle_velocity
							# print('Lane 3 ## follow distance is {:.2f} Vehicle {}'.format(follow_distance,self.tracks[i].track_id))
					
					if trans_x_y[0]>=40 and trans_x_y[0]<70 and self.last_rec_time_l3 == 0:
						# print('Lane 3 ## recorded at {} Vehicle {}'.format(count_frame,self.tracks[i].track_id))
						self.last_rec_time_l3 = count_frame
						self.tracks[i].used_lane = 3

					## Lane 2 
					if trans_x_y[0]>=70 and trans_x_y[0]<100 and self.last_rec_time_l2 != 0:
						follow_time = (count_frame - self.last_rec_time_l2) * self.dt
						self.tracks[i].used_lane = 2
						if follow_time< 5.0:							
							# print('Lane 2 ## follow time  is {:.2f}'.format(follow_time))
							self.tracks[i].headway = follow_time
							
						# else: 
						# 	print('Lane 2 ##  new follownig situation')
						self.last_rec_time_l2 = count_frame

						if self.tracks[i].vehicle_velocity !=1500:
							
							follow_distance = follow_time * self.tracks[i].vehicle_velocity
							# print('Lane 2 ## follow distance is {:.2f} Vehicle {}'.format(follow_distance,self.tracks[i].track_id))
					
					if trans_x_y[0]>=70 and trans_x_y[0]<100 and self.last_rec_time_l2 == 0:
						# print('Lane 2 ## recorded at {} Vehicle {}'.format(count_frame,self.tracks[i].track_id))
						self.last_rec_time_l2 = count_frame
						self.tracks[i].used_lane = 2

					## Lane 1

					if trans_x_y[0]>=100 and self.last_rec_time_l1 != 0:
						follow_time = (count_frame - self.last_rec_time_l1) * self.dt
						self.tracks[i].used_lane = 1
						if follow_time< 5.0:
							# print('Lane 1 ## follow time  is {:.2f}'.format(follow_time))
							self.tracks[i].headway = follow_time
							
						# else:
						# 	print('Lane 1 ##  new follownig situation')
						self.last_rec_time_l1 = count_frame
						if self.tracks[i].vehicle_velocity !=1500:	
							follow_distance = follow_time * self.tracks[i].vehicle_velocity
							# print('Lane 1 ## follow distance is {:.2f} Vehicle {}'.format(follow_distance,self.tracks[i].track_id))
					if trans_x_y[0]>=100 and self.last_rec_time_l1 == 0:
						# print('Lane 1 ## recorded at {} Vehicle {}'.format(count_frame,self.tracks[i].track_id))
						self.last_rec_time_l1 = count_frame
						self.tracks[i].used_lane = 1


					

					
					
						
						
			self.tracks[i].prediction = self.tracks[i].KF.predict()

		if (len(self.tracks) == 0):
			
			for i in range(len(detections)):
				track = Track(detections[i], self.tmp_trackIdCount)
				# print(int(detections[i][0]),int(detections[i][1])," tmp track created : ", self.tmp_trackIdCount)
				# self.trackIdCount += 1
				self.tmp_trackIdCount += 1
				self.tracks.append(track)
				# self.tracks[-1].age++

		# Calculate cost using sum of square distance between predicted vs detected centroids
	
		N = len(self.tracks)
		M = len(detections)
		
		cost = np.zeros(shape=(N, M))   # Cost matrix
		

		for i in range(len(self.tracks)):
			#self.tracks[i].age +=1
			if self.tracks[i].age==0:
				self.trackIdCount+= 1
				self.tracks[i].track_id= self.trackIdCount
				# print("tmp track ", self.tmp_trackIdCount, " to real track ", self.tracks[i].track_id , "age ", self.tracks[i].age)
				# self.tracks[i].tmp_trackIdCount=0
				self.tracks[i].tmp_track_id=0
				self.tracks[i].stat="vehicle"

				
			for j in range(len(detections)):
				try:
					# print("pred: ",self.tracks[i].prediction[0],self.tracks[i].prediction[1]) 
					# print("det: " , detections[j][0],detections[j][1])
					diff_0 = int(self.tracks[i].prediction[0])-int(detections[j][0])
					diff_1 = int(self.tracks[i].prediction[1])-int(detections[j][1])

					sos = diff_0*diff_0 + diff_1*diff_1

					distance = np.sqrt(sos)
					# print("tr ",i," det ",j," distance : ",distance,count_frame)
					if distance> 2*self.dist_thresh:
						distance = 1e6
					else:
						distance/=2

					cost[i][j] = distance
					# print("calc costs" , distance, "track : ", i, "det", j)
				except Exception as e:
					print(e)
					pass

		# Using Hungarian Algorithm assign the correct detected measurements
		# to predicted tracks
		assignment = []
		for _ in range(N):
			assignment.append(-1)
		row_ind, col_ind = linear_sum_assignment(cost)
		for i in range(len(row_ind)):
			assignment[row_ind[i]] = col_ind[i]
			# print(assignment[row_ind[i]] ,"assigned with ", col_ind[i])

		# Identify tracks with no assignment, if any
		un_assigned_tracks = []
		for i in range(len(assignment)):
			if (assignment[i] != -1):
				# check for cost distance threshold.
				# If cost is very high then un_assign (delete) the track
				if (cost[i][assignment[i]] > self.dist_thresh):
					assignment[i] = -1
					un_assigned_tracks.append(i)
					self.tracks[i].skipped_frames += 1
					# print("skipped frame")
				pass
			else:
				self.tracks[i].skipped_frames += 1
				# print("skipped frame")

		
		# Now look for un_assigned detects

		for i in range(len(assignment)):
			self.tracks[i].age +=1

		un_assigned_detects = []
		for i in range(len(detections)):
				if i not in assignment:
					un_assigned_detects.append(i)



		# Update KalmanFilter state, lastResults and tracks trace
		# for i in range(len(assignment)):
		for i in range(len(self.tracks)):

			if(assignment[i] != -1):
				self.tracks[i].skipped_frames = 0
				self.tracks[i].prediction = self.tracks[i].KF.correct(detections[assignment[i]], self.tracks[i].trace)
				# print(vehicle_types[assignment[i]],self.tracks[i].track_id,self.tracks[i].age )
				if self.tracks[i].age>0 and self.tracks[i].vehicle_type=="":
					self.tracks[i].vehicle_type = vehicle_types[assignment[i]]
					# print(self.tracks[i].vehicle_type)

			if(len(self.tracks[i].trace) > self.max_trace_length):
				for j in range(len(self.tracks[i].trace) - self.max_trace_length):
					del self.tracks[i].trace[j]
			
			if self.tracks[i].age> -1:
				
				# self.tracks[i].trace.append(self.tracks[i].prediction)
				self.tracks[i].trace.append([self.tracks[i].KF.u[0],self.tracks[i].KF.u[1],self.tracks[i].vehicle_velocity])

		# Start new tracks
		if(len(un_assigned_detects) != 0):
			for i in range(len(un_assigned_detects)):
				track = Track(detections[un_assigned_detects[i]], self.tmp_trackIdCount)
				self.tmp_trackIdCount += 1
				self.tracks.append(track)

		# If tracks are not detected for long time, remove them
		del_tracks = []
		
		for i in range(len(self.tracks)):
			if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
				del_tracks.append(i)
				if self.tracks[i].used_lane!=0 and self.tracks[i].vehicle_type != "":
					
					mean_speed = 0
					if self.tracks[i].nbr_speed>0:
						mean_speed = (self.tracks[i].sum_speed/self.tracks[i].nbr_speed)
					print("{}|{}|{}|{}|{:.2f}|{:.2f} Km/h" .format(self.tracks[i].track_id,self.tracks[i].time_stamp,self.tracks[i].vehicle_type,self.tracks[i].used_lane,self.tracks[i].headway,mean_speed))
					self.result_text= "{} | Lane {} | {:.2f} | {:.2f} Km/h".format(self.tracks[i].vehicle_type,self.tracks[i].used_lane,self.tracks[i].headway,mean_speed)
					
					record_to_file = open("records.txt","a+")
					record_to_file.write("{};{};{};{};{};{:.2f} \n" .format(self.tracks[i].track_id,self.tracks[i].time_stamp,self.tracks[i].vehicle_type,self.tracks[i].used_lane,self.tracks[i].headway,mean_speed))
					record_to_file.close()
			if (self.tracks[i].stat=="tmp" and self.tracks[i].skipped_frames>0):
				del_tracks.append(i)
				
					
		if len(del_tracks) > 0:  
			del_tracks.sort(reverse=True)
			for id in del_tracks:
				if id < len(self.tracks):
					# print(("{};{};{};{};{};{:.2f} \n" .format(self.tracks[id].track_id,self.tracks[id].time_stamp,self.tracks[id].vehicle_type,self.tracks[id].used_lane,self.tracks[id].headway,self.tracks[id].sum_speed)))
					del self.tracks[id]
					
					# del assignment[id]
				else:
					print("ERROR: id is greater than length of tracks")


