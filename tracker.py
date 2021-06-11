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
		self.min_recorded_speed = 1000
		self.nbr_speed = 0
		self.vehicle_type = ""
		self.trans_x_y = [0,0]

		self.zone_from = 100
		self.old_zone_from = 100
		self.zone_to = 500
		self.zone_history = []
		self.all_other_zones = []

		self.used_lane = 0
		self.headway = 0

		# print("tmp track created: ", np.asarray(prediction))
	
	def update_histories(self,zone_id,zone1,zone2,zone3,zone11,zone0):
		if zone_id not in self.zone_history:
			self.zone_history.append(zone_id)
			self.other_zones_counter(zone1,zone2,zone3,zone11,zone0)

	def other_zones_counter(self,zone1,zone2,zone3,zone11,zone0):

		# call function after adding the current zone
		# [zone0,zone1,zone2,zone11,zone0]

		other_zones = [0,0,0,0,0]
		id_zone = 0
		if self.zone_history[-1]==11:
			id_zone = 3
		elif self.zone_history[-1]==0:
			id_zone = 4
		else:
			id_zone= self.zone_history[-1]-1

		
		other_zones[0]= zone1
		other_zones[1]= zone2
		other_zones[2]= zone3
		other_zones[3]= zone11
		other_zones[4]= zone0

		other_zones[id_zone]=-1

		self.all_other_zones.append(other_zones)

	def correct_zone_from(self,zone_id,zone1,zone2,zone3,zone11,zone0):
		
		if len(self.zone_history)==0:
			self.old_zone_from = zone_id
		
		if zone_id==0:
			
			

			if self.trans_x_y[0]>= 1000 and self.trans_x_y[1]>= 1200:
				zone_id = 3
			elif self.trans_x_y[0]<= 1000 and self.trans_x_y[1]>= 1200:
				zone_id = 2
			elif self.trans_x_y[0]<= 1200 and self.trans_x_y[1]<= 1200:
				zone_id = 1
			else:
				zone_id= -1

			self.zone_from=zone_id
			
			if len(self.zone_history)==0:

				self.update_histories(zone_id,zone1,zone2,zone3,zone11,zone0)

		if zone_id==11:

			self.zone_from=1
			self.update_histories(1,zone1,zone2,zone3,zone11,zone0)
	

	def route_finder(self,zone_id,zone1,zone2,zone3,zone11,zone0):
		if self.zone_from == 100:
			
			if zone_id!=0 and zone_id!=11:
				self.zone_from = zone_id
				self.route_history(zone_id,zone1,zone2,zone3,zone11,zone0)
			else:
				self.correct_zone_from(zone_id,zone1,zone2,zone3,zone11,zone0)			

		if self.zone_to == 500:
			self.zone_to = zone_id
		elif self.zone_to==self.zone_from:
			self.zone_to = 500
		else:
			self.zone_to = zone_id
	
	def route_history(self,zone_id,zone1,zone2,zone3,zone11,zone0):	
		if len(self.zone_history)==0:
			if zone_id != 100:
				self.update_histories(zone_id,zone1,zone2,zone3,zone11,zone0)

		elif self.zone_history[-1]!= zone_id and zone_id !=100:
			if zone_id not in self.zone_history:
				
				self.update_histories(zone_id,zone1,zone2,zone3,zone11,zone0)

	def route_history_saver(self):
		result = ""
		other_zones_result = ""
		if (1 in self.zone_history) or (11 in self.zone_history):
			if len(self.zone_history)==4:
				# print(self.track_id,self.zone_history)
				result = "{};{};{};[{},{},{},{}]; " .format(self.track_id,self.time_stamp,self.vehicle_type,self.zone_history[0],self.zone_history[1],self.zone_history[2],self.zone_history[3])
				
											
		else:
			if len(self.zone_history)==3:
				# print(self.track_id,self.zone_history)
				result = "{};{};{};[{},{},{}];" .format(self.track_id,self.time_stamp,self.vehicle_type,self.zone_history[0],self.zone_history[1],self.zone_history[2])
		
		for x in range(len(self.all_other_zones)):
			other_zones_result+= "[{},{},{},{},{}];".format(self.all_other_zones[x][0],self.all_other_zones[x][1],self.all_other_zones[x][2],self.all_other_zones[x][3],self.all_other_zones[x][4])

		if result!="":
			result+= other_zones_result

		return result




class Tracker(object):

	def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length):

		self.dist_thresh = dist_thresh
		self.max_frames_to_skip = max_frames_to_skip
		self.max_trace_length = max_trace_length
		self.tracks = []
		self.trackIdCount = 1
		self.tmp_trackIdCount = 1
		self.dt = 1/30
		
		self.pixel_metre = 100
		
		self.counting_step = 15
		

		self.result_text = "Class | Lane | Headway | Mean speed"
		
		## intersection analysis variables
		self.trajectory_history = []
		self.non_respect_row_history = []
		self.respect_row_history = []
		self.non_stopped_history = []
		self.stopped_history = []

		self.zone1 = 0
		self.zone2 = 0
		self.zone3 = 0
		self.zone11 = 0
		self.zone0 = 0
	
	def Update(self, detections,vehicle_types,count_frame,matrix_h):

		# Create tracks if no tracks vector found
		# print(len(detections) , len(self.tracks) , count_frame)
		for i in range(len(self.tracks)):

			if self.tracks[i].stat=="vehicle":
				
				orig_x = int(self.tracks[i].KF.u[0])
				orig_y = int(self.tracks[i].KF.u[1])
				
				trans_x_y = trans_pixel(orig_x,orig_y,matrix_h)
				self.tracks[i].trans_x_y = trans_x_y


				if count_frame%self.counting_step==0:
					
					if self.tracks[i].KF.trans_lastResult!=(0, 0):
						
						vehicle_velocity = np.sqrt(((trans_x_y[0]- self.tracks[i].KF.trans_lastResult[0])/(self.pixel_metre*self.counting_step*self.dt)) **2 + ((trans_x_y[1]- self.tracks[i].KF.trans_lastResult[1])/(self.pixel_metre*self.counting_step*self.dt)) **2)				
						# print("Vehicle {} frame {} Velocity {:.2f} Km/h".format(self.tracks[i].track_id ,count_frame,vehicle_velocity*3.6) )
						self.tracks[i].vehicle_velocity = vehicle_velocity

						# stopped vehicle detection

						if self.tracks[i].min_recorded_speed>vehicle_velocity:
							self.tracks[i].min_recorded_speed = vehicle_velocity

						if vehicle_velocity>0.0:
							
							if vehicle_velocity>=4.0:
								
								# self.tracks[i].sum_speed += vehicle_velocity*3.6
								self.tracks[i].sum_speed = round(vehicle_velocity*3.6)
								# self.tracks[i].nbr_speed += 1 
								self.tracks[i].nbr_speed =  1
							else:
								# self.tracks[i].sum_speed += 0
								self.tracks[i].sum_speed += 0
								# self.tracks[i].nbr_speed += 1 
								self.tracks[i].nbr_speed = 1 
					
					else:
						self.tracks[i].time_stamp = count_frame



					self.tracks[i].KF.trans_lastResult = trans_x_y
					#print("lastResult updated : ", trans_x_y, " frame ", count_frame , "id track ",i)
			
				#Following time / distance 

					
			self.tracks[i].prediction = self.tracks[i].KF.predict()

		if (len(self.tracks) == 0):
			
			for i in range(len(detections)):
				track = Track(detections[i], self.tmp_trackIdCount)

				self.tmp_trackIdCount += 1
				self.tracks.append(track)

	
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
				if self.tracks[i].vehicle_type != "":
					
					mean_speed = 0
					if self.tracks[i].nbr_speed>0:
						mean_speed = (self.tracks[i].sum_speed//self.tracks[i].nbr_speed)

					
					result = self.tracks[i].route_history_saver()
					if result!="":
						txt_rs = self.tracks[i].zone_history
						# print("#### ",txt_rs+"\n")
						self.trajectory_history.append(txt_rs)

						# stopped and non_stopped vehicle counting
						if len(self.tracks[i].zone_history)==4 and self.tracks[i].zone_history[0]==1:
							if self.tracks[i].min_recorded_speed<0.3:
								# self.stopped_history.append(str(self.tracks[i].min_recorded_speed))
								self.stopped_history.append("0")
							else:
								self.non_stopped_history.append("0")
								print("non stopped ", self.tracks[i].time_stamp, self.tracks[i].zone_history )

						#row violation counting 
						if (self.tracks[i].zone_history[0]==1 and len(self.tracks[i].zone_history)==4): # [1,11,0,x]
							if self.tracks[i].all_other_zones[2][1]+self.tracks[i].all_other_zones[2][2]>0 or self.tracks[i].all_other_zones[2][4]>0:
								self.non_respect_row_history.append("0")
								print("non row ", self.tracks[i].time_stamp, self.tracks[i].zone_history )

							else:
								self.respect_row_history.append("0")

						if (self.tracks[i].zone_history[0]==2 and len(self.tracks[i].zone_history)==4): # [2,0,11,1]
							if (self.tracks[i].all_other_zones[2][4]>0 or self.tracks[i].all_other_zones[2][2])>0: # check when track i enters zone 11 if someone is in zone 0 or zone 2
								self.non_respect_row_history.append("0")
								print("non row ", self.tracks[i].time_stamp, self.tracks[i].zone_history )

							else:
								self.respect_row_history.append("0")

						# record_to_file = open("./scene_stats_2021.txt","a+")
						# record_to_file.write(result+str(mean_speed)+"\n")
						# record_to_file.close()

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
	


	def locate_zone(self,desired_track=-100):
		returned_zone_id = 100
		zone1 = 0
		zone2 = 0
		zone3 = 0
		zone11 = 0
		zone0 = 0
		
		for i in range(len(self.tracks)):

			
			# zone 1
			zone_id = 100 
			if (self.tracks[i].trans_x_y[0]>=500 and self.tracks[i].trans_x_y[0]<1500 and self.tracks[i].trans_x_y[1]<650):
				zone1+=1
				zone_id = 1


			# zone 2
			if (self.tracks[i].trans_x_y[0]<500 and self.tracks[i].trans_x_y[1]>=1100):
				zone2+=1
				zone_id = 2					
			
			# zone 3 
			if (self.tracks[i].trans_x_y[0]>=1500 and self.tracks[i].trans_x_y[1]>=1100):
				zone3+=1
				zone_id = 3				

			# zone11
			# if (self.tracks[i].trans_x_y[0]>=500 and self.tracks[i].trans_x_y[0]<1500 and self.tracks[i].trans_x_y[1]>=650 and self.tracks[i].trans_x_y[1]<1100):
			if (self.tracks[i].trans_x_y[0]>=500 and self.tracks[i].trans_x_y[0]<1600 and self.tracks[i].trans_x_y[1]>=650 and self.tracks[i].trans_x_y[1]<1150):
				zone11+=1
				zone_id = 11			
			
			# zone0
			# if (self.tracks[i].trans_x_y[0]>=500 and self.tracks[i].trans_x_y[0]<1500 and self.tracks[i].trans_x_y[1]>=1100):
			if (self.tracks[i].trans_x_y[0]>=500 and self.tracks[i].trans_x_y[0]<1600 and self.tracks[i].trans_x_y[1]>=1150):
				zone0+=1
				zone_id = 0

			if i==desired_track:

				returned_zone_id = zone_id

		self.zone1 = zone1
		self.zone2 = zone2
		self.zone3 = zone3
		self.zone11 = zone11
		self.zone0 = zone0

		# print("zone count : ",desired_track,self.zone1,self.zone2,self.zone3,self.zone11,self.zone0)

		
			
		return returned_zone_id


	def calculate_histories(self):

		for i in range(len(self.tracks)):
			zone_id = self.locate_zone(i)

			if len(self.tracks[i].zone_history)==0:
				
				self.tracks[i].route_finder(zone_id,self.zone1,self.zone2,self.zone3,self.zone11,self.zone0)			
				self.tracks[i].route_history(zone_id,self.zone1,self.zone2,self.zone3,self.zone11,self.zone0)
				# self.tracks[i].route_history_saver()
				
				if self.tracks[i].old_zone_from==0:
					self.tracks[i].old_zone_from=100			

					self.locate_zone()
					self.tracks[i].update_histories(0,self.zone1,self.zone2,self.zone3,self.zone11,self.zone0)


				elif self.tracks[i].old_zone_from==11:

					self.tracks[i].old_zone_from=100
					
					self.locate_zone()
					self.tracks[i].update_histories(11,self.zone1,self.zone2,self.zone3,self.zone11,self.zone0)			

					self.locate_zone()
					self.tracks[i].update_histories(0,self.zone1,self.zone2,self.zone3,self.zone11,self.zone0)			
				
				else:
					self.tracks[i].route_finder(zone_id,self.zone1,self.zone2,self.zone3,self.zone11,self.zone0)			
					self.tracks[i].route_history(zone_id,self.zone1,self.zone2,self.zone3,self.zone11,self.zone0)

				self.tracks[i].route_history_saver()

			
			else:
				self.tracks[i].route_finder(zone_id,self.zone1,self.zone2,self.zone3,self.zone11,self.zone0)			
				self.tracks[i].route_history(zone_id,self.zone1,self.zone2,self.zone3,self.zone11,self.zone0)
				self.tracks[i].route_history_saver()

	

			# print("zone history of : ",self.tracks[i].track_id,self.tracks[i].zone_history )				

		# print("| zone1 : ",len(zone1),"| zone2 : ",len(zone2),"| zone3 : ",len(zone3),"| zone11 : ",len(zone11), "| zone0 : ",len(zone0))

	# def locate_trans(self):

		
	# 	for i in range(len(self.tracks)):
	# 		print(self.tracks[i].track_id," : " ,  self.tracks[i].trans_x_y[0],self.tracks[i].trans_x_y[1])