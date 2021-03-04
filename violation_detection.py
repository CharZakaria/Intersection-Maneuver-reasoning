import pandas as pd
import ast

pd_scene=pd.read_csv("final_stats.csv", sep=';')
pd_scene.columns=["vehicle_type","trajectory","stat_t1","stat_t2","stat_t3","stat_t4","mean_speed","min_stop_speed"]


#Zero_speed at stop zone violation
pd_scene[pd_scene['min_stop_speed']>0]

def convert(x):
    try:
        return ast.literal_eval(x)
    except:
        
        return ast.literal_eval(x.split(' ')[1])
        
def preprocess_data(pd_scene):
	
	pd_scene['stat_t1']=pd_scene['stat_t1'].apply(lambda x:convert(x))
	pd_scene['stat_t2']=pd_scene['stat_t2'].apply(lambda x:convert(x))
	pd_scene['stat_t3']=pd_scene['stat_t3'].apply(lambda x:convert(x))

	return pd_scene


def calculate_right_of_way_violations(pd_scene):

	pd_scene['vl1_11']=-1 # for vehicle coming from stop zone
	pd_scene['vl2_11']=-1 # for vehicle coming from zone 2 --> zone 1 
	pd_scene['vl3_11']=-1 # for vehicle coming from zone 3 --> zone 1 


	pd_scene["vl1_11"]=pd_scene.apply(lambda row:1 if (row['trajectory']=='[1,11,0,2]' or row['trajectory']== '[1,11,0,3]') and sum(row['stat_t3'][1:3])!=0 else 0,axis=1)
	pd_scene["vl2_11"]=pd_scene.apply(lambda row:1 if (row['trajectory']=='[2,0,11,1]') and row['stat_t2'][2]+row['stat_t2'][4]>0 else 0,axis=1)
	pd_scene["vl3_11"]=pd_scene.apply(lambda row:1 if (row['trajectory']=='[3,0,11,1]') and row['stat_t2'][4]>0 else 0,axis=1)

	len_vl1_11 = len(pd_scene[pd_scene['vl1_11']==1])
	len_vl2_11 = len(pd_scene[pd_scene['vl2_11']==1])
	len_vl3_11 = len(pd_scene[pd_scene['vl3_11']==1])

	return len_vl1_11, len_vl2_11, len_vl3_11 