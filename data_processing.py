import pickle
import numpy as np
import matplotlib.pyplot as pl
import pymatreader

datapath = '/home/jagrole/AAU/9.Sem/Data/Processed_data_ALL.mat'

all_data = pymatreader.read_mat('/home/jagrole/AAU/9.Sem/Code/Processed_data_ALL.mat')

cgm_data = all_data['pkf']['cgm'][0:2]
timedata = all_data['pkf']['timecgm'][0:293]

stop_flag = False  
y_prev = None 

for x in range(len(cgm_data)):
    current_data = cgm_data[x]
    current_time_data = timedata[x]
    
    for idx, y in enumerate(current_time_data):
        try:
            if y_prev is not None and abs(y - y_prev) > 1000:  
                stop_flag = True
                stop_time_idx = idx  
                break
        except:
            pass 
        
        y_prev = y  
    
    if stop_flag == True:
        cgm_block1 = current_data[0:stop_time_idx]  
        cgm_block2 = current_data[stop_time_idx:]

        time_block1 = current_time_data[0:stop_time_idx]
        time_block2 = current_time_data[stop_time_idx:]

        if len(time_block1) > 1:
            db_list = []
            db_list2 = []
            dbfile1 = open('cgm_block1_idx'+ str(x), 'wb')
            for i in range(len(time_block1)):
                db_list.append([cgm_block1[i], time_block1[i]])
            pickle.dump(db_list, dbfile1)                    
            dbfile1.close()
            dbfile2 = open('cgm_block2_idx' + str(x), 'wb')
            for i in range(len(time_block2)):
                db_list2.append([cgm_block2[i], time_block2[i]])
            pickle.dump(db_list2, dbfile2)
            dbfile2.close()
        else:
            db_list = []
            dbfile1 = open('cgm_block1_idx'+ str(x), 'wb')
            for i in range(len(time_block1)):
                db_list.append([cgm_block1[i], time_block1[i]])
            pickle.dump(db_list, dbfile1)                    
            dbfile1.close()


