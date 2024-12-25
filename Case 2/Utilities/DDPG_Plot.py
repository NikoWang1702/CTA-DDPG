import numpy as np
import time
import math
import datetime
import pickle
# from tqdm import trange







#@ --------------------------------------------------------------------------------------------------------------------------- @#
class DDPG_Plot():
    def __init__(self):
        pass


#* ------------------------------------------------------------------------------------- *# 
    def __save_Total_Reward_Offline__(self, PARAM):
        file_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = "Total_Reward_Offline_" + file_timestamp + ".txt"
        with open("./Data/NN Model/{}".format(file_name), 'wb') as f:
            pickle.dump(PARAM.Total_Reward_Record, f)
        with open("./Data/NN Model/Total_Reward_Offline_Filename.txt", 'wb') as f:
            pickle.dump(file_name,f)       
        print("// ************************************************************************* //")
        print("   Total Reward saved successfully in file: " + file_name)
        print("// ************************************************************************* //")
        return file_name

    def __load_Total_Reward_Offline__(self):
        with open("./Data/NN Model/Total_Reward_Filename_Offline.txt", 'rb') as f:
            file_name = pickle.load(f)
        with open("./Data/NN Model/{}".format(file_name), 'rb') as f:
            obj = pickle.load(f)      
        return obj

    def __save_Action__(self, PARAM):
        file_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = "Action_" + file_timestamp + ".txt"
        with open("./Data/NN Model/{}".format(file_name), 'wb') as f:
            pickle.dump(PARAM.Action_Reward_Record, f)
        with open("./Data/NN Model/Action_Filename.txt", 'wb') as f:
            pickle.dump(file_name,f)       
        print("// ************************************************************************* //")
        print("   Action saved successfully in file: " + file_name)
        print("// ************************************************************************* //")
        return file_name

    def __load_Action__(self):
        with open("./Data/NN Model/Action_Filename.txt", 'rb') as f:
            file_name = pickle.load(f)
        with open("./Data/NN Model/{}".format(file_name), 'rb') as f:
            obj = pickle.load(f)     
        return obj
#* ------------------------------------------------------------------------------------- *# 


#* ------------------------------------------------------------------------------------- *# 
#? TSTA-DDPG Offline Pre-training
    def __save_Traj_Offline__(self, PARAM):
        with open("./Data/Traj Recording/Offline Training/per_step_reward_traj_offline.txt", 'wb') as f:
            pickle.dump(PARAM.reward_Offline_deque, f)

        with open("./Data/Traj Recording/Offline Training/per_step_err_traj_offline.txt", 'wb') as f:
            pickle.dump(PARAM.err_Offline_deque, f)

        
        print("// ************************************************************************* //")
        print("Offline Training trajectories have been saved in 2 seperate files!")
        print("// ************************************************************************* //")
        pass

    def __load_Traj_Offline__(self, PARAM):
        with open("./Data/Traj Recording/Offline Training/per_step_reward_traj_offline.txt", 'rb') as f:
            PARAM.reward_Offline_deque = pickle.load(f)

        with open("./Data/Traj Recording/Offline Training/per_step_err_traj_offline.txt", 'rb') as f:
            PARAM.err_Offline_deque = pickle.load(f)

        
        print("// ************************************************************************* //")
        print("Offline Training trajectories have been loaded!")
        print("// ************************************************************************* //")
        pass
#* ------------------------------------------------------------------------------------- *# 


#* ------------------------------------------------------------------------------------- *# 
#? TSTA-DDPG Offline Pre-training Figure
    def __save_Traj_Offline_forFigure__(self, PARAM):
        with open("./Data/Traj Recording/Offline Training/per_step_reward_traj_offline_for_fig.txt", 'wb') as f:
            pickle.dump(PARAM.reward_Offline_deque, f)

        with open("./Data/Traj Recording/Offline Training/per_step_err_traj_offline_for_fig.txt", 'wb') as f:
            pickle.dump(PARAM.err_Offline_deque, f)

        
        print("// ************************************************************************* //")
        print("Offline Training trajectories for Figure have been saved in 2 seperate files!")
        print("// ************************************************************************* //")
        pass

    def __load_Traj_Offline_forFigure__(self, PARAM):
        with open("./Data/Traj Recording/Offline Training/per_step_reward_traj_offline_for_fig.txt", 'rb') as f:
            PARAM.reward_Offline_deque = pickle.load(f)

        with open("./Data/Traj Recording/Offline Training/per_step_err_traj_offline_for_fig.txt", 'rb') as f:
            PARAM.err_Offline_deque = pickle.load(f)

        
        print("// ************************************************************************* //")
        print("Offline Training trajectories for Figure have been loaded!")
        print("// ************************************************************************* //")
        pass
#* ------------------------------------------------------------------------------------- *# 



#* ------------------------------------------------------------------------------------- *# 
#? TSTA-DDPG Regular Online Learning
    def __save_Traj_Online_Regular__(self, PARAM):
        with open("./Data/Traj Recording/Online Learning/yt_traj_online_regular.txt", 'wb') as f:
            pickle.dump(PARAM.yt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/yrt_traj_online_regular.txt", 'wb') as f:
            pickle.dump(PARAM.yrt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/errt_traj_online_regular.txt", 'wb') as f:
            pickle.dump(PARAM.errt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/ut_traj_online_regular.txt", 'wb') as f:
            pickle.dump(PARAM.ut_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/rewardt_traj_online_regular.txt", 'wb') as f:
            pickle.dump(PARAM.rewardt_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/average_step_reward_traj_online_regular.txt", 'wb') as f:
            pickle.dump(PARAM.Expected_step_reward_Online_list, f)
        

        with open("./Data/Traj Recording/Online Learning/lumped_disturbance_online_regular.txt", 'wb') as f:
            pickle.dump(PARAM.distrubance_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/periodic_disturbance_online_regular.txt", 'wb') as f:
            pickle.dump(PARAM.periodic_distur_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/pump_disturbance_online_regular.txt", 'wb') as f:
            pickle.dump(PARAM.pump_distur_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/disturbance_indices_online_regular.txt", 'wb') as f:
            pickle.dump(PARAM.Online_Noise_Index, f)


        with open("./Data/Traj Recording/Online Learning/total_reward_online_regular.txt", 'wb') as f:
            pickle.dump(PARAM.Total_Reward_Record_Online_list, f)
        with open("./Data/Traj Recording/Online Learning/variable_batch_length_online_regular.txt", 'wb') as f:
            pickle.dump(PARAM.Variable_Batch_Length_Online_list, f)       
        with open("./Data/Traj Recording/Online Learning/variable_initial_state_online_regular.txt", 'wb') as f:
            pickle.dump(PARAM.Variable_Initial_State_Online_list, f)

        
        print("// ************************************************************************* //")
        print("Online Learning trajectories in Regular Mode have been saved in 12 files!")
        print("// ************************************************************************* //")
        
    
    def __load_Traj_Online_Regular__(self, PARAM):
        with open("./Data/Traj Recording/Online Learning/yt_traj_online_regular.txt", 'rb') as f:
            PARAM.yt_traj_Online_deque = pickle.load(f)       
        with open("./Data/Traj Recording/Online Learning/yrt_traj_online_regular.txt", 'rb') as f:
            PARAM.yrt_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/errt_traj_online_regular.txt", 'rb') as f:
            PARAM.errt_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/ut_traj_online_regular.txt", 'rb') as f:
            PARAM.ut_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/rewardt_traj_online_regular.txt", 'rb') as f:
            PARAM.rewardt_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/average_step_reward_traj_online_regular.txt", 'rb') as f:
            PARAM.Expected_step_reward_Online_list = pickle.load(f)
        

        with open("./Data/Traj Recording/Online Learning/lumped_disturbance_online_regular.txt", 'rb') as f:
            PARAM.distrubance_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/periodic_disturbance_online_regular.txt", 'rb') as f:
            PARAM.periodic_distur_Online_deque = pickle.load(f)     
        with open("./Data/Traj Recording/Online Learning/pump_disturbance_online_regular.txt", 'rb') as f:
            PARAM.pump_distur_Online_deque = pickle.load(f)      
        with open("./Data/Traj Recording/Online Learning/disturbance_indices_online_regular.txt", 'rb') as f:
            PARAM.Online_Noise_Index = pickle.load(f)
        

        with open("./Data/Traj Recording/Online Learning/total_reward_online_regular.txt", 'rb') as f:
            PARAM.Total_Reward_Record_Online_list = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/variable_batch_length_online_regular.txt", 'rb') as f:
            PARAM.Variable_Batch_Length_Online_list = pickle.load(f)     
        with open("./Data/Traj Recording/Online Learning/variable_initial_state_online_regular.txt", 'rb') as f:
            PARAM.Variable_Initial_State_Online_list = pickle.load(f)

        print("// ************************************************************************* //")
        print("Online Learning trajectories in Regular Mode have been loaded!")
        print("// ************************************************************************* //")  
#* ------------------------------------------------------------------------------------- *#




#* ------------------------------------------------------------------------------------- *# 
#? TSTA-DDPG No_PI Online Learning
    def __save_Traj_Online_No_PI__(self, PARAM):
        with open("./Data/Traj Recording/Online Learning/yt_traj_online_no_pi.txt", 'wb') as f:
            pickle.dump(PARAM.yt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/yrt_traj_online_no_pi.txt", 'wb') as f:
            pickle.dump(PARAM.yrt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/errt_traj_online_no_pi.txt", 'wb') as f:
            pickle.dump(PARAM.errt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/ut_traj_online_no_pi.txt", 'wb') as f:
            pickle.dump(PARAM.ut_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/rewardt_traj_online_no_pi.txt", 'wb') as f:
            pickle.dump(PARAM.rewardt_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/average_step_reward_traj_online_no_pi.txt", 'wb') as f:
            pickle.dump(PARAM.Expected_step_reward_Online_list, f)
        

        with open("./Data/Traj Recording/Online Learning/lumped_disturbance_online_no_pi.txt", 'wb') as f:
            pickle.dump(PARAM.distrubance_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/periodic_disturbance_online_no_pi.txt", 'wb') as f:
            pickle.dump(PARAM.periodic_distur_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/pump_disturbance_online_no_pi.txt", 'wb') as f:
            pickle.dump(PARAM.pump_distur_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/disturbance_indices_online_no_pi.txt", 'wb') as f:
            pickle.dump(PARAM.Online_Noise_Index, f)


        with open("./Data/Traj Recording/Online Learning/total_reward_online_no_pi.txt", 'wb') as f:
            pickle.dump(PARAM.Total_Reward_Record_Online_list, f)
        with open("./Data/Traj Recording/Online Learning/variable_batch_length_online_no_pi.txt", 'wb') as f:
            pickle.dump(PARAM.Variable_Batch_Length_Online_list, f)      
        with open("./Data/Traj Recording/Online Learning/variable_initial_state_online_no_pi.txt", 'wb') as f:
            pickle.dump(PARAM.Variable_Initial_State_Online_list, f)
      
        print("// ************************************************************************* //")
        print("Online Learning trajectories in No_PI Mode have been saved in 12 files!")
        print("// ************************************************************************* //")
        
    
    def __load_Traj_Online_No_PI__(self, PARAM):
        with open("./Data/Traj Recording/Online Learning/yt_traj_online_no_pi.txt", 'rb') as f:
            PARAM.yt_traj_Online_deque = pickle.load(f)     
        with open("./Data/Traj Recording/Online Learning/yrt_traj_online_no_pi.txt", 'rb') as f:
            PARAM.yrt_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/errt_traj_online_no_pi.txt", 'rb') as f:
            PARAM.errt_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/ut_traj_online_no_pi.txt", 'rb') as f:
            PARAM.ut_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/rewardt_traj_online_no_pi.txt", 'rb') as f:
            PARAM.rewardt_Online_deque = pickle.load(f)     
        with open("./Data/Traj Recording/Online Learning/average_step_reward_traj_online_no_pi.txt", 'rb') as f:
            PARAM.Expected_step_reward_Online_list = pickle.load(f)
        

        with open("./Data/Traj Recording/Online Learning/lumped_disturbance_online_no_pi.txt", 'rb') as f:
            PARAM.distrubance_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/periodic_disturbance_online_no_pi.txt", 'rb') as f:
            PARAM.periodic_distur_Online_deque = pickle.load(f)       
        with open("./Data/Traj Recording/Online Learning/pump_disturbance_online_no_pi.txt", 'rb') as f:
            PARAM.pump_distur_Online_deque = pickle.load(f)     
        with open("./Data/Traj Recording/Online Learning/disturbance_indices_online_no_pi.txt", 'rb') as f:
            PARAM.Online_Noise_Index = pickle.load(f)
        

        with open("./Data/Traj Recording/Online Learning/total_reward_online_no_pi.txt", 'rb') as f:
            PARAM.Total_Reward_Record_Online_list = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/variable_batch_length_online_no_pi.txt", 'rb') as f:
            PARAM.Variable_Batch_Length_Online_list = pickle.load(f)       
        with open("./Data/Traj Recording/Online Learning/variable_initial_state_online_no_pi.txt", 'rb') as f:
            PARAM.Variable_Initial_State_Online_list = pickle.load(f)

        print("// ************************************************************************* //")
        print("Online Learning trajectories in No_PI Mode have been loaded!")
        print("// ************************************************************************* //")  
#* ------------------------------------------------------------------------------------- *#



#* ------------------------------------------------------------------------------------- *# 
#? TSTA-DDPG No_STER Online Learning
    def __save_Traj_Online_No_STER__(self, PARAM):
        with open("./Data/Traj Recording/Online Learning/yt_traj_online_no_ster.txt", 'wb') as f:
            pickle.dump(PARAM.yt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/yrt_traj_online_no_ster.txt", 'wb') as f:
            pickle.dump(PARAM.yrt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/errt_traj_online_no_ster.txt", 'wb') as f:
            pickle.dump(PARAM.errt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/ut_traj_online_no_ster.txt", 'wb') as f:
            pickle.dump(PARAM.ut_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/rewardt_traj_online_no_ster.txt", 'wb') as f:
            pickle.dump(PARAM.rewardt_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/average_step_reward_traj_online_no_ster.txt", 'wb') as f:
            pickle.dump(PARAM.Expected_step_reward_Online_list, f)
        

        with open("./Data/Traj Recording/Online Learning/lumped_disturbance_online_no_ster.txt", 'wb') as f:
            pickle.dump(PARAM.distrubance_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/periodic_disturbance_online_no_ster.txt", 'wb') as f:
            pickle.dump(PARAM.periodic_distur_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/pump_disturbance_online_no_ster.txt", 'wb') as f:
            pickle.dump(PARAM.pump_distur_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/disturbance_indices_online_no_ster.txt", 'wb') as f:
            pickle.dump(PARAM.Online_Noise_Index, f)


        with open("./Data/Traj Recording/Online Learning/total_reward_online_no_ster.txt", 'wb') as f:
            pickle.dump(PARAM.Total_Reward_Record_Online_list, f)
        with open("./Data/Traj Recording/Online Learning/variable_batch_length_online_no_ster.txt", 'wb') as f:
            pickle.dump(PARAM.Variable_Batch_Length_Online_list, f)      
        with open("./Data/Traj Recording/Online Learning/variable_initial_state_online_no_ster.txt", 'wb') as f:
            pickle.dump(PARAM.Variable_Initial_State_Online_list, f)
        
        print("// ************************************************************************* //")
        print("Online Learning trajectories in No_STER Mode have been saved in 12 files!")
        print("// ************************************************************************* //")
        
    
    def __load_Traj_Online_No_STER__(self, PARAM):
        with open("./Data/Traj Recording/Online Learning/yt_traj_online_no_ster.txt", 'rb') as f:
            PARAM.yt_traj_Online_deque = pickle.load(f)     
        with open("./Data/Traj Recording/Online Learning/yrt_traj_online_no_ster.txt", 'rb') as f:
            PARAM.yrt_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/errt_traj_online_no_ster.txt", 'rb') as f:
            PARAM.errt_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/ut_traj_online_no_ster.txt", 'rb') as f:
            PARAM.ut_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/rewardt_traj_online_no_ster.txt", 'rb') as f:
            PARAM.rewardt_Online_deque = pickle.load(f)      
        with open("./Data/Traj Recording/Online Learning/average_step_reward_traj_online_no_ster.txt", 'rb') as f:
            PARAM.Expected_step_reward_Online_list = pickle.load(f)
        

        with open("./Data/Traj Recording/Online Learning/lumped_disturbance_online_no_ster.txt", 'rb') as f:
            PARAM.distrubance_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/periodic_disturbance_online_no_ster.txt", 'rb') as f:
            PARAM.periodic_distur_Online_deque = pickle.load(f)       
        with open("./Data/Traj Recording/Online Learning/pump_disturbance_online_no_ster.txt", 'rb') as f:
            PARAM.pump_distur_Online_deque = pickle.load(f)      
        with open("./Data/Traj Recording/Online Learning/disturbance_indices_online_no_ster.txt", 'rb') as f:
            PARAM.Online_Noise_Index = pickle.load(f)
        

        with open("./Data/Traj Recording/Online Learning/total_reward_online_no_ster.txt", 'rb') as f:
            PARAM.Total_Reward_Record_Online_list = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/variable_batch_length_online_no_ster.txt", 'rb') as f:
            PARAM.Variable_Batch_Length_Online_list = pickle.load(f)       
        with open("./Data/Traj Recording/Online Learning/variable_initial_state_online_no_ster.txt", 'rb') as f:
            PARAM.Variable_Initial_State_Online_list = pickle.load(f)

        print("// ************************************************************************* //")
        print("Online Learning trajectories in No_STER Mode have been loaded!")
        print("// ************************************************************************* //")  
#* ------------------------------------------------------------------------------------- *#



#* ------------------------------------------------------------------------------------- *# 
#? TSTA-DDPG Model Mismatch Online Learning
    def __save_Traj_Online_Mismatch__(self, PARAM):
        with open("./Data/Traj Recording/Online Learning/yt_traj_online_mismatch.txt", 'wb') as f:
            pickle.dump(PARAM.yt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/yrt_traj_online_mismatch.txt", 'wb') as f:
            pickle.dump(PARAM.yrt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/errt_traj_online_mismatch.txt", 'wb') as f:
            pickle.dump(PARAM.errt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/ut_traj_online_mismatch.txt", 'wb') as f:
            pickle.dump(PARAM.ut_traj_Online_deque, f)

        with open("./Data/Traj Recording/Online Learning/rewardt_traj_online_mismatch.txt", 'wb') as f:
            pickle.dump(PARAM.rewardt_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/average_step_reward_traj_online_mismatch.txt", 'wb') as f:
            pickle.dump(PARAM.Expected_step_reward_Online_list, f)
        

        with open("./Data/Traj Recording/Online Learning/lumped_disturbance_online_mismatch.txt", 'wb') as f:
            pickle.dump(PARAM.distrubance_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/periodic_disturbance_online_mismatch.txt", 'wb') as f:
            pickle.dump(PARAM.periodic_distur_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/pump_disturbance_online_mismatch.txt", 'wb') as f:
            pickle.dump(PARAM.pump_distur_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/disturbance_indices_online_mismatch.txt", 'wb') as f:
            pickle.dump(PARAM.Online_Noise_Index, f)


        with open("./Data/Traj Recording/Online Learning/total_reward_online_mismatch.txt", 'wb') as f:
            pickle.dump(PARAM.Total_Reward_Record_Online_list, f)
        with open("./Data/Traj Recording/Online Learning/variable_batch_length_online_mismatch.txt", 'wb') as f:
            pickle.dump(PARAM.Variable_Batch_Length_Online_list, f)        
        with open("./Data/Traj Recording/Online Learning/variable_initial_state_online_mismatch.txt", 'wb') as f:
            pickle.dump(PARAM.Variable_Initial_State_Online_list, f)
        
        print("// ************************************************************************* //")
        print("Online Learning trajectories in Mismatch Mode have been saved in 12 files!")
        print("// ************************************************************************* //")
        
    
    def __load_Traj_Online_Mismatch__(self, PARAM):
        with open("./Data/Traj Recording/Online Learning/yt_traj_online_mismatch.txt", 'rb') as f:
            PARAM.yt_traj_Online_deque = pickle.load(f)      
        with open("./Data/Traj Recording/Online Learning/yrt_traj_online_mismatch.txt", 'rb') as f:
            PARAM.yrt_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/errt_traj_online_mismatch.txt", 'rb') as f:
            PARAM.errt_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/ut_traj_online_mismatch.txt", 'rb') as f:
            PARAM.ut_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/rewardt_traj_online_mismatch.txt", 'rb') as f:
            PARAM.rewardt_Online_deque = pickle.load(f)      
        with open("./Data/Traj Recording/Online Learning/average_step_reward_traj_online_mismatch.txt", 'rb') as f:
            PARAM.Expected_step_reward_Online_list = pickle.load(f)


        with open("./Data/Traj Recording/Online Learning/lumped_disturbance_online_mismatch.txt", 'rb') as f:
            PARAM.distrubance_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/periodic_disturbance_online_mismatch.txt", 'rb') as f:
            PARAM.periodic_distur_Online_deque = pickle.load(f)      
        with open("./Data/Traj Recording/Online Learning/pump_disturbance_online_mismatch.txt", 'rb') as f:
            PARAM.pump_distur_Online_deque = pickle.load(f)    
        with open("./Data/Traj Recording/Online Learning/disturbance_indices_online_mismatch.txt", 'rb') as f:
            PARAM.Online_Noise_Index = pickle.load(f)
        

        with open("./Data/Traj Recording/Online Learning/total_reward_online_mismatch.txt", 'rb') as f:
            PARAM.Total_Reward_Record_Online_list = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/variable_batch_length_online_mismatch.txt", 'rb') as f:
            PARAM.Variable_Batch_Length_Online_list = pickle.load(f)      
        with open("./Data/Traj Recording/Online Learning/variable_initial_state_online_mismatch.txt", 'rb') as f:
            PARAM.Variable_Initial_State_Online_list = pickle.load(f)

        print("// ************************************************************************* //")
        print("Online Learning trajectories in Mismatch Mode have been loaded!")
        print("// ************************************************************************* //")  
#* ------------------------------------------------------------------------------------- *#





#* ------------------------------------------------------------------------------------- *# 
#? TSTA-DDPG Noise_Expo Online Learning
    def __save_Traj_Online_Noise_Expo__(self, PARAM):
        with open("./Data/Traj Recording/Online Learning/yt_traj_online_noise_expo.txt", 'wb') as f:
            pickle.dump(PARAM.yt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/yrt_traj_online_noise_expo.txt", 'wb') as f:
            pickle.dump(PARAM.yrt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/errt_traj_online_noise_expo.txt", 'wb') as f:
            pickle.dump(PARAM.errt_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/ut_traj_online_noise_expo.txt", 'wb') as f:
            pickle.dump(PARAM.ut_traj_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/rewardt_traj_online_noise_expo.txt", 'wb') as f:
            pickle.dump(PARAM.rewardt_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/average_step_reward_traj_online_noise_expo.txt", 'wb') as f:
            pickle.dump(PARAM.Expected_step_reward_Online_list, f)
        

        with open("./Data/Traj Recording/Online Learning/lumped_disturbance_online_noise_expo.txt", 'wb') as f:
            pickle.dump(PARAM.distrubance_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/periodic_disturbance_online_noise_expo.txt", 'wb') as f:
            pickle.dump(PARAM.periodic_distur_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/pump_disturbance_online_noise_expo.txt", 'wb') as f:
            pickle.dump(PARAM.pump_distur_Online_deque, f)
        with open("./Data/Traj Recording/Online Learning/disturbance_indices_online_noise_expo.txt", 'wb') as f:
            pickle.dump(PARAM.Online_Noise_Index, f)


        with open("./Data/Traj Recording/Online Learning/total_reward_online_noise_expo.txt", 'wb') as f:
            pickle.dump(PARAM.Total_Reward_Record_Online_list, f)
        with open("./Data/Traj Recording/Online Learning/variable_batch_length_online_noise_expo.txt", 'wb') as f:
            pickle.dump(PARAM.Variable_Batch_Length_Online_list, f)     
        with open("./Data/Traj Recording/Online Learning/variable_initial_state_online_noise_expo.txt", 'wb') as f:
            pickle.dump(PARAM.Variable_Initial_State_Online_list, f)

        
        print("// ************************************************************************* //")
        print("Online Learning trajectories in Noise_Expo Mode have been saved in 12 files!")
        print("// ************************************************************************* //")
        
    
    def __load_Traj_Online_Noise_Expo__(self, PARAM):
        with open("./Data/Traj Recording/Online Learning/yt_traj_online_noise_expo.txt", 'rb') as f:
            PARAM.yt_traj_Online_deque = pickle.load(f)      
        with open("./Data/Traj Recording/Online Learning/yrt_traj_online_noise_expo.txt", 'rb') as f:
            PARAM.yrt_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/errt_traj_online_noise_expo.txt", 'rb') as f:
            PARAM.errt_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/ut_traj_online_noise_expo.txt", 'rb') as f:
            PARAM.ut_traj_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/rewardt_traj_online_noise_expo.txt", 'rb') as f:
            PARAM.rewardt_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/average_step_reward_traj_online_noise_expo.txt", 'rb') as f:
            PARAM.Expected_step_reward_Online_list = pickle.load(f)
        

        with open("./Data/Traj Recording/Online Learning/lumped_disturbance_online_noise_expo.txt", 'rb') as f:
            PARAM.distrubance_Online_deque = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/periodic_disturbance_online_noise_expo.txt", 'rb') as f:
            PARAM.periodic_distur_Online_deque = pickle.load(f)       
        with open("./Data/Traj Recording/Online Learning/pump_disturbance_online_noise_expo.txt", 'rb') as f:
            PARAM.pump_distur_Online_deque = pickle.load(f)       
        with open("./Data/Traj Recording/Online Learning/disturbance_indices_online_noise_expo.txt", 'rb') as f:
            PARAM.Online_Noise_Index = pickle.load(f)
        

        with open("./Data/Traj Recording/Online Learning/total_reward_online_noise_expo.txt", 'rb') as f:
            PARAM.Total_Reward_Record_Online_list = pickle.load(f)
        with open("./Data/Traj Recording/Online Learning/variable_batch_length_online_noise_expo.txt", 'rb') as f:
            PARAM.Variable_Batch_Length_Online_list = pickle.load(f)      
        with open("./Data/Traj Recording/Online Learning/variable_initial_state_online_noise_expo.txt", 'rb') as f:
            PARAM.Variable_Initial_State_Online_list = pickle.load(f)

        print("// ************************************************************************* //")
        print("Online Learning trajectories in Noise_Expo Mode have been loaded!")
        print("// ************************************************************************* //")  
#* ------------------------------------------------------------------------------------- *#































#* ------------------------------------------------------------------------------------- *#
#? ILC Zero Modification
    def __save_Traj_ILC_Zero__(self, PARAM):
        with open("./Data/Traj Recording/ILC Zero/yt_traj_ilc_zero.txt", 'wb') as f:
            pickle.dump(PARAM.yt_ilc_zero_traj_deque, f)
        
        with open("./Data/Traj Recording/ILC Zero/yrt_traj_ilc_zero.txt", 'wb') as f:
            pickle.dump(PARAM.yrt_ilc_zero_traj_deque, f)
        
        with open("./Data/Traj Recording/ILC Zero/errt_traj_ilc_zero.txt", 'wb') as f:
            pickle.dump(PARAM.err_ilc_zero_traj_deque, f)

        with open("./Data/Traj Recording/ILC Zero/ut_traj_ilc_zero.txt", 'wb') as f:
            pickle.dump(PARAM.ut_ilc_zero_traj_deque, f)
   
        print("// ************************************************************************* //")
        print("ILC Zero-modification trajectories have been saved in 4 seperate files!")
        print("// ************************************************************************* //")
        pass

    def __load_Traj_ILC_Zero__(self, PARAM):
        with open("./Data/Traj Recording/ILC Zero/yt_traj_ilc_zero.txt", 'rb') as f:
            PARAM.yt_ilc_zero_traj_deque = pickle.load(f)

        with open("./Data/Traj Recording/ILC Zero/yrt_traj_ilc_zero.txt", 'rb') as f:
            PARAM.yrt_ilc_zero_traj_deque = pickle.load(f)

        with open("./Data/Traj Recording/ILC Zero/errt_traj_ilc_zero.txt", 'rb') as f:
            PARAM.err_ilc_zero_traj_deque = pickle.load(f)

        with open("./Data/Traj Recording/ILC Zero/ut_traj_ilc_zero.txt", 'rb') as f:
            PARAM.ut_ilc_zero_traj_deque = pickle.load(f)
   
        print("// ************************************************************************* //")
        print("ILC Zero-modification trajectories have been loaded!")
        print("// ************************************************************************* //")
        pass
#* ------------------------------------------------------------------------------------- *#

#* ------------------------------------------------------------------------------------- *#
#? ILC Nearest Modification
    def __save_Traj_ILC_Near__(self, PARAM):
        with open("./Data/Traj Recording/ILC Near/yt_traj_ilc_near.txt", 'wb') as f:
            pickle.dump(PARAM.yt_ilc_near_traj_deque, f)
        
        with open("./Data/Traj Recording/ILC Near/yrt_traj_ilc_near.txt", 'wb') as f:
            pickle.dump(PARAM.yrt_ilc_near_traj_deque, f)
        
        with open("./Data/Traj Recording/ILC Near/errt_traj_ilc_near.txt", 'wb') as f:
            pickle.dump(PARAM.err_ilc_near_traj_deque, f)

        with open("./Data/Traj Recording/ILC Near/ut_traj_ilc_near.txt", 'wb') as f:
            pickle.dump(PARAM.ut_ilc_near_traj_deque, f)
   
        print("// ************************************************************************* //")
        print("ILC Nearest-modification trajectories have been saved in 4 seperate files!")
        print("// ************************************************************************* //")
        pass

    def __load_Traj_ILC_Near__(self, PARAM):
        with open("./Data/Traj Recording/ILC Near/yt_traj_ilc_near.txt", 'rb') as f:
            PARAM.yt_ilc_near_traj_deque = pickle.load(f)

        with open("./Data/Traj Recording/ILC Near/yrt_traj_ilc_near.txt", 'rb') as f:
            PARAM.yrt_ilc_near_traj_deque = pickle.load(f)

        with open("./Data/Traj Recording/ILC Near/errt_traj_ilc_near.txt", 'rb') as f:
            PARAM.err_ilc_near_traj_deque = pickle.load(f)

        with open("./Data/Traj Recording/ILC Near/ut_traj_ilc_near.txt", 'rb') as f:
            PARAM.ut_ilc_near_traj_deque = pickle.load(f)
   
        print("// ************************************************************************* //")
        print("ILC Nearest-modification trajectories have been loaded!")
        print("// ************************************************************************* //")
        pass
#* ------------------------------------------------------------------------------------- *#



    def __cal_RMSE__(self, temp_list):
        squ_err = list(e**2 for e in temp_list)
        mse = sum(squ_err) / len(squ_err)
        rmse = math.sqrt(mse)
        # rmse = np.round(rmse, 4)
        
        return rmse
    
    def __cal_Max_ABS_Err__(self, temp_list):
        abs_err = list(np.abs(e) for e in temp_list)
        max_err = max(abs_err)[0]
        # max_err = np.round(max_err, 4)

        return max_err



# ============================================== End of File  =============================================== #

