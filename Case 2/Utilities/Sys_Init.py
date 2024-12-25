import numpy as np

from collections import deque



class Sys_Init():

    def __init__(self):
        #* Online Stochastic Noise Index
        self.Online_Noise_Index = []    # e.g. 15 fixed time points

        self.Phase_Index = 0
        self.Scen_Index  = 1

    #@ --------------- Plot and Comparasion related parameters -------------- @# 
        self.reward_Offline_list            = []        
        self.reward_Offline_deque           = deque()

        self.err_Offline_list               = []        
        self.err_Offline_deque              = deque()
        


        #* Offline traj.
        self.yt_traj_Offline_list           = []
        self.yrt_traj_Offline_list          = []
        self.errt_traj_Offline_list         = []
        self.ut_traj_Offline_list           = []
        self.reward_traj_Offline_list       = []

        self.episode_traj_offline_list     = []

        self.yt_traj_Offline_deque          = deque()
        self.yrt_traj_Offline_deque         = deque()
        self.errt_traj_Offline_deque        = deque()
        self.ut_traj_Offline_deque          = deque()
        self.reward_traj_Offline_deque      = deque()


        #* Online traj.
        self.yt_traj_Online_list            = []
        self.yrt_traj_Online_list           = []
        self.errt_traj_Online_list          = []
        self.ut_traj_Online_list            = []
        self.rewardt_Online_list            = []

        self.disturbance_Online_list        = []        # disturbance
        self.periodic_distur_Online_list    = []        # periodic disturbance
        self.pump_distur_Online_list        = []        # pump disturbance

        self.P1_Online_list      = []
        self.P2_Online_list      = []
        self.Q1_Online_list      = []
        self.Q2_Online_list      = []
        self.action1_Online_list = []
        self.action2_Online_list = []
        self.action_Online_list  = []


        self.yt_traj_Online_deque       = deque()
        self.yrt_traj_Online_deque      = deque()
        self.errt_traj_Online_deque     = deque()       # max rmse
        self.ut_traj_Online_deque       = deque()       # u_t
        self.rewardt_Online_deque       = deque()    
        
        self.distrubance_Online_deque     = deque()       # disturbance
        self.periodic_distur_Online_deque = deque()
        self.pump_distur_Online_deque     = deque()

        self.P1_Online_deque = deque()
        self.P2_Online_deque = deque()
        self.Q1_Online_deque = deque()
        self.Q2_Online_deque = deque()
        self.action1_Online_deque = deque()
        self.action2_Online_deque = deque()
        self.action_Online_deque = deque()


        self.Total_Reward_Record_Online_list    = []

        self.Variable_Batch_Length_Online_list  = []    # batch length

        self.Expected_step_reward_Online_list   = []

        self.Variable_Initial_State_Online_list = []    # initial state x0(2)

    #@ -------------------------- Debug related parameters ------------------------- @#
        # self.action_1_list = []
        # self.action_2_list = []
        # self.action_hybrid = []
        # self.P_1_list      = []
        # self.P_2_list      = []
        self.tau_hybrid_list = []


# ------------------------------------ original ILC ------------------------------------ #
        # tiem axis
        self.xt_ilc_list    = []
        self.ut_ilc_list    = []
        self.yt_ilc_list    = []
        self.yrt_ilc_list   = []
        self.err_ilc_list   = []

        self.xt_ilc_traj_list    = []
        self.ut_ilc_traj_list    = []
        self.yt_ilc_traj_list    = []
        self.yrt_ilc_traj_list   = []
        self.err_ilc_traj_list   = []

        # batch axis
        self.xt_ilc_deque   = deque()
        self.ut_ilc_deque   = deque()
        self.yt_ilc_deque   = deque()
        self.yrt_ilc_deque  = deque()
        self.err_ilc_deque  = deque()

        self.xt_ilc_traj_deque   = deque()
        self.ut_ilc_traj_deque   = deque()
        self.yt_ilc_traj_deque   = deque()
        self.yrt_ilc_traj_deque  = deque()
        self.err_ilc_traj_deque  = deque()

# ------------------------------------ zero-modification ILC ------------------------------------ #
        # tiem axis
        self.xt_ilc_zero_list    = []
        self.ut_ilc_zero_list    = []
        self.yt_ilc_zero_list    = []
        self.yrt_ilc_zero_list   = []
        self.err_ilc_zero_list   = []

        self.xt_ilc_zero_traj_list    = []
        self.ut_ilc_zero_traj_list    = []
        self.yt_ilc_zero_traj_list    = []
        self.yrt_ilc_zero_traj_list   = []
        self.err_ilc_zero_traj_list   = []

        # batch axis
        self.xt_ilc_zero_deque   = deque()
        self.ut_ilc_zero_deque   = deque()
        self.yt_ilc_zero_deque   = deque()
        self.yrt_ilc_zero_deque  = deque()
        self.err_ilc_zero_deque  = deque()

        self.xt_ilc_zero_traj_deque   = deque()
        self.ut_ilc_zero_traj_deque   = deque()
        self.yt_ilc_zero_traj_deque   = deque()
        self.yrt_ilc_zero_traj_deque  = deque()
        self.err_ilc_zero_traj_deque  = deque()

# ------------------------------------ near-modification ILC ------------------------------------ #
        # tiem axis
        self.xt_ilc_near_list    = []
        self.ut_ilc_near_list    = []
        self.yt_ilc_near_list    = []
        self.yrt_ilc_near_list   = []
        self.err_ilc_near_list   = []

        self.xt_ilc_near_traj_list    = []
        self.ut_ilc_near_traj_list    = []
        self.yt_ilc_near_traj_list    = []
        self.yrt_ilc_near_traj_list   = []
        self.err_ilc_near_traj_list   = []

        # batch axis
        self.xt_ilc_near_deque   = deque()
        self.ut_ilc_near_deque   = deque()
        self.yt_ilc_near_deque   = deque()
        self.yrt_ilc_near_deque  = deque()
        self.err_ilc_near_deque  = deque()

        self.xt_ilc_near_traj_deque   = deque()
        self.ut_ilc_near_traj_deque   = deque()
        self.yt_ilc_near_traj_deque   = deque()
        self.yrt_ilc_near_traj_deque  = deque()
        self.err_ilc_near_traj_deque  = deque()

#* --------------------------------------------------------------------------------------------------------------------------------
#* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# ============================================== End of File  =============================================== #
