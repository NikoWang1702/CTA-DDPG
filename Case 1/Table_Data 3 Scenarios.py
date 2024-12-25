import numpy as np
import pandas as pd

from Utilities import *



PARAM = Sys_Init()
Agent_Table = DDPG_Plot()

PARAM.Phase_Index = 0
PARAM.Scen_Index  = 3
Agent_Table.__load_Traj_Online_No_STER__(PARAM)
Agent_Table.__load_Traj_ILC_Zero__(PARAM)
Agent_Table.__load_Traj_ILC_Near__(PARAM)

rmse_regular      = []
max_err_regular   = []

rmse_near         = []
max_err_near      = []

rmse_zero         = []
max_err_zero      = []

batch_length = PARAM.errt_traj_Online_deque.__len__()

for i in range(batch_length):
    err_traj_list = PARAM.errt_traj_Online_deque[i]
    rmse_regular.append(Agent_Table.__cal_RMSE__(err_traj_list))
    max_err_regular.append(Agent_Table.__cal_Max_ABS_Err__(err_traj_list))

    err_traj_list = PARAM.err_ilc_near_traj_deque[i]
    rmse_near.append(Agent_Table.__cal_RMSE__(err_traj_list))
    max_err_near.append(Agent_Table.__cal_Max_ABS_Err__(err_traj_list))

    err_traj_list = PARAM.err_ilc_zero_traj_deque[i]
    rmse_zero.append(Agent_Table.__cal_RMSE__(err_traj_list))
    max_err_zero.append(Agent_Table.__cal_Max_ABS_Err__(err_traj_list))

mean_rmse_regular = sum(rmse_regular) / len(rmse_regular)
IAE_Regular  = sum(max_err_regular)
IIAE_Regular = sum([(max_err_regular.index(e)+1)*e for e in max_err_regular])
max_err_squre_regular = [e**2 for e in max_err_regular]
ISE_Regular  = sum(max_err_squre_regular)
IISE_Regular = sum([(max_err_squre_regular.index(e)+1)*e for e in max_err_squre_regular])

mean_rmse_near = sum(rmse_near) / len(rmse_near)
IAE_Near  = sum(max_err_near)
IIAE_Near = sum([(max_err_near.index(e)+1)*e for e in max_err_near])
max_err_squre_near = [e**2 for e in max_err_near]
ISE_Near  = sum(max_err_squre_near)
IISE_Near = sum([(max_err_squre_near.index(e)+1)*e for e in max_err_squre_near])

mean_rmse_zero = sum(rmse_zero) / len(rmse_zero)
IAE_Zero  = sum(max_err_zero)
IIAE_Zero = sum([(max_err_zero.index(e)+1)*e for e in max_err_zero])
max_err_squre_zero = [e**2 for e in max_err_zero]
ISE_Zero  = sum(max_err_squre_zero)
IISE_Zero = sum([(max_err_squre_zero.index(e)+1)*e for e in max_err_squre_zero])



batch_indices = np.arange(1, len(rmse_regular)+1)

df_rmse_regular = pd.DataFrame({'Online Episode':batch_indices, 'RMSE Regular': rmse_regular})
df_mae_regular  = pd.DataFrame({'MAE': max_err_regular})
df_mean_regular = pd.DataFrame({'Mean_RMSE_Regular': mean_rmse_regular}, index=[0])
df_iae_regular  = pd.DataFrame({'IAE': IAE_Regular}, index=[0])
df_iiae_regular = pd.DataFrame({'IIAE': IIAE_Regular}, index=[0])
df_ise_regular  = pd.DataFrame({'ISE': ISE_Regular}, index=[0])
df_iise_regular = pd.DataFrame({'IISE': IISE_Regular}, index=[0])
df_regular = pd.concat([df_rmse_regular, df_mae_regular, df_mean_regular, df_iae_regular, df_iiae_regular, df_ise_regular, df_iise_regular], axis=1)

df_rmse_near = pd.DataFrame({'Online Episode':batch_indices, 'RMSE Near': rmse_near})
df_mae_near  = pd.DataFrame({'MAE': max_err_near})
df_mean_near = pd.DataFrame({'Mean_RMSE_Near': mean_rmse_near}, index=[0])
df_iae_near  = pd.DataFrame({'IAE':IAE_Near}, index=[0])
df_iiae_near = pd.DataFrame({'IIAE':IIAE_Near}, index=[0])
df_ise_near  = pd.DataFrame({'ISE':ISE_Near}, index=[0])
df_iise_near = pd.DataFrame({'IISE':IISE_Near}, index=[0])
df_near = pd.concat([df_rmse_near, df_mae_near, df_mean_near, df_iae_near, df_iiae_near, df_ise_near, df_iise_near], axis=1)

df_rmse_zero = pd.DataFrame({'Online Episode':batch_indices, 'RMSE Zero': rmse_zero})
df_mae_zero  = pd.DataFrame({'MAE': max_err_zero})
df_mean_zero = pd.DataFrame({'Mean_RMSE_Zero': mean_rmse_zero}, index=[0])
df_iae_zero  = pd.DataFrame({'IAE':IAE_Zero}, index=[0])
df_iiae_zero = pd.DataFrame({'IIAE':IIAE_Zero}, index=[0])
df_ise_zero  = pd.DataFrame({'ISE':ISE_Zero}, index=[0])
df_iise_zero = pd.DataFrame({'IISE':IISE_Zero}, index=[0])
df_zero = pd.concat([df_rmse_zero, df_mae_zero, df_mean_zero, df_iae_zero, df_iiae_zero, df_ise_zero, df_iise_zero], axis=1)

df_err_related = pd.concat([df_regular, df_mae_zero, df_near, df_zero], axis=1)

df_err_related.to_csv('./Data/Raw Data CSV/RMSE_Related in Scenario {}.csv'.format(PARAM.Scen_Index), mode='w', index=False, encoding='utf_8_sig')








