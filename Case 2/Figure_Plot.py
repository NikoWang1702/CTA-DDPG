from Table_Data import PARAM_No_STER
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from collections import deque

from Utilities import *
import math

np.random.seed(19980823)

#------------------------------------------------------------------------------------------------------------------
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# font dict
font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size'  :  10,}


# ******************************************************************************************************************
# todo: Fig.14：Tracking performance during training in the offline phase.
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()

Agent_Plot.__load_Traj_Offline_forFigure__(PARAM)

batch_length = PARAM.err_Offline_deque.__len__()

rmse_offline = []
for i in range(batch_length):
    err_traj_list = PARAM.err_Offline_deque[i]
    rmse_offline.append(Agent_Plot.__cal_RMSE__(err_traj_list))

xticks = np.arange(1, batch_length+1)

fig2 = plt.figure(figsize=(7.2, 4))
ax2  = fig2.add_subplot()

ax2.plot(xticks, rmse_offline, label='RMSE through Training', \
         linestyle='-', color='#546de5', linewidth=1.2, dash_capstyle='round', zorder=1)

ax2.set_xlim([-50, batch_length+50])
plt.xlabel("Training Episode", family='Arial', size=12, weight='normal', labelpad=7)

plt.ylabel("Tracking RMSE", family='Arial', size=12, weight='normal', labelpad=5)

ax2.tick_params(axis='both', direction="in", right=True, top=True)


# todo: enlarged figure
axins = ax2.inset_axes((0.15, 0.3, 0.8, 0.6))
axins.plot(xticks, rmse_offline, linestyle='-', color='#546de5', linewidth=0.8, dash_capstyle='round')

zone_left = 1800
zone_right = 2999

x_ratio = 0.05
y_ratio = 0.05

xlim0 = xticks[zone_left]-(xticks[zone_right]-xticks[zone_left])*x_ratio
xlim1 = xticks[zone_right]+(xticks[zone_right]-xticks[zone_left])*x_ratio

y = np.hstack((rmse_offline[zone_left:zone_right]))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

axins.set_yticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35])

rect = plt.Rectangle((1800, -2), 1200, 4.5, fill=False, color='#f78fb3')
ax2.add_patch(rect)
ax2.annotate('', xy=(2250, 25), xytext=(2350, 5), \
             arrowprops=dict(facecolor='black', width=1.5, headwidth=6.5, headlength=8, shrink=0.05, alpha=0.8), \
             fontproperties = 'Arial', size = 11)

plt.savefig('./Data/Figures/Fig14 - Tracking performance during training in the offline phase.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.18：Tracking RMSE along batches during the online stage.
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()

Agent_Plot.__load_Traj_Online_Mismatch__(PARAM)
Agent_Plot.__load_Traj_ILC_Zero__(PARAM)
Agent_Plot.__load_Traj_ILC_Near__(PARAM)

batch_length = PARAM.errt_traj_Online_deque.__len__()

rmse_online = []
for i in range(batch_length):
    err_traj_list = PARAM.errt_traj_Online_deque[i]
    rmse_online.append(Agent_Plot.__cal_RMSE__(err_traj_list))

rmse_ilc_zero = []
for i in range(batch_length):
    err_traj_list = PARAM.err_ilc_zero_traj_deque[i]
    rmse_ilc_zero.append(Agent_Plot.__cal_RMSE__(err_traj_list))

rmse_ilc_near = []
for i in range(batch_length):
    err_traj_list = PARAM.err_ilc_near_traj_deque[i]
    rmse_ilc_near.append(Agent_Plot.__cal_RMSE__(err_traj_list))   

xticks = np.arange(1, batch_length+1)

fig7 = plt.figure(figsize=(7.2, 4))
ax8  = fig7.add_subplot()

ax8.plot(xticks, rmse_online, label='CTA-DDPG', \
         linestyle='-', color='#EE1F23', linewidth=1.5, dash_capstyle='round', zorder=1)

ax8.plot(xticks, rmse_ilc_zero, label='Zero ILC', \
         linestyle='--', color='#0D7F3F', linewidth=1.5, dash_capstyle='round', zorder=2)

ax8.plot(xticks, rmse_ilc_near, label='Nearest ILC', \
         linestyle='-.', color='#0B72BA', linewidth=1.5, dash_capstyle='round', zorder=3)

ax8.legend(prop=font_legend, frameon=True, loc='upper right')

ax8.set_xlim([0, batch_length+0.5])

ax8.xaxis.set_major_locator(MultipleLocator(5))
plt.xlabel("Batch ($k$)", family='Arial', size=12, weight='normal', labelpad=7)

plt.ylabel("Tracking RMSE", family='Arial', size=12, weight='normal', labelpad=5)

ax8.tick_params(axis='both', direction="in", right=True, top=True)

plt.savefig('./Data/Figures/Fig18 - Tracking RMSE along batches during the online stage.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.15-1：Tracking trajectories of injection velocity vz in Case 2 (CTA-DDPG).
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()
Agent_Plot.__load_Traj_Online_Regular__(PARAM)

fig8 = plt.figure(figsize=(7.2, 6.5))
ax9 = fig8.add_subplot(111, projection='3d')

ax9.grid(False)
ax9.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax9.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax9.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax9.get_proj = lambda: np.dot(Axes3D.get_proj(ax9), np.diag([1.2, 0.8, 0.8, 1]))

colors = ['#ffa502', '#34e7e4', '#2ed573', '#f368e0', \
          '#ff4757', '#747d8c', '#0652DD', '#e15f41']

yticks = list(np.arange(1, 51, 7))

for c,k in zip(colors, yticks):
    yt  = list(PARAM.yt_traj_Online_deque[k-1])
    yrt = list(PARAM.yrt_traj_Online_deque[k-1])
    if len(yt) == len(yrt):     
        t = len(yt)
    else:
        print("yt not match with yrt!")
    ts = np.arange(0,t)
    ax9.plot(ts, yrt, zs=k, zdir='y', linestyle='--',color='dimgray', dash_capstyle='round', alpha=0.8, linewidth=2.7)
    ax9.plot(ts, yt, zs=k, zdir='y',  linestyle='-', color=c, dash_capstyle='round', alpha=0.7, linewidth=1.7)


ax9.set_xlabel('Time ($t$)', family='Arial', size=12, weight='normal', labelpad=10)
ax9.set_ylabel('Batch ($k$)', family='Arial', size=12, weight='normal', labelpad=5)
ax9.set_zlabel('$v_{z}$', family='Arial', size=12, weight='normal', labelpad=5)

ax9.zaxis._axinfo['juggled'] = (1,2,1)
ax9.zaxis.set_rotate_label(0)

ax9.set_zlim(0, 3)

ax9.set_yticks(yticks)

ax9.xaxis.set_major_locator(MultipleLocator(200))
ax9.zaxis.set_major_locator(MultipleLocator(0.5))

ax9.view_init(16, -44)

fig8.tight_layout()
fig8.subplots_adjust(left=0.1)  # plot outside the normal area
plt.savefig("./Data/Figures/Fig15-1 - Tracking trajectories of injection velocity vz in Case 2 (CTA-DDPG).png", dpi=300)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.15-2：Tracking trajectories of injection velocity vz in Case 2 (Zero ILC)
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()

Agent_Plot.__load_Traj_ILC_Zero__(PARAM)

fig9 = plt.figure(figsize=(7.2, 6.5))
ax10 = fig9.add_subplot(111, projection='3d')

ax10.grid(False)
ax10.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax10.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax10.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax10.get_proj = lambda: np.dot(Axes3D.get_proj(ax10), np.diag([1.2, 0.8, 0.8, 1]))

colors = ['#ffa502', '#34e7e4', '#2ed573', '#f368e0', \
          '#ff4757', '#747d8c', '#0652DD', '#e15f41']

yticks = list(np.arange(1, 51, 7))

for c,k in zip(colors, yticks):
    yt  = list(PARAM.yt_ilc_zero_traj_deque[k-1])
    yrt = list(PARAM.yrt_ilc_zero_traj_deque[k-1])

    if len(yt) == len(yrt):     
        t = len(yt)
    else:
        print("yt not match with yrt!")
    ts = np.arange(0,t)
    # Plot
    ax10.plot(ts, yrt, zs=k, zdir='y', linestyle='--',color='dimgray', dash_capstyle='round', alpha=0.8, linewidth=2.7)
    ax10.plot(ts, yt, zs=k, zdir='y',  linestyle='-', color=c, dash_capstyle='round', alpha=0.7, linewidth=1.7)


ax10.set_xlabel('Time ($t$)', family='Arial', size=12, weight='normal', labelpad=10)
ax10.set_ylabel('Batch ($k$)', family='Arial', size=12, weight='normal', labelpad=5)
ax10.set_zlabel("$v_{z}$", family='Arial', size=12, weight='normal', labelpad=5)

ax10.zaxis._axinfo['juggled'] = (1,2,1)
ax10.zaxis.set_rotate_label(0)

ax10.set_zlim(0, 3)

ax10.set_yticks(yticks)

ax10.xaxis.set_major_locator(MultipleLocator(200))
ax10.zaxis.set_major_locator(MultipleLocator(0.5))

ax10.view_init(16, -44)

fig9.tight_layout()
fig9.subplots_adjust(left=0.1)  # plot outside the normal area
plt.savefig("./Data/Figures/Fig15-2 - Tracking trajectories of injection velocity vz in Case 2 (Zero ILC).png", dpi=300)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.15-3：Tracking trajectories of injection velocity vz in Case 2 (Near ILC).
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()
Agent_Plot.__load_Traj_ILC_Near__(PARAM)

fig10 = plt.figure(figsize=(7.2, 6.5))
ax11 = fig10.add_subplot(111, projection='3d')

ax11.grid(False)
ax11.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax11.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax11.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax11.get_proj = lambda: np.dot(Axes3D.get_proj(ax11), np.diag([1.2, 0.8, 0.8, 1]))

colors = ['#ffa502', '#34e7e4', '#2ed573', '#f368e0', \
          '#ff4757', '#747d8c', '#0652DD', '#e15f41']

yticks = list(np.arange(1, 51, 7))

for c,k in zip(colors, yticks):
    yt  = list(PARAM.yt_ilc_near_traj_deque[k-1])
    yrt = list(PARAM.yrt_ilc_near_traj_deque[k-1])
    if len(yt) == len(yrt):     
        t = len(yt)
    else:
        print("yt not match with yrt!")
    ts = np.arange(0,t)
    # Plot
    ax11.plot(ts, yrt, zs=k, zdir='y', linestyle='--',color='dimgray', dash_capstyle='round', alpha=0.8, linewidth=2.7)
    ax11.plot(ts, yt, zs=k, zdir='y',  linestyle='-', color=c, dash_capstyle='round', alpha=0.7, linewidth=1.7)


ax11.set_xlabel('Time ($t$)', family='Arial', size=12, weight='normal', labelpad=10)
ax11.set_ylabel('Batch ($k$)', family='Arial', size=12, weight='normal', labelpad=5)
ax11.set_zlabel("$v_{z}$", family='Arial', size=12, weight='normal', labelpad=5)

ax11.zaxis._axinfo['juggled'] = (1,2,1)
ax11.zaxis.set_rotate_label(0)

ax11.set_zlim(0, 3)

ax11.set_yticks(yticks)

ax11.xaxis.set_major_locator(MultipleLocator(200))
ax11.zaxis.set_major_locator(MultipleLocator(0.4))

ax11.view_init(16, -44)

fig10.tight_layout()
fig10.subplots_adjust(left=0.1)  # plot outside the normal area
plt.savefig("./Data/Figures/Fig15-3 - Tracking trajectories of injection velocity vz in Case 2 (Near ILC).png", dpi=600)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.20：Tracking performance comparison of CTA-DDPG with/withou PI and STER.
PARAM_Regular   = Sys_Init()
PARAM_No_STER   = Sys_Init()
PARAM_No_PI     = Sys_Init()
PARAM_Mismatch  = Sys_Init()

Agent_Plot = DDPG_Plot()

Agent_Plot.__load_Traj_Online_Regular__(PARAM_Regular)
Agent_Plot.__load_Traj_Online_No_STER__(PARAM_No_STER)
Agent_Plot.__load_Traj_Online_No_PI__(PARAM_No_PI)

batch_length = PARAM_Regular.errt_traj_Online_deque.__len__()

rmse_regular  = []
rmse_no_ster  = []
rmse_no_pi    = []
rmse_mismatch = []

for i in range(batch_length):
    err_traj_list = PARAM_Regular.errt_traj_Online_deque[i]
    rmse_regular.append(Agent_Plot.__cal_RMSE__(err_traj_list))

for i in range(batch_length):
    err_traj_list = PARAM_No_STER.errt_traj_Online_deque[i]
    rmse_no_ster.append(Agent_Plot.__cal_RMSE__(err_traj_list))

for i in range(batch_length):
    err_traj_list = PARAM_No_PI.errt_traj_Online_deque[i]
    rmse_no_pi.append(Agent_Plot.__cal_RMSE__(err_traj_list))

xticks = np.arange(1, batch_length+1)

fig11 = plt.figure(figsize=(7.2, 4))
ax12  = fig11.add_subplot()

ax12.plot(xticks, rmse_regular, label='CTA-DDPG', \
          linestyle='-', color='#EE1F23', linewidth=1.5, dash_capstyle='round')

ax12.plot(xticks, rmse_no_ster, label='Without STER', \
          linestyle='--', color='#0D7F3F', linewidth=1.5, dash_capstyle='round')

ax12.plot(xticks, rmse_no_pi, label='Without PI', \
          linestyle='-.', color='#0B72BA', linewidth=1.5, dash_capstyle='round')

ax12.legend(prop=font_legend, frameon=True, loc='upper left')

ax12.set_xlim([0, batch_length+1])
ax12.xaxis.set_major_locator(MultipleLocator(10))
plt.xlabel("Batch ($k$)", family='Arial', size=12, weight='normal', labelpad=7)

plt.ylabel("Tracking RMSE", family='Arial', size=12, weight='normal', labelpad=7)

ax12.tick_params(axis='both', direction="in", right=True, top=True)

axins1 = ax12.inset_axes((0.1, 0.15, 0.35, 0.35))
axins1.plot(xticks, rmse_regular, label='CTA-DDPG', \
          linestyle='-', color='#EE1F23', linewidth=0.8, dash_capstyle='round')

axins1.plot(xticks, rmse_no_ster, label='Without STER', \
          linestyle='--', color='#0D7F3F', linewidth=0.8, dash_capstyle='round')

axins1.plot(xticks, rmse_no_pi, label='Without PI', \
          linestyle='-.', color='#0B72BA', linewidth=0.8, dash_capstyle='round')


zone_left = 0
zone_right = 49

x_ratio = 0.05  
y_ratio = 0.05  

xlim0 = xticks[zone_left]-(xticks[zone_right]-xticks[zone_left])*x_ratio
xlim1 = xticks[zone_right]+(xticks[zone_right]-xticks[zone_left])*x_ratio

y1 = np.hstack((rmse_regular[zone_left:zone_right]))
y2 = np.hstack((rmse_no_ster[zone_left:zone_right]))
ylim0 = np.min(y1)-(np.max(y1)-np.min(y1))*y_ratio
ylim1 = np.max(y2)+(np.max(y2)-np.min(y2))*y_ratio

axins1.set_xlim(xlim0, xlim1)
axins1.set_ylim(ylim0-0.0005, ylim1+0.0005)

rect = plt.Rectangle((1, -5), 49, 10, fill=False, color='#f78fb3', linewidth=1.5)
ax12.add_patch(rect)
ax12.annotate('', xy=(24, 31.7), xytext=(28.5, 12), \
             arrowprops=dict(facecolor='black', width=1.5, headwidth=6.5, headlength=8, shrink=0.05, alpha=0.8), \
             fontproperties = 'Arial', size = 11)
axins1.text(0, 0.0163, 'Subgraph 2', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':10})
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# todo: enlarged figure
axins2 = ax12.inset_axes((0.35, 0.6, 0.35, 0.35))
axins2.plot(xticks, rmse_regular, label='CTA-DDPG', \
          linestyle='-', color='#EE1F23', linewidth=0.8, dash_capstyle='round')

axins2.plot(xticks, rmse_no_ster, label='Without STER', \
          linestyle='--', color='#0D7F3F', linewidth=0.8, dash_capstyle='round')

axins2.plot(xticks, rmse_no_pi, label='Without PI', \
          linestyle='-.', color='#0B72BA', linewidth=0.8, dash_capstyle='round')

zone_left = 0
zone_right = 35
x_ratio = 0.05  
y_ratio = 0.05  

xlim0 = xticks[zone_left]-(xticks[zone_right]-xticks[zone_left])*x_ratio
xlim1 = xticks[zone_right]+(xticks[zone_right]-xticks[zone_left])*x_ratio

y1 = np.hstack((rmse_regular[zone_left:zone_right]))
y2 = np.hstack((rmse_no_pi[zone_left:zone_right]))
ylim0 = np.min(y1)-(np.max(y1)-np.min(y1))*y_ratio
ylim1 = np.max(y2)+(np.max(y2)-np.min(y2))*y_ratio

axins2.set_xlim(xlim0, xlim1)
axins2.set_ylim(-0.1, 3.5)

axins2.set_yticks([0, 1, 2, 3])

ax12.annotate('', xy=(31, 70), xytext=(32, 15), \
             arrowprops=dict(facecolor='black', width=1.5, headwidth=6.5, headlength=8, shrink=0.05, alpha=0.8), \
             fontproperties = 'Arial', size = 11)
axins2.text(0, 2.9, 'Subgraph 1', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':10})
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

plt.savefig('./Data/Figures/Fig20 - Tracking performance comparison of CTA-DDPG with or withou PI and STER.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.16：Tracking error of injection velocity vz and hydraulic flowrate qh in 6 batches.
PARAM_Noise_Expo = Sys_Init()
Agent_Plot = DDPG_Plot()
Agent_Plot.__load_Traj_Online_Noise_Expo__(PARAM_Noise_Expo)
batch_length = PARAM_Noise_Expo.errt_traj_Online_deque.__len__()

fig14 = plt.figure(figsize=(7.2, 5.5))
ax16  = fig14.add_subplot(211)
ax17  = fig14.add_subplot(212)

selected_batch = [8, 12, 20, 26, 30, 37]
colors = ['#ffa502', '#34e7e4', '#2ed573', '#f368e0', '#ff4757', '#747d8c']
label = ['8th Batch', '12th Batch', '20th Batch', '26th Batch', '30th Batch', '37th Batch']
order = [1, 2, 3, 4, 5, 6]

for c,k,l,z in zip(colors, selected_batch, label, order):
    err     = list(PARAM_Noise_Expo.errt_traj_Online_deque[k-1])
    control = list(PARAM_Noise_Expo.ut_traj_Online_deque[k-1])
    if len(err) == len(control):     
        t = len(err)
    else:
        print("err mismatch with control!")
    xticks = np.arange(1,t+1)
    # Plot
    ax16.plot(xticks, err,  label=l, \
              linestyle='-.', color=c, linewidth=1.2, dash_capstyle='round', zorder=z)
    ax17.plot(xticks, control,  label=l, \
              linestyle='--', color=c, linewidth=1.2, dash_capstyle='round', zorder=z)
    
ax17.legend(prop=font_legend, frameon=True, loc='lower right', ncol=2)

ax16.set_ylabel('Tracking Error $e_{t}$', family='Arial', size=12, weight='normal', labelpad=5)
ax17.set_ylabel("Hydraulic Flowrate $q_{h}$", family='Arial', size=12, weight='normal', labelpad=5)
plt.xlabel('Time ($t$)', family='Arial', size=12, weight='normal', labelpad=7)


ax16.set_xlim([0-10, 1100])
ax17.set_xlim([0-10, 1100])
ax16.xaxis.set_major_locator(MultipleLocator(100))
ax17.xaxis.set_major_locator(MultipleLocator(100))

ax17.set_ylim([-0.15, 0.18])
ax17.yaxis.set_major_locator(MultipleLocator(0.05))

ax16.tick_params(axis='both', direction="in", right=True, top=True)
ax17.tick_params(axis='both', direction="in", right=True, top=True)

plt.savefig('./Data/Figures/Fig16 - Tracking error of injection velocity vz and hydraulic flowrate qh in 6 batches.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.17-1：Tracking trajectory of injection velocity vz in the 38th batch.
PARAM_Mismatch = Sys_Init()
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()

Agent_Plot.__load_Traj_Online_Mismatch__(PARAM_Mismatch)
Agent_Plot.__load_Traj_Online_Regular__(PARAM)
Agent_Plot.__load_Traj_ILC_Zero__(PARAM)
Agent_Plot.__load_Traj_ILC_Near__(PARAM)

selected_batch = 38

yrt         = list(PARAM.yrt_traj_Online_deque[selected_batch-1])
yt_regular  = list(PARAM.yt_traj_Online_deque[selected_batch-1])
yt_zero_ilc = list(PARAM.yt_ilc_zero_traj_deque[selected_batch-1])
yt_near_ilc = list(PARAM.yt_ilc_near_traj_deque[selected_batch-1])
yt_mimatch  = list(PARAM_Mismatch.yt_traj_Online_deque[selected_batch-1])

batch_duration = len(yrt)
xticks = np.arange(1, batch_duration+1)

fig15 = plt.figure(figsize=(7.2, 4))
ax18  = fig15.add_subplot()

ax18.plot(xticks, yrt, label='Reference Trajectory', \
          linestyle='--', color='#2f3542', linewidth=2.5, dash_capstyle='round', zorder=1)
ax18.plot(xticks, yt_regular, label='CTA-DDPG', \
          linestyle='-', color='#546de5', linewidth=1.5, dash_capstyle='round', zorder=4)
ax18.plot(xticks, yt_zero_ilc, label='Zero ILC', \
          linestyle='-', color='#63cdda', linewidth=1.5, dash_capstyle='round', zorder=2)
ax18.plot(xticks, yt_near_ilc, label='Nearest ILC', \
          linestyle='-', color='#cf6a87', linewidth=1.5, dash_capstyle='round', zorder=3)

ax18.legend(prop=font_legend, frameon=True, loc='lower right')


ax18.set_xlim([0-10, batch_duration+10])
ax18.xaxis.set_major_locator(MultipleLocator(200))
plt.xlabel("Time ($t$)", family='Arial', size=12, weight='normal', labelpad=7)

ax18.set_ylim([-0.2, 3.2])
plt.ylabel("Injection Velocity $v_{z}$", family='Arial', size=12, weight='normal', labelpad=5)

ax18.tick_params(axis='both', direction="in", right=True, top=True)

ax18.text(830, 0.9, '38th batch', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':12})

axins = ax18.inset_axes((0.2, 0.3, 0.4, 0.4))
axins.plot(xticks, yrt, label='Reference Trajectory', \
          linestyle='--', color='#2f3542', linewidth=1.5, dash_capstyle='round', zorder=1)
axins.plot(xticks, yt_regular, label='CTA-DDPG', \
          linestyle='-', color='#546de5', linewidth=1.0, dash_capstyle='round', zorder=4)
axins.plot(xticks, yt_zero_ilc, label='Zero-ILC', \
          linestyle='-', color='#63cdda', linewidth=1.0, dash_capstyle='round', zorder=2)
axins.plot(xticks, yt_near_ilc, label='Near-ILC', \
          linestyle='-', color='#cf6a87', linewidth=1.0, dash_capstyle='round', zorder=3)

zone_left = 845
zone_right = batch_duration-1

x_ratio = 0.01  
y_ratio = 0.05  

xlim0 = xticks[zone_left]-(xticks[zone_right]-xticks[zone_left])*x_ratio
xlim1 = xticks[zone_right]+(xticks[zone_right]-xticks[zone_left])*x_ratio

y1 = np.hstack((yt_zero_ilc[zone_left:zone_right]))
y2 = np.hstack((yt_regular[zone_left:zone_right]))
ylim0 = np.min(y1)-(np.max(y1)-np.min(y1))*y_ratio
ylim1 = np.max(y2)+(np.max(y2)-np.min(y2))*y_ratio

axins.set_xlim(xlim0, xlim1)
axins.set_ylim(1.88, 2.12)

rect = plt.Rectangle((840, 1.87), batch_duration-840, 0.25, fill=False, color='#f78fb3', linewidth=1.5)
ax18.add_patch(rect)
ax18.annotate('', xy=(635, 1.45), xytext=(790, 1.77), \
             arrowprops=dict(facecolor='black', width=1.5, headwidth=6.5, headlength=8, shrink=0.05, alpha=0.8), \
             fontproperties = 'Arial', size = 11)

plt.savefig('./Data/Figures/Fig17-1 - Tracking trajectory of injection velocity vz in the 38th batch.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.17-2：Tracking trajectory of injection velocity vz in the 25th batch.
PARAM_Mismatch = Sys_Init()
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()

Agent_Plot.__load_Traj_Online_Mismatch__(PARAM_Mismatch)
Agent_Plot.__load_Traj_Online_Regular__(PARAM)
Agent_Plot.__load_Traj_ILC_Zero__(PARAM)
Agent_Plot.__load_Traj_ILC_Near__(PARAM)

selected_batch = 25

yrt         = list(PARAM.yrt_traj_Online_deque[selected_batch-1])
yt_regular  = list(PARAM.yt_traj_Online_deque[selected_batch-1])
yt_zero_ilc = list(PARAM.yt_ilc_zero_traj_deque[selected_batch-1])
yt_near_ilc = list(PARAM.yt_ilc_near_traj_deque[selected_batch-1])
yt_mimatch  = list(PARAM_Mismatch.yt_traj_Online_deque[selected_batch-1])

batch_duration = len(yrt)
xticks = np.arange(1, batch_duration+1)

fig16 = plt.figure(figsize=(7.2, 4))
ax19  = fig16.add_subplot()

ax19.plot(xticks, yrt, label='Reference Trajectory', \
          linestyle='--', color='#2f3542', linewidth=2.5, dash_capstyle='round', zorder=1)
ax19.plot(xticks, yt_regular, label='CTA-DDPG', \
          linestyle='-', color='#546de5', linewidth=1.5, dash_capstyle='round', zorder=4)
ax19.plot(xticks, yt_zero_ilc, label='Zero ILC', \
          linestyle='-', color='#63cdda', linewidth=1.5, dash_capstyle='round', zorder=2)
ax19.plot(xticks, yt_near_ilc, label='Nearest ILC', \
          linestyle='-', color='#cf6a87', linewidth=1.5, dash_capstyle='round', zorder=3)

ax19.legend(prop=font_legend, frameon=True, loc='lower right')

ax19.set_xlim([0-10, batch_duration+10])

ax19.xaxis.set_major_locator(MultipleLocator(200))
plt.xlabel("Time ($t$)", family='Arial', size=12, weight='normal', labelpad=7)

ax19.set_ylim([-0.2, 3.2])
plt.ylabel("Injection Velocity $v_{z}$", family='Arial', size=12, weight='normal', labelpad=5)

ax19.tick_params(axis='both', direction="in", right=True, top=True)

ax19.text(830, 0.9, '25th batch', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':12})

axins = ax19.inset_axes((0.2, 0.3, 0.4, 0.4))
axins.plot(xticks, yrt, label='Reference Trajectory', \
          linestyle='--', color='#2f3542', linewidth=1.5, dash_capstyle='round', zorder=1)
axins.plot(xticks, yt_regular, label='CTA-DDPG', \
          linestyle='-', color='#546de5', linewidth=1.0, dash_capstyle='round', zorder=4)
axins.plot(xticks, yt_zero_ilc, label='Zero-ILC', \
          linestyle='-', color='#63cdda', linewidth=1.0, dash_capstyle='round', zorder=2)
axins.plot(xticks, yt_near_ilc, label='Near-ILC', \
          linestyle='-', color='#cf6a87', linewidth=1.0, dash_capstyle='round', zorder=3)

zone_left = 845
zone_right = batch_duration-1

x_ratio = 0.01 
y_ratio = 0.05 

xlim0 = xticks[zone_left]-(xticks[zone_right]-xticks[zone_left])*x_ratio
xlim1 = xticks[zone_right]+(xticks[zone_right]-xticks[zone_left])*x_ratio

y1 = np.hstack((yt_zero_ilc[zone_left:zone_right]))
y2 = np.hstack((yt_regular[zone_left:zone_right]))
ylim0 = np.min(y1)-(np.max(y1)-np.min(y1))*y_ratio
ylim1 = np.max(y2)+(np.max(y2)-np.min(y2))*y_ratio

axins.set_xlim(xlim0, xlim1)
axins.set_ylim(1.75, 2.1)

rect = plt.Rectangle((840, 1.87), batch_duration-840, 0.25, fill=False, color='#f78fb3', linewidth=1.5)
ax19.add_patch(rect)
ax19.annotate('', xy=(635, 1.45), xytext=(790, 1.77), \
             arrowprops=dict(facecolor='black', width=1.5, headwidth=6.5, headlength=8, shrink=0.05, alpha=0.8), \
             fontproperties = 'Arial', size = 11)

plt.savefig('./Data/Figures/Fig17-2 - Tracking trajectory of injection velocity vz in the 25th batch.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.19-1：Tracking errors in former time points.
PARAM  = Sys_Init()
PARAM_Mismatch  = Sys_Init()
Agent_Plot = DDPG_Plot()
Agent_Plot.__load_Traj_Online_Regular__(PARAM)
Agent_Plot.__load_Traj_Online_Mismatch__(PARAM_Mismatch)

selected_batch = [1, 2, 3, 4, 5, 6]

batch_length = PARAM.errt_traj_Online_deque.__len__()
err_traj_regular  = []
err_traj_mismatch = []

fig22 = plt.figure(figsize=(7.2, 4))

ax24  = fig22.add_subplot(111)

colors = ['#ffa502', '#34e7e4', '#2ed573', '#f368e0', '#ff4757', '#747d8c']
label = ['1st Batch', '2nd Batch', '3rd Batch', '4th Batch', '5th Batch', '6th Batch']

for i,c,l in zip(selected_batch,colors,label):
    err_traj_regular  = list(PARAM.errt_traj_Online_deque[i-1])
    err_traj_mismatch = list(PARAM_Mismatch.errt_traj_Online_deque[i-1])
    ts = np.arange(1, len(err_traj_regular)+1)
    ax24.plot(ts, err_traj_regular, label=l, linestyle='-.', color=c, linewidth=1.5, dash_capstyle='round')

x_range = 25
xticks  = [1, 5, 10, 15, 20, 25]
xlabels = [1, 5, 10, 15, 20, 25]

ax24.set_xlim([1, x_range+0.5])
ax24.set_xticks(xticks, xlabels)

plt.xlabel("Time ($t$)", family='Arial', size=12, weight='normal', labelpad=7)

ax24.set_ylim([-0.01, 0.015])

ax24.set_xlabel("Time $t$", family='Arial', size=12, weight='normal', labelpad=7)
ax24.set_ylabel("Error $e_{t}$", family='Arial', size=12, weight='normal', labelpad=7)

ax24.text(11.18+0.35, -0.00825, 'CTA-DDPG', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':12})

ax24.tick_params(axis='both', direction="in", right=True, top=True)

font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size'  :  10,}

lines, labels = fig22.axes[-1].get_legend_handles_labels()
ax24.legend(lines, labels, loc='lower right', ncol=2, prop=font_legend)

plt.tight_layout()

plt.savefig('./Data/Figures/Fig19-1 - Tracking errors in former time points.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.19-2：Tracking errors in former time points.
PARAM  = Sys_Init()
Agent_Plot = DDPG_Plot()
Agent_Plot.__load_Traj_ILC_Zero__(PARAM)
Agent_Plot.__load_Traj_ILC_Near__(PARAM)

selected_batch = [1, 2, 3, 4, 5, 6]

batch_length = PARAM.errt_traj_Online_deque.__len__()
err_traj_zero     = []
err_traj_near     = []

fig22 = plt.figure(figsize=(7.2, 4))

ax27  = fig22.add_subplot(111)

colors = ['#ffa502', '#34e7e4', '#2ed573', '#f368e0', '#ff4757', '#747d8c']
label = ['1st Batch', '2nd Batch', '3rd Batch', '4th Batch', '5th Batch', '6th Batch']

for i,c,l in zip(selected_batch,colors,label):
    err_traj_zero     = list(PARAM.err_ilc_zero_traj_deque[i-1])
    err_traj_near     = list(PARAM.err_ilc_near_traj_deque[i-1])
    ts = np.arange(1, len(err_traj_zero)+1)
    ax27.plot(ts, err_traj_near, label=l, linestyle='-.', color=c, linewidth=1.5, dash_capstyle='round')

x_range = 25
xticks  = [1, 5, 10, 15, 20, 25]
xlabels = [1, 5, 10, 15, 20, 25]

ax27.set_xlim([1, x_range+0.5])
ax27.set_xticks(xticks, xlabels)

plt.xlabel("Time ($t$)", family='Arial', size=12, weight='normal', labelpad=7)

ax27.set_ylim([-0.1, 0.7])

ax27.set_xlabel("Time $t$", family='Arial', size=12, weight='normal', labelpad=7)
ax27.set_ylabel("Error $e_{t}$", family='Arial', size=12, weight='normal', labelpad=7)

ax27.text(11, -0.042, 'Zero / Near ILC', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':11})

ax27.tick_params(axis='both', direction="in", right=True, top=True)

font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size'  :  10,}

lines, labels = fig22.axes[-1].get_legend_handles_labels()
ax27.legend(lines, labels, loc='lower right', ncol=2, prop=font_legend)

plt.tight_layout()

plt.savefig('./Data/Figures/Fig19-2 - Tracking errors in former time points.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************






plt.show()





# ------------------------------------------------------------------------------------------ 
# ============================================== End of File  =============================================== #


