import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sympy import symbols, latex

from collections import deque

from Utilities import DDPG_Plot, Sys_Init
import math

np.random.seed(19980823)


# font_dict
font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size'  :  10,}

# ******************************************************************************************************************
# todo: Fig.5：The training curve in the offline phase in Case 1.
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()
# read file
Agent_Plot.__load_Traj_Offline_Figure__(PARAM)


batch_length = PARAM.reward_Offline_deque.__len__()

total_reward        = []
average_step_reward = []
for i in range(batch_length):
    reward_traj = PARAM.reward_Offline_deque[i]
    total_reward.append(sum(reward_traj))
    average_step_reward.append(sum(reward_traj) / len(reward_traj))

xticks = np.arange(1, batch_length+1)

fig1 = plt.figure(figsize=(7.2, 4))
ax1  = fig1.add_subplot()

ax1.plot(xticks, total_reward, label='Total Reward per Episode', \
         linestyle='-', color='#FF7F0E', linewidth=1.1, dash_capstyle='round', zorder=1)

ax1.set_xlim([-50, batch_length+50])
plt.xlabel("Training Episode", family='Arial', size=12, weight='normal', labelpad=7)

ax1.set_ylim([-50, 1000])
plt.ylabel("Total Reward  " + r"$r_k^{total}$", family='Arial', size=12, weight='normal', labelpad=5)

ax1.tick_params(axis='both', direction="in", right=True, top=True)

# todo: enlarged figure
axins = ax1.inset_axes((0.27, 0.17, 0.7, 0.4))
axins.plot(xticks, total_reward, linestyle='-', color='#FF7F0E', linewidth=0.8, dash_capstyle='round')

zone_left = 2850
zone_right = 2999

x_ratio = 0.02
y_ratio = 0.02

xlim0 = xticks[zone_left]-(xticks[zone_right]-xticks[zone_left])*x_ratio
xlim1 = xticks[zone_right]+(xticks[zone_right]-xticks[zone_left])*x_ratio

y = np.hstack((total_reward[zone_left:zone_right]))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

axins.set_xlim(xlim0, xlim1)
axins.set_ylim(875, 950)

axins.set_xticks([2850,2900,2950,3000])

ellipse = Ellipse(xy=(2925,910), width=200*1.2, height=80*1.2, fill=False, color='#f78fb3', linewidth=1.2)
ax1.add_patch(ellipse)
ax1.annotate('', xy=(2600, 600), xytext=(2860, 820), \
             arrowprops=dict(facecolor='black', width=1.5, headwidth=6.5, headlength=8, shrink=0.05, alpha=0.8), \
             fontproperties = 'Arial', size = 11)

plt.savefig('./Data/Figures/Fig5 - The training curve in the offline phase in Case 1.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.6：Tracking performance of the agent during the offline stage.
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()

Agent_Plot.__load_Traj_Offline__(PARAM)

font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size'  :  9.5,}
yt_traj  = []
yrt_traj = []

colors = ['#57dbcb', '#db57a6', '#ffa502', '#57db7c', \
          '#9157db', '#cf6a87', '#57acdb', '#747d8c', '#0652DD']

fig24 = plt.figure(figsize=(14.4, 8))
ax29  = fig24.add_subplot(221)
ax30  = fig24.add_subplot(222)
ax31  = fig24.add_subplot(223)
ax32  = fig24.add_subplot(224)

yrt_traj = PARAM.yrt_traj_Offline_deque[0]
xticks   = np.arange(1, len(yrt_traj)+1)

selected_batch = PARAM.episode_traj_offline_list    # list
print(selected_batch)
print(len(selected_batch))
selected_batch_dict = {element: index for index, element in enumerate(selected_batch)}
# subfigure 1
ax29.plot(xticks, yrt_traj, label='Reference Trajectory',
          linestyle='--', color='#f6e58d', linewidth=2.5, dash_capstyle='round', zorder=3100, alpha=0.95)
batch_legend = ['1st episode','2nd episode', '3rd episode', '4th episode', '5th episode', '6th episode', '7th episode', '8th episode', '9th episode']
for i,c,l in zip([0,1,2,3,4,5,6,7,8], colors, batch_legend):
    yt_traj = PARAM.yt_traj_Offline_deque[selected_batch_dict[i]]
    ax29.plot(xticks, yt_traj, label=l, \
          linestyle='-.', color=c, linewidth=0.8, dash_capstyle='round', zorder=i)

ax29.legend(prop=font_legend, frameon=True, loc='upper right', ncol=2)

ax29.xaxis.set_major_locator(MultipleLocator(200))
ax29.set_xlim([0-10, 1000+10])

ax29.set_ylim([-0.8, 0.9])
ax29.set_xlabel("Time ($t$)", family='Arial', size=12, weight='normal', labelpad=5)
ax29.set_ylabel("$y_{t}$", family='Arial', size=12, weight='normal', labelpad=5)

ax29.tick_params(axis='both', direction="in", right=True, top=True)

ax29.text(27, 0.7, '1', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':16})


# subfigure 2
ax30.plot(xticks, yrt_traj, label='Reference Trajectory', \
          linestyle='--', color='#f6e58d', linewidth=2.5, dash_capstyle='round', zorder=3100, alpha=0.95)
for i,c in zip([100,200,300,400,500,600,700,800,900], colors):
    yt_traj = PARAM.yt_traj_Offline_deque[selected_batch_dict[i]]
    ax30.plot(xticks, yt_traj, label="{}th episode".format(i), \
          linestyle='-.', color=c, linewidth=0.8, dash_capstyle='round', zorder=i)

ax30.legend(prop=font_legend, frameon=True, loc='upper right', ncol=2)

ax30.xaxis.set_major_locator(MultipleLocator(200))
ax30.set_xlim([0-10, 1000+10])

ax30.set_ylim([-0.8, 0.9])
ax30.set_xlabel("Time ($t$)", family='Arial', size=12, weight='normal', labelpad=5)
ax30.set_ylabel("$y_{t}$", family='Arial', size=12, weight='normal', labelpad=5)

ax30.tick_params(axis='both', direction="in", right=True, top=True)

ax30.text(27, 0.7, '2', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':16})


# subfigure 3
ax31.plot(xticks, yrt_traj, label='Reference Trajectory', \
          linestyle='--', color='#f6e58d', linewidth=2.5, dash_capstyle='round', zorder=3100, alpha=0.95)
for i,c in zip([1200,1300,1400,1500,1600,1700,1800,1900,2000], colors):
    yt_traj = PARAM.yt_traj_Offline_deque[selected_batch_dict[i]]
    ax31.plot(xticks, yt_traj, label="{}th episode".format(i), \
          linestyle='-.', color=c, linewidth=0.8, dash_capstyle='round', zorder=i)

ax31.legend(prop=font_legend, frameon=True, loc='upper right', ncol=2)

ax31.xaxis.set_major_locator(MultipleLocator(200))
ax31.set_xlim([0-10, 1000+10])

ax31.set_ylim([-0.8, 0.9])
ax31.set_xlabel("Time ($t$)", family='Arial', size=12, weight='normal', labelpad=5)
ax31.set_ylabel("$y_{t}$", family='Arial', size=12, weight='normal', labelpad=5)

ax31.tick_params(axis='both', direction="in", right=True, top=True)

ax31.text(27, 0.7, '3', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':16})


# subfigure 4
ax32.plot(xticks, yrt_traj, label='Reference Trajectory', \
          linestyle='--', color='#f6e58d', linewidth=2.5, dash_capstyle='round', zorder=3100, alpha=0.95)
for i,c in zip([2000,2100,2200,2300,2400,2500,2600,2700,2800], colors):
    yt_traj = PARAM.yt_traj_Offline_deque[selected_batch_dict[i]]
    ax32.plot(xticks, yt_traj, label="{}th episode".format(i+200), \
          linestyle='-.', color=c, linewidth=0.8, dash_capstyle='round', zorder=i)

ax32.legend(prop=font_legend, frameon=True, loc='upper right', ncol=2)
ax32.set_xlim([0-10, 1000+10])

ax32.xaxis.set_major_locator(MultipleLocator(200))

ax32.set_ylim([-0.8, 0.9])
ax32.set_xlabel("Time ($t$)", family='Arial', size=12, weight='normal', labelpad=5)
ax32.set_ylabel("$y_{t}$", family='Arial', size=12, weight='normal', labelpad=5)

ax32.tick_params(axis='both', direction="in", right=True, top=True)

ax32.text(27, 0.7, '4', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':16})

plt.savefig('./Data/Figures/Fig6 - Tracking performance of the agent during the offline stage.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************


# ******************************************************************************************************************
# todo: Fig.7-1：Tracking trajectories in Scenario 1 (CTA-DDPG).
PARAM = Sys_Init()
PARAM.Phase_Index = 0
PARAM.Scen_Index  = 1
Agent_Plot = DDPG_Plot()

Agent_Plot.__load_Traj_Online_No_STER__(PARAM)

fig8 = plt.figure(figsize=(7.2, 6.5))
ax9 = fig8.add_subplot(111, projection='3d')

ax9.grid(False)

ax9.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax9.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax9.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax9.get_proj = lambda: np.dot(Axes3D.get_proj(ax9), np.diag([1.2, 0.8, 0.8, 1]))

colors = ['#ffa502', '#34e7e4', '#2ed573', '#f368e0', \
          '#ff4757', '#747d8c', '#0652DD', '#e15f41']

yticks = [1, 4, 8, 12, 16, 20]

for c,k in zip(colors, yticks):

    yt  = list(PARAM.yt_traj_Online_deque[k-1])
    yrt = list(PARAM.yrt_traj_Online_deque[k-1])

    if len(yt) == len(yrt):     
        t = len(yt)
    else:
        print("yt did not match yrt!")
    ts = np.arange(0,t)
    # Plot
    ax9.plot(ts, yrt, zs=k, zdir='y', linestyle='--',color='dimgray', dash_capstyle='round', alpha=0.8, linewidth=2.7, label='Reference Output')
    ax9.plot(ts, yt, zs=k, zdir='y',  linestyle='-', color=c, dash_capstyle='round', alpha=0.7, linewidth=1.3)

ax9.set_xlabel('Time ($t$)', family='Arial', size=12, weight='normal', labelpad=10)
ax9.set_ylabel('Batch ($k$)', family='Arial', size=12, weight='normal', labelpad=5)
ax9.set_zlabel('$y_{t}$', family='Arial', size=12, weight='normal', labelpad=5)

ax9.zaxis._axinfo['juggled'] = (1,2,1)
ax9.zaxis.set_rotate_label(0)

ax9.set_zlim(-0.9, 0.9)

ax9.set_yticks(yticks)

ax9.xaxis.set_major_locator(MultipleLocator(200))
ax9.zaxis.set_major_locator(MultipleLocator(0.4))

ax9.view_init(14, -56)

fig8.tight_layout()
fig8.subplots_adjust(left=0.1)  # plot outside the normal area
plt.savefig("./Data/Figures/Fig7-1 - Tracking trajectories in Scenario 1 (CTA-DDPG).png", dpi=300)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.10-1：Tracking trajectories in Scenario 2 (CTA-DDPG).
PARAM = Sys_Init()
PARAM.Phase_Index = 0
PARAM.Scen_Index  = 2
Agent_Plot = DDPG_Plot()

Agent_Plot.__load_Traj_Online_No_STER__(PARAM)

fig8 = plt.figure(figsize=(7.2, 6.5))
ax9 = fig8.add_subplot(111, projection='3d')

ax9.grid(False)

ax9.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax9.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax9.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax9.get_proj = lambda: np.dot(Axes3D.get_proj(ax9), np.diag([1.2, 0.8, 0.8, 1]))

colors = ['#ffa502', '#34e7e4', '#2ed573', '#f368e0', \
          '#ff4757', '#747d8c', '#0652DD', '#e15f41']

yticks = [1, 4, 8, 12, 16, 20]


for c,k in zip(colors, yticks):
    yt  = list(PARAM.yt_traj_Online_deque[k-1])
    yrt = list(PARAM.yrt_traj_Online_deque[k-1])
    if len(yt) == len(yrt):     
        t = len(yt)
    else:
        print("yt did not match yrt!")
    ts = np.arange(0,t)
    # Plot
    ax9.plot(ts, yrt, zs=k, zdir='y', linestyle='--',color='dimgray', dash_capstyle='round', alpha=0.8, linewidth=2.7, label='Reference Output')
    ax9.plot(ts, yt, zs=k, zdir='y',  linestyle='-', color=c, dash_capstyle='round', alpha=0.7, linewidth=1.3)

ax9.set_xlabel('Time ($t$)', family='Arial', size=12, weight='normal', labelpad=10)
ax9.set_ylabel('Batch ($k$)', family='Arial', size=12, weight='normal', labelpad=5)
ax9.set_zlabel('$y_{t}$', family='Arial', size=12, weight='normal', labelpad=5)

ax9.zaxis._axinfo['juggled'] = (1,2,1)
ax9.zaxis.set_rotate_label(0)

ax9.set_zlim(-0.9, 0.9)

ax9.set_yticks(yticks)

ax9.xaxis.set_major_locator(MultipleLocator(200))
ax9.zaxis.set_major_locator(MultipleLocator(0.4))

ax9.view_init(14, -56)

fig8.tight_layout()
fig8.subplots_adjust(left=0.1)  # plot outside the normal area
plt.savefig("./Data/Figures/Fig10-1 - Tracking trajectories in Scenario 2 (CTA-DDPG).png", dpi=300)
# ******************************************************************************************************************


# ******************************************************************************************************************
# todo: Fig.11-1：Tracking trajectories in Scenario 3 (CTA-DDPG).
PARAM = Sys_Init()
PARAM.Phase_Index = 0
PARAM.Scen_Index  = 3
Agent_Plot = DDPG_Plot()

Agent_Plot.__load_Traj_Online_No_STER__(PARAM)

fig8 = plt.figure(figsize=(7.2, 6.5))
ax9 = fig8.add_subplot(111, projection='3d')

ax9.grid(False)

ax9.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax9.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax9.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax9.get_proj = lambda: np.dot(Axes3D.get_proj(ax9), np.diag([1.2, 0.8, 0.8, 1]))


colors = ['#ffa502', '#34e7e4', '#2ed573', '#f368e0', \
          '#ff4757', '#747d8c', '#0652DD', '#e15f41']

yticks = [1, 6, 11, 16, 21, 26, 31]


for c,k in zip(colors, yticks):

    yt  = list(PARAM.yt_traj_Online_deque[k-1])
    yrt = list(PARAM.yrt_traj_Online_deque[k-1])

    if len(yt) == len(yrt):     
        t = len(yt)
    else:
        print("yt did not match yrt!")
    ts = np.arange(0,t)
    # Plot
    ax9.plot(ts, yrt, zs=k, zdir='y', linestyle='--',color='dimgray', dash_capstyle='round', alpha=0.8, linewidth=2.7, label='Reference Output')
    ax9.plot(ts, yt, zs=k, zdir='y',  linestyle='-', color=c, dash_capstyle='round', alpha=0.7, linewidth=1.3)

ax9.set_xlabel('Time ($t$)', family='Arial', size=12, weight='normal', labelpad=10)
ax9.set_ylabel('Batch ($k$)', family='Arial', size=12, weight='normal', labelpad=5)
ax9.set_zlabel('$y_{t}$', family='Arial', size=12, weight='normal', labelpad=5)

ax9.zaxis._axinfo['juggled'] = (1,2,1)
ax9.zaxis.set_rotate_label(0)

ax9.set_zlim(-0.9, 0.9)

ax9.set_yticks(yticks)

ax9.xaxis.set_major_locator(MultipleLocator(200))
ax9.zaxis.set_major_locator(MultipleLocator(0.4))

ax9.view_init(14, -56)

fig8.tight_layout()
fig8.subplots_adjust(left=0.1)  # plot outside the normal area
plt.savefig("./Data/Figures/Fig11-1 - Tracking trajectories in Scenario 3 (CTA-DDPG).png", dpi=300)
# ******************************************************************************************************************




# ******************************************************************************************************************
# todo: Fig.7-2：Tracking trajectories in Scenario 1 (Zero ILC).
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()
PARAM.Phase_Index = 0
PARAM.Scen_Index  = 1
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

yticks = [1, 4, 8, 12, 16, 20]

for c,k in zip(colors, yticks):
    yt  = list(PARAM.yt_ilc_zero_traj_deque[k-1])
    yrt = list(PARAM.yrt_ilc_zero_traj_deque[k-1])
    if len(yt) == len(yrt):     
        t = len(yt)
    else:
        print("yt did not match yrt!")
    ts = np.arange(0,t)
    # Plot
    ax10.plot(ts, yrt, zs=k, zdir='y', linestyle='--',color='dimgray', dash_capstyle='round', alpha=0.8, linewidth=2.7)       
    ax10.plot(ts, yt, zs=k, zdir='y',  linestyle='-', color=c, dash_capstyle='round', alpha=0.7, linewidth=1.3)

ax10.set_xlabel('Time ($t$)', family='Arial', size=12, weight='normal', labelpad=10)
ax10.set_ylabel('Batch ($k$)', family='Arial', size=12, weight='normal', labelpad=5)
ax10.set_zlabel('$y_{t}$', family='Arial', size=12, weight='normal', labelpad=5)

ax10.zaxis._axinfo['juggled'] = (1,2,1)
ax10.zaxis.set_rotate_label(0)

ax10.set_zlim(-0.9, 0.9)

ax10.set_yticks(yticks)

ax10.xaxis.set_major_locator(MultipleLocator(200))
ax10.zaxis.set_major_locator(MultipleLocator(0.4))

ax10.view_init(14, -56)

fig9.tight_layout()
fig9.subplots_adjust(left=0.1)  # plot outside the normal area
plt.savefig("./Data/Figures/Fig7-2 - Tracking trajectories in Scenario 1 (Zero ILC).png", dpi=300)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.10-2：Tracking trajectories in Scenario 2 (Zero ILC).
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()
PARAM.Phase_Index = 0
PARAM.Scen_Index  = 2
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

yticks = [1, 4, 8, 12, 16, 20]

for c,k in zip(colors, yticks):
    yt  = list(PARAM.yt_ilc_zero_traj_deque[k-1])
    yrt = list(PARAM.yrt_ilc_zero_traj_deque[k-1])
    if len(yt) == len(yrt):     
        t = len(yt)
    else:
        print("yt did not match yrt!")
    ts = np.arange(0,t)
    # Plot
    ax10.plot(ts, yrt, zs=k, zdir='y', linestyle='--',color='dimgray', dash_capstyle='round', alpha=0.8, linewidth=2.7)     
    ax10.plot(ts, yt, zs=k, zdir='y',  linestyle='-', color=c, dash_capstyle='round', alpha=0.7, linewidth=1.3)

ax10.set_xlabel('Time ($t$)', family='Arial', size=12, weight='normal', labelpad=10)
ax10.set_ylabel('Batch ($k$)', family='Arial', size=12, weight='normal', labelpad=5)
ax10.set_zlabel('$y_{t}$', family='Arial', size=12, weight='normal', labelpad=5)

ax10.zaxis._axinfo['juggled'] = (1,2,1)
ax10.zaxis.set_rotate_label(0)

ax10.set_zlim(-0.9, 0.9)

ax10.set_yticks(yticks)

ax10.xaxis.set_major_locator(MultipleLocator(200))
ax10.zaxis.set_major_locator(MultipleLocator(0.4))

ax10.view_init(14, -56)

fig9.tight_layout()
fig9.subplots_adjust(left=0.1)  # plot outside the normal area
plt.savefig("./Data/Figures/Fig10-2 - Tracking trajectories in Scenario 2 (Zero ILC).png", dpi=300)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.11-2：Tracking trajectories in Scenario 3 (Zero ILC).
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()
PARAM.Phase_Index = 0
PARAM.Scen_Index  = 3
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

yticks = [1, 6, 11, 16, 21, 26, 31]

for c,k in zip(colors, yticks):
    yt  = list(PARAM.yt_ilc_zero_traj_deque[k-1])
    yrt = list(PARAM.yrt_ilc_zero_traj_deque[k-1])
    if len(yt) == len(yrt):     
        t = len(yt)
    else:
        print("yt did not match yrt!")
    ts = np.arange(0,t)
    # Plot
    ax10.plot(ts, yrt, zs=k, zdir='y', linestyle='--',color='dimgray', dash_capstyle='round', alpha=0.8, linewidth=2.7)        
    ax10.plot(ts, yt, zs=k, zdir='y',  linestyle='-', color=c, dash_capstyle='round', alpha=0.7, linewidth=1.3)

ax10.set_xlabel('Time ($t$)', family='Arial', size=12, weight='normal', labelpad=10)
ax10.set_ylabel('Batch ($k$)', family='Arial', size=12, weight='normal', labelpad=5)
ax10.set_zlabel('$y_{t}$', family='Arial', size=12, weight='normal', labelpad=5)

ax10.zaxis._axinfo['juggled'] = (1,2,1)
ax10.zaxis.set_rotate_label(0)

ax10.set_zlim(-0.9, 0.9)

ax10.set_yticks(yticks)

ax10.xaxis.set_major_locator(MultipleLocator(200))
ax10.zaxis.set_major_locator(MultipleLocator(0.4))

ax10.view_init(14, -56)

fig9.tight_layout()
fig9.subplots_adjust(left=0.1)  # plot outside the normal area
plt.savefig("./Data/Figures/Fig11-2 - Tracking trajectories in Scenario 3 (Zero ILC).png", dpi=300)
# ******************************************************************************************************************




# ******************************************************************************************************************
# todo: Fig.7-3：Tracking trajectories in Scenario 1 (Near ILC).
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()
PARAM.Phase_Index = 0
PARAM.Scen_Index  = 1
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

yticks = [1, 4, 8, 12, 16, 20]

for c,k in zip(colors, yticks):
    yt  = list(PARAM.yt_ilc_near_traj_deque[k-1])
    yrt = list(PARAM.yrt_ilc_near_traj_deque[k-1])
    if len(yt) == len(yrt):     
        t = len(yt)
    else:
        print("yt did not match yrt!")
    ts = np.arange(0,t)
    # Plot
    ax11.plot(ts, yrt, zs=k, zdir='y', linestyle='--',color='dimgray', dash_capstyle='round', alpha=0.8, linewidth=2.7)
    ax11.plot(ts, yt, zs=k, zdir='y',  linestyle='-', color=c, dash_capstyle='round', alpha=0.7, linewidth=1.3)

ax11.set_xlabel('Time ($t$)', family='Arial', size=12, weight='normal', labelpad=10)
ax11.set_ylabel('Batch ($k$)', family='Arial', size=12, weight='normal', labelpad=5)
ax11.set_zlabel('$y_{t}$', family='Arial', size=12, weight='normal', labelpad=5)

ax11.zaxis._axinfo['juggled'] = (1,2,1)
ax11.zaxis.set_rotate_label(0)

ax11.set_zlim(-0.9, 0.9)

ax11.set_yticks(yticks)

ax11.xaxis.set_major_locator(MultipleLocator(200))
ax11.zaxis.set_major_locator(MultipleLocator(0.4))

ax11.view_init(14, -56)

fig10.tight_layout()
fig10.subplots_adjust(left=0.1)  # plot outside the normal area
plt.savefig("./Data/Figures/Fig7-3 - Tracking trajectories in Scenario 1 (Near ILC).png", dpi=300)
# ******************************************************************************************************************




# ******************************************************************************************************************
# todo: Fig.10-3：Tracking trajectories in Scenario 2 (Near ILC)
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()
PARAM.Phase_Index = 0
PARAM.Scen_Index  = 2
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

yticks = [1, 4, 8, 12, 16, 20]

for c,k in zip(colors, yticks):
    yt  = list(PARAM.yt_ilc_near_traj_deque[k-1])
    yrt = list(PARAM.yrt_ilc_near_traj_deque[k-1])
    if len(yt) == len(yrt):     
        t = len(yt)
    else:
        print("yt did not match yrt!")
    ts = np.arange(0,t)
    # Plot
    ax11.plot(ts, yrt, zs=k, zdir='y', linestyle='--',color='dimgray', dash_capstyle='round', alpha=0.8, linewidth=2.7)
    ax11.plot(ts, yt, zs=k, zdir='y',  linestyle='-', color=c, dash_capstyle='round', alpha=0.7, linewidth=1.3)

ax11.set_xlabel('Time ($t$)', family='Arial', size=12, weight='normal', labelpad=10)
ax11.set_ylabel('Batch ($k$)', family='Arial', size=12, weight='normal', labelpad=5)
ax11.set_zlabel('$y_{t}$', family='Arial', size=12, weight='normal', labelpad=5)

ax11.zaxis._axinfo['juggled'] = (1,2,1)
ax11.zaxis.set_rotate_label(0)

ax11.set_zlim(-0.9, 0.9)

ax11.set_yticks(yticks)

ax11.xaxis.set_major_locator(MultipleLocator(200))
ax11.zaxis.set_major_locator(MultipleLocator(0.4))

ax11.view_init(14, -56)

fig10.tight_layout()
fig10.subplots_adjust(left=0.1)  # plot outside the normal area
plt.savefig("./Data/Figures/Fig10-3 - Tracking trajectories in Scenario 2 (Near ILC).png", dpi=300)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.11-3：Tracking trajectories in Scenario 3 (Near ILC)
PARAM = Sys_Init()
Agent_Plot = DDPG_Plot()
PARAM.Phase_Index = 0
PARAM.Scen_Index  = 3
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


yticks = [1, 6, 11, 16, 21, 26, 31]

for c,k in zip(colors, yticks):
    yt  = list(PARAM.yt_ilc_near_traj_deque[k-1])
    yrt = list(PARAM.yrt_ilc_near_traj_deque[k-1])

    if len(yt) == len(yrt):     
        t = len(yt)
    else:
        print("yt did not match yrt!")
    ts = np.arange(0,t)
    # Plot
    ax11.plot(ts, yrt, zs=k, zdir='y', linestyle='--',color='dimgray', dash_capstyle='round', alpha=0.8, linewidth=2.7)
    ax11.plot(ts, yt, zs=k, zdir='y',  linestyle='-', color=c, dash_capstyle='round', alpha=0.7, linewidth=1.3)

ax11.set_xlabel('Time ($t$)', family='Arial', size=12, weight='normal', labelpad=10)
ax11.set_ylabel('Batch ($k$)', family='Arial', size=12, weight='normal', labelpad=5)
ax11.set_zlabel('$y_{t}$', family='Arial', size=12, weight='normal', labelpad=5)

ax11.zaxis._axinfo['juggled'] = (1,2,1)
ax11.zaxis.set_rotate_label(0)

ax11.set_zlim(-0.9, 0.9)

ax11.set_yticks(yticks)

ax11.xaxis.set_major_locator(MultipleLocator(200))
ax11.zaxis.set_major_locator(MultipleLocator(0.4))

ax11.view_init(14, -56)

fig10.tight_layout()
fig10.subplots_adjust(left=0.1)  # plot outside the normal area
plt.savefig("./Data/Figures/Fig11-3 - Tracking trajectories in Scenario 3 (Near ILC).png", dpi=300)
# ******************************************************************************************************************



# ******************************************************************************************************************
# todo: Fig.12：Visualization of policy integration factor and average step reward during the online stage in 3 scenarios.
PARAM = Sys_Init()
PARAM.Phase_Index = 0

Agent_Plot = DDPG_Plot()
Agent_Plot.__load_Traj_Online_No_STER__(PARAM)

p11_mean_value = []
p11_max_value  = []
p11_min_value  = []
p1_mean_value  = []
p1_max_value   = []
p1_min_value   = []
p21_mean_value = []
p21_max_value  = []
p21_min_value  = []
p2_mean_value  = []
p2_max_value   = []
p2_min_value   = []

avr_step_reward_1 = []
avr_step_reward = []

for index in range(1, 4):   # Scenario 1 ~ 3
    PARAM.Scen_Index = index
    Agent_Plot.__load_Traj_Online_No_STER__(PARAM)
    phase_length = PARAM.P1_Online_deque.__len__()
    for i in range(phase_length):
        p11_mean_value.append(sum(PARAM.P1_Online_deque[i]) / len(PARAM.P1_Online_deque[i]))
        p11_max_value.append(max(PARAM.P1_Online_deque[i]))
        p11_min_value.append(min(PARAM.P1_Online_deque[i]))
        p21_mean_value.append(sum(PARAM.P2_Online_deque[i]) / len(PARAM.P2_Online_deque[i]))
        p21_max_value.append(max(PARAM.P2_Online_deque[i]))
        p21_min_value.append(min(PARAM.P2_Online_deque[i]))
        avr_step_reward_1 = PARAM.Expected_step_reward_Online_list
    p1_mean_value.extend(p11_mean_value)
    p1_max_value.extend(p11_max_value)
    p1_min_value.extend(p11_min_value)
    p2_mean_value.extend(p21_mean_value)
    p2_max_value.extend(p21_max_value)
    p2_min_value.extend(p21_min_value)
    avr_step_reward.extend(avr_step_reward_1)
    p11_mean_value = []
    p11_max_value  = []
    p11_min_value  = []
    p21_mean_value = []
    p21_max_value  = []
    p21_min_value  = []
    avr_step_reward_1=[]

batch_length = len(p1_mean_value)


fig27 = plt.figure(figsize=(7.5, 4))
ax40  = fig27.add_subplot(111)
ax41  = ax40.twinx()
xticks = np.arange(1, batch_length+1)

ax40.plot(xticks, p1_mean_value, label=r'${\bar c_1}$', \
          linestyle='-', color='#cf6a87', linewidth=2, dash_capstyle='round', zorder=2)
ax40.plot(xticks, p2_mean_value, label=r'${\bar c_2}$', \
          linestyle='-', color='#0652dd', linewidth=2, dash_capstyle='round', zorder=2)
ax40.fill_between(x=xticks, y1=p1_max_value, y2=p1_min_value, facecolor='#cf6a87', alpha=0.35, label='Range of 'r'$c_1$')
ax40.fill_between(x=xticks, y1=p2_max_value, y2=p2_min_value, facecolor='#546de5', alpha=0.35, label='Range of 'r'$c_2$')

ax40.axvline(x=21, color='dimgrey', linewidth=2.5, dash_capstyle='round', linestyle='--', alpha=0.65)
ax40.axvline(x=41, color='dimgrey', linewidth=2.5, dash_capstyle='round', linestyle='--', alpha=0.65)

ax41.plot(xticks, avr_step_reward, label=r'${\bar r_k}$', \
          linestyle='-', color='#ffa502', linewidth=1.7, dash_capstyle='round', zorder=1, \
          marker='X', markersize=4.3)
ax41.axhline(y=0.9, color='red', linewidth=2, dash_capstyle='round', linestyle='-.', alpha=0.55)

lines  = []
labels = []
lines_01, labels_01 = fig27.axes[0].get_legend_handles_labels()
lines.extend(lines_01); labels.extend(labels_01)
lines_02, labels_02 = fig27.axes[1].get_legend_handles_labels()
lines.extend(lines_02); labels.extend(labels_02)

font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
               'size'  :  9,}
fig27.legend(lines, labels,
             loc='lower center', prop=font_legend, frameon=True, \
             bbox_to_anchor=(0.77+0.062-0.012, 0.515-0.38), ncol=1)

ax40.set_xlabel('Batch ($k$)', family='Arial', size=12, weight='normal', labelpad=10)
ax40.set_ylabel('Integration Factor  ' + r'$c_1,c_2$', family='Arial', size=12, weight='normal', labelpad=5)
ax41.set_ylabel('Average Step Reward ' + r'${\bar r_k}$', family='Arial', size=12, weight='normal', labelpad=5)

xticks1 = [0, 5, 10, 15, 20]
xticks  = []
xticks.extend(xticks1)
xticks1 = [5, 10, 15, 20]
xticks1  =list(map(lambda x: x+20, xticks1))
xticks.extend(xticks1)
xticks1  =list(map(lambda x: x+20, xticks1))
xticks.extend(xticks1)
xticks.extend([65, 70, 75, 80])
xlabels = [0, 5, 10, 15, 20]
xlabels.extend([5, 10, 15, 20])
xlabels.extend([5, 10, 15, 20, 25, 30, 35, 40])
ax40.set_xlim([0+0.5, batch_length+0.5])

ax40.xaxis.set_major_locator(MultipleLocator(5))
ax40.set_xticks(xticks, xlabels)

ax40.yaxis.set_major_locator(MultipleLocator(0.1))
ax41.set_ylim([0.6, 0.93])
ax41.yaxis.set_major_locator(MultipleLocator(0.05))


ax40.tick_params(axis='both', direction="in", right=False, top=True)
ax41.tick_params(axis='both', direction="in", right=True, top=True)

ax40.text(8-4-0.5, 0.07, 'Scenario 1', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':14})
ax40.text(28-4, 0.07, 'Scenario 2', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':14})
ax40.text(50.5-3-0.5, 0.07, 'Scenario 3', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':14})

# todo: enlarged figure

axins = ax40.inset_axes((0.79, 0.58, 0.2, 0.22))

xticks = np.arange(1, batch_length+1)
axins.plot(xticks, p1_mean_value, label=r'${\bar c_1}$', \
          linestyle='-', color='#cf6a87', linewidth=2, dash_capstyle='round', zorder=1)
axins.plot(xticks, p2_mean_value, label=r'${\bar c_2}$', \
          linestyle='-', color='#546de5', linewidth=2, dash_capstyle='round', zorder=1)
axins.fill_between(x=xticks, y1=p1_max_value, y2=p1_min_value, facecolor='#cf6a87', alpha=0.35, label=r'$c_1$')
axins.fill_between(x=xticks, y1=p2_max_value, y2=p2_min_value, facecolor='#546de5', alpha=0.35, label=r'$c_2$')

zone_left  = 62
zone_right = 79

x_ratio = 0.1
y_ratio = 0.05

xlim0 = xticks[zone_left]-(xticks[zone_right]-xticks[zone_left])*x_ratio
xlim1 = xticks[zone_right]+(xticks[zone_right]-xticks[zone_left])*x_ratio

xticks2 = [55, 60, 65, 70, 75, 80]
xlabels = [15, 20, 25, 30, 35, 40]
axins.set_xticks(xticks2, xlabels)

y1 = np.hstack((p1_mean_value[zone_left:zone_right]))
y2 = np.hstack((p2_mean_value[zone_left:zone_right]))
ylim0 = np.min(y1)-(np.max(y1)-np.min(y1))*y_ratio
ylim1 = np.max(y2)+(np.max(y2)-np.min(y2))*y_ratio

axins.set_xlim(xlim0, xlim1-1)
axins.set_ylim(0.49, 0.51)

axins.tick_params(labelsize=9, axis='both', direction="in", right=True, top=True)

plt.savefig('./Data/Figures/Fig12 - Visualization of policy integration factor and average step reward during the online stage in 3 scenarios.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************





# ******************************************************************************************************************
# todo: Fig.13：Policy integration factor at six randomly disturbed time points (27, 306, 553, 672, 784, 859) under Scenario 2.
PARAM = Sys_Init()
PARAM.Phase_Index = 0

PARAM.Scen_Index = 2

Agent_Plot = DDPG_Plot()
Agent_Plot.__load_Traj_Online_No_STER__(PARAM)

batch_length = PARAM.P1_Online_deque.__len__()

p1_list_1 = []
p1_list_2 = []
p1_list_3 = []
p1_list_4 = []
p1_list_5 = []
p1_list_6 = []
p1_list_7 = []
p1_list_8 = []

p2_list_1 = []
p2_list_2 = []
p2_list_3 = []
p2_list_4 = []
p2_list_5 = []
p2_list_6 = []
p2_list_7 = []
p2_list_8 = []

print(PARAM.Online_Noise_Index)
print(sorted(PARAM.Online_Noise_Index))
print(type(PARAM.Online_Noise_Index))

time_list = [27, 306, 553, 672, 784, 859]

colors = ['#57dbcb', '#db57a6', '#ffa502', '#57db7c', \
          '#9157db', '#cf6a87', '#57acdb', '#747d8c', '#0652DD']

for i in range(batch_length):
    p1_list_1.append(PARAM.P1_Online_deque[i][time_list[0]])
    p1_list_2.append(PARAM.P1_Online_deque[i][time_list[1]])
    p1_list_3.append(PARAM.P1_Online_deque[i][time_list[2]])
    p1_list_4.append(PARAM.P1_Online_deque[i][time_list[3]])
    p1_list_5.append(PARAM.P1_Online_deque[i][time_list[4]])
    p1_list_6.append(PARAM.P1_Online_deque[i][time_list[5]])

    p2_list_1.append(PARAM.P2_Online_deque[i][time_list[0]])
    p2_list_2.append(PARAM.P2_Online_deque[i][time_list[1]])
    p2_list_3.append(PARAM.P2_Online_deque[i][time_list[2]])
    p2_list_4.append(PARAM.P2_Online_deque[i][time_list[3]])
    p2_list_5.append(PARAM.P2_Online_deque[i][time_list[4]])
    p2_list_6.append(PARAM.P2_Online_deque[i][time_list[5]])

fig26 = plt.figure(figsize=(15.7, 8))
ax32  = fig26.add_subplot(231)
ax33  = fig26.add_subplot(232)
ax34  = fig26.add_subplot(233)
ax35  = fig26.add_subplot(234)
ax36  = fig26.add_subplot(235)
ax37  = fig26.add_subplot(236)

xticks = np.arange(1, batch_length+1)

# subfigure 1
ax32.plot(xticks, p1_list_1, label=r'$c_1$',
          linestyle='-.', color='#cf6a87', linewidth=2.5, dash_capstyle='round', zorder=1, alpha=0.95)
ax32.plot(xticks, p2_list_1, label=r'$c_2$',
          linestyle='-.', color='#546de5', linewidth=2.5, dash_capstyle='round', zorder=1, alpha=0.95)

ax32.legend(prop=font_legend, frameon=True, loc='upper right', ncol=2)

ax32.xaxis.set_major_locator(MultipleLocator(5))
ax32.set_xlim([12.7, batch_length+0.2])

ax32.set_ylim([0.496, 0.504])
ax32.yaxis.set_major_locator(MultipleLocator(0.004))
ax32.set_xlabel("Batch ($k$)", family='Arial', size=12, weight='normal', labelpad=5)
ax32.set_ylabel("Integration Factor", family='Arial', size=12, weight='normal', labelpad=2)

ax32.tick_params(axis='both', direction="in", right=True, top=True)

ax32.text(18, 0.4965, 'time 0{}'.format(time_list[0]), fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':16})

# subfigure 2
ax33.plot(xticks, p1_list_2, label=r'$c_1$',
          linestyle='-.', color='#cf6a87', linewidth=2.5, dash_capstyle='round', zorder=1, alpha=0.95)
ax33.plot(xticks, p2_list_2, label=r'$c_2$',
          linestyle='-.', color='#546de5', linewidth=2.5, dash_capstyle='round', zorder=1, alpha=0.95)

ax33.legend(prop=font_legend, frameon=True, loc='upper right', ncol=2)

ax33.xaxis.set_major_locator(MultipleLocator(5))
ax33.set_xlim([12.7, batch_length+0.2])

ax33.set_ylim([0.48, 0.52])

ax33.yaxis.set_major_locator(MultipleLocator(0.008))
ax33.set_xlabel("Batch ($k$)", family='Arial', size=12, weight='normal', labelpad=5)
ax33.set_ylabel("Integration Factor", family='Arial', size=12, weight='normal', labelpad=2)

ax33.tick_params(axis='both', direction="in", right=True, top=True)

ax33.text(18, 0.48243, 'time {}'.format(time_list[1]), fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':16})

# subfigure 3
ax34.plot(xticks, p1_list_3, label=r'$c_1$',
          linestyle='-.', color='#cf6a87', linewidth=2.5, dash_capstyle='round', zorder=1, alpha=0.95)
ax34.plot(xticks, p2_list_3, label=r'$c_2$',
          linestyle='-.', color='#546de5', linewidth=2.5, dash_capstyle='round', zorder=1, alpha=0.95)

ax34.legend(prop=font_legend, frameon=True, loc='upper right', ncol=2)

ax34.xaxis.set_major_locator(MultipleLocator(5))
ax34.set_xlim([12.7, batch_length+0.2])

ax34.set_ylim([0.496, 0.504])
ax34.yaxis.set_major_locator(MultipleLocator(0.004))
ax34.set_xlabel("Batch ($k$)", family='Arial', size=12, weight='normal', labelpad=5)
ax34.set_ylabel("Integration Factor", family='Arial', size=12, weight='normal', labelpad=2)

ax34.tick_params(axis='both', direction="in", right=True, top=True)

ax34.text(18, 0.4965, 'time {}'.format(time_list[2]), fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':16})

# subfigure 4
ax35.plot(xticks, p1_list_4, label=r'$c_1$',
          linestyle='-.', color='#cf6a87', linewidth=2.5, dash_capstyle='round', zorder=1, alpha=0.95)
ax35.plot(xticks, p2_list_4, label=r'$c_2$',
          linestyle='-.', color='#546de5', linewidth=2.5, dash_capstyle='round', zorder=1, alpha=0.95)

ax35.legend(prop=font_legend, frameon=True, loc='upper right', ncol=2)

ax35.xaxis.set_major_locator(MultipleLocator(5))
ax35.set_xlim([12.7, batch_length+0.2])

ax35.set_ylim([0.496, 0.504])
ax35.yaxis.set_major_locator(MultipleLocator(0.004))
ax35.set_xlabel("Batch ($k$)", family='Arial', size=12, weight='normal', labelpad=5)
ax35.set_ylabel("Integration Factor", family='Arial', size=12, weight='normal', labelpad=2)

ax35.tick_params(axis='both', direction="in", right=True, top=True)

ax35.text(18, 0.4965, 'time {}'.format(time_list[3]), fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':16})

# subfigure 5
ax36.plot(xticks, p1_list_5, label=r'$c_1$',
          linestyle='-.', color='#cf6a87', linewidth=2.5, dash_capstyle='round', zorder=1, alpha=0.95)
ax36.plot(xticks, p2_list_5, label=r'$c_2$',
          linestyle='-.', color='#546de5', linewidth=2.5, dash_capstyle='round', zorder=1, alpha=0.95)

ax36.legend(prop=font_legend, frameon=True, loc='upper right', ncol=2)

ax36.xaxis.set_major_locator(MultipleLocator(5))
ax36.set_xlim([12.7, batch_length+0.2])

ax36.set_ylim([0.496, 0.504])
ax36.yaxis.set_major_locator(MultipleLocator(0.004))
ax36.set_xlabel("Batch ($k$)", family='Arial', size=12, weight='normal', labelpad=5)
ax36.set_ylabel("Integration Factor", family='Arial', size=12, weight='normal', labelpad=2)

ax36.tick_params(axis='both', direction="in", right=True, top=True)

ax36.text(18, 0.4965, 'time {}'.format(time_list[4]), fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':16})

# subfigure 6
ax37.plot(xticks, p1_list_6, label=r'$c_1$',
          linestyle='-.', color='#cf6a87', linewidth=2.5, dash_capstyle='round', zorder=1, alpha=0.95)
ax37.plot(xticks, p2_list_6, label=r'$c_2$',
          linestyle='-.', color='#546de5', linewidth=2.5, dash_capstyle='round', zorder=1, alpha=0.95)

ax37.legend(prop=font_legend, frameon=True, loc='upper right', ncol=2)

ax37.xaxis.set_major_locator(MultipleLocator(5))
ax37.set_xlim([12.7, batch_length+0.2])

ax37.set_ylim([0.496, 0.504])
ax37.yaxis.set_major_locator(MultipleLocator(0.004))
ax37.set_xlabel("Batch ($k$)", family='Arial', size=12, weight='normal', labelpad=5)
ax37.set_ylabel("Integration Factor", family='Arial', size=12, weight='normal', labelpad=2)

ax37.tick_params(axis='both', direction="in", right=True, top=True)

ax37.text(18, 0.4965, 'time {}'.format(time_list[5]), fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':16})

fig26.subplots_adjust(wspace=0.25,hspace=0.2)

# 保存图片
plt.savefig('./Data/Figures/Fig13 - Policy integration factor at six randomly disturbed time points (27, 306, 553, 672, 784, 859) under Scenario 2.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************







# ******************************************************************************************************************
# todo: Fig.8：Tracking RMSE along batches with CTA-DDPG, zero modification ILC and nearest modification ILC in Scenario 1-3.
PARAM = Sys_Init()
PARAM.Phase_Index = 0
Agent_Plot = DDPG_Plot()
Agent_Plot.__load_Traj_Online_No_STER__(PARAM)

s00_rmse_online = []
rmse_online = []

s00_rmse_zero = []
rmse_zero = []

s00_rmse_near = []
rmse_near = []

for index in range(1, 4):
    PARAM.Scen_Index = index
    Agent_Plot.__load_Traj_Online_No_STER__(PARAM)
    Agent_Plot.__load_Traj_ILC_Zero__(PARAM)
    Agent_Plot.__load_Traj_ILC_Near__(PARAM)
    phase_length = PARAM.Expected_step_reward_Online_list.__len__()

    for i in range(phase_length):
        err_traj_list = PARAM.errt_traj_Online_deque[i]
        s00_rmse_online.append(Agent_Plot.__cal_RMSE__(err_traj_list))
        err_traj_list = PARAM.err_ilc_zero_traj_deque[i]
        s00_rmse_zero.append(Agent_Plot.__cal_RMSE__(err_traj_list))
        err_traj_list = PARAM.err_ilc_near_traj_deque[i]
        s00_rmse_near.append(Agent_Plot.__cal_RMSE__(err_traj_list))
        pass
    rmse_online.extend(s00_rmse_online)
    rmse_zero.extend(s00_rmse_zero)
    rmse_near.extend(s00_rmse_near)
    s00_rmse_online = []
    s00_rmse_zero = []
    s00_rmse_near = []

batch_length = len(rmse_online)

fig29 = plt.figure(figsize=(7.5, 4))
ax42  = fig29.add_subplot(111)

xticks = np.arange(1, batch_length+1)

ax42.plot(xticks, rmse_online, label='CTA-DDPG', \
         linestyle='-', color='#EE1F23', linewidth=1.5, dash_capstyle='round', zorder=1)
ax42.plot(xticks, rmse_zero, label='Zero ILC', \
         linestyle='--', color='#0D7F3F', linewidth=1.5, dash_capstyle='round', zorder=1)
ax42.plot(xticks, rmse_near, label='Nearest ILC', \
         linestyle='-.', color='#0B72BA', linewidth=1.5, dash_capstyle='round', zorder=1)

ax42.axvline(x=21, color='dimgrey', linewidth=2.5, dash_capstyle='round', linestyle='--', alpha=0.65)
ax42.axvline(x=41, color='dimgrey', linewidth=2.5, dash_capstyle='round', linestyle='--', alpha=0.65)

ax42.legend(prop=font_legend, frameon=True, loc='upper right')

ax42.set_xlim([0.5, batch_length+0.5])

plt.xlabel("Batch ($k$)", family='Arial', size=12, weight='normal', labelpad=7)

xticks1 = [0, 5, 10, 15, 20]
xticks  = []
xticks.extend(xticks1)
xticks1 = [5, 10, 15, 20]
xticks1  =list(map(lambda x: x+20, xticks1))
xticks.extend(xticks1)
xticks1  =list(map(lambda x: x+20, xticks1))
xticks.extend(xticks1)
xticks.extend([65, 70, 75, 80])

xlabels = [0, 5, 10, 15, 20]
xlabels.extend([5, 10, 15, 20])
xlabels.extend([5, 10, 15, 20, 25, 30, 35, 40])

ax42.set_xlim([0+0.5, batch_length+0.5])

ax42.xaxis.set_major_locator(MultipleLocator(5))
ax42.set_xticks(xticks, xlabels)

ax42.set_ylim([-0.02, 0.7])
ax42.yaxis.set_major_locator(MultipleLocator(0.1))
# plt.ylabel("RMSE", family='Times New Roman', size=16, weight='normal', labelpad=5)
plt.ylabel("RMSE", family='Arial', size=12, weight='normal', labelpad=5)

ax42.tick_params(axis='both', direction="in", right=True, top=True)

ax42.text(4-0.5, 0.63, 'Scenario 1', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':14})
ax42.text(25-0.5, 0.63, 'Scenario 2', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':14})
ax42.text(45, 0.63, 'Scenario 3', fontdict={'family':'Arial', 'style':'italic', 'weight':'bold', 'color':'#2f3542', 'fontsize':14})

plt.savefig('./Data/Figures/Fig8 - Tracking RMSE along batches with CTA-DDPG, zero modification ILC and nearest modification ILC in Scenario 1-3..png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************







# ******************************************************************************************************************
# todo: Fig.9：Tracking errors in 0-200 time points for CTA-DDPG, zero modification ILC and nearest modification ILC in Scenario 1.
PARAM = Sys_Init()
PARAM.Phase_Index = 0
PARAM.Scen_Index  = 1
Agent_Plot = DDPG_Plot()
Agent_Plot.__load_Traj_Online_No_STER__(PARAM)

Agent_Plot.__load_Traj_Online_No_STER__(PARAM)
Agent_Plot.__load_Traj_ILC_Zero__(PARAM)
Agent_Plot.__load_Traj_ILC_Near__(PARAM)

batch_length = PARAM.errt_traj_Online_deque.__len__()

batch_scale = 5
time_scale  = 11
scaler      = 20

err_regular_time   = []  
err_regular_batch  = []  
err_regular_batch_list = []

err_mismatch_time  = []
err_mismatch_batch = []
err_mismatch_batch_list = []

err_zero_time      = []
err_zero_batch     = []
err_zero_batch_list = []

err_near_time      = []
err_near_batch     = []
err_near_batch_list = []

for i in range(batch_scale):
    err_regular_batch_list.append(PARAM.errt_traj_Online_deque[i])
    err_zero_batch_list.append(PARAM.err_ilc_zero_traj_deque[i])
    err_near_batch_list.append(PARAM.err_ilc_near_traj_deque[i])

for t in range(time_scale):
    for item in err_regular_batch_list:
        err_regular_time.append(item[scaler*t])
    err_regular_batch.append(err_regular_time)
    err_regular_time = []

    for item in err_mismatch_batch_list:
        err_mismatch_time.append(item[scaler*t])
    err_mismatch_batch.append(err_mismatch_time)
    err_mismatch_time = []

    for item in err_zero_batch_list:
        err_zero_time.append(item[scaler*t])
    err_zero_batch.append(err_zero_time)
    err_zero_time = []

    for item in err_near_batch_list:
        err_near_time.append(item[scaler*t])
    err_near_batch.append(err_near_time)
    err_near_time = []

mean_data_regular = []  
bar_data_regular  = np.zeros([2, time_scale])

mean_data_mismatch = []  
bar_data_mismatch  = np.zeros([2, time_scale])

mean_data_zero = []  
bar_data_zero  = np.zeros([2, time_scale])

mean_data_near = []  
bar_data_near  = np.zeros([2, time_scale])


for i in range(time_scale):
    mean_value_regular  = (sum(err_regular_batch[i]) / len(err_regular_batch[i]))[0]     
    max_value_regular   = max(err_regular_batch[i])
    min_value_regular   = min(err_regular_batch[i])
    upper_limit_regular = max_value_regular - mean_value_regular
    lower_limit_regular = mean_value_regular - min_value_regular
    mean_data_regular.append(mean_value_regular)
    bar_data_regular[0,i] = lower_limit_regular
    bar_data_regular[1,i] = upper_limit_regular

    mean_value_zero  = (sum(err_zero_batch[i]) / len(err_zero_batch[i]))[0]        
    max_value_zero   = max(err_zero_batch[i])
    min_value_zero   = min(err_zero_batch[i])
    upper_limit_zero = max_value_zero - mean_value_zero
    lower_limit_zero = mean_value_zero - min_value_zero
    mean_data_zero.append(mean_value_zero)
    bar_data_zero[0,i] = lower_limit_zero
    bar_data_zero[1,i] = upper_limit_zero

    mean_value_near  = (sum(err_near_batch[i]) / len(err_near_batch[i]))[0]     
    max_value_near   = max(err_near_batch[i])
    min_value_near   = min(err_near_batch[i])
    upper_limit_near = max_value_near - mean_value_near
    lower_limit_near = mean_value_near - min_value_near
    mean_data_near.append(mean_value_near)
    bar_data_near[0,i] = lower_limit_near
    bar_data_near[1,i] = upper_limit_near
    
xticks = np.arange(1, time_scale+1)

fig23 = plt.figure(figsize=(7.2, 4))
ax28  = fig23.add_subplot()

ax28.errorbar(xticks, mean_data_regular, yerr=bar_data_regular[:,0:time_scale],
              fmt='o-', color='#cf6a87', ecolor='#cf6a87', capsize=6, elinewidth=2, capthick=3, label='CTA-DDPG')
ax28.errorbar(xticks, mean_data_zero, yerr=bar_data_zero[:,0:time_scale],
              fmt='o-', color='#ffa502', ecolor='#ffa502', capsize=6, elinewidth=2, capthick=3, label='Zero ILC')
ax28.errorbar(xticks, mean_data_near, yerr=bar_data_near[:,0:time_scale],
              fmt='o-', color='#546de5', ecolor='#546de5', capsize=6, elinewidth=2, capthick=3, label='Near ILC')

ax28.set_xlabel("Time $(t)$", family='Times New Roman', size=12, weight='normal', labelpad=7)
ax28.set_ylabel(' Err $e_{t}$', family='Arial', size=12, weight='normal', labelpad=5)

ax28.legend(prop=font_legend, frameon=True, loc='upper left')

ax28.set_xticks(xticks, np.arange(time_scale)*scaler)

ax28.tick_params(axis='both', direction="in", right=True, top=True)

plt.savefig('./Data/Figures/Fig9 - Tracking errors in 0-200 time points for CTA-DDPG, zero modification ILC and nearest modification ILC in Scenario 1.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
# ******************************************************************************************************************






plt.show()
# ------------------------------------------------------------------------------------------ 
# ============================================== End of File  =============================================== #



