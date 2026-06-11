#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colormaps as mltcmap
import importlib


###### Environment Configuration ######
import _bootstrap
import Configuration
Root_Folder = Configuration.PROJECT_FOLDER
Data_Folder = Configuration.DATA_FOLDER
Output_Folder = Configuration.OUTPUT_FOLDER
Release_Folder = Configuration.RESULT_FOLDER

###### import Personal Package ######
import Domain
import DataCleaning as DC
import importlib

###### Import Models
###### Model import ######
import Models.Persistent as Persistent
import Models.ARFixLogit as ARFixLogit
import Models.ARFlxLogit as ARFlxLogit
import Models.BayesFix as BayesFix
import Models.RLS as RLS
import Models.RMLE as RMLE
import Models.BayesFlx as BayesFlx
import Evaluation.EvaluateCRPS as EvaluateCRPS
from DataObjectModels.ProbPredResultDO import IProbaPredResult


# # Color Palette / Line Scheme Configuration
# * https://www.color-hex.com/color-palette/111710

# In[ ]:


#region color palette

###### Color rgb value
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


def plot_colortable(colors, *, ncols=4, sort_colors=True):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig

###### Model EnDevgi Name
Persistence_DevName = r"Persistent"
ARL_DevName = r"AR Logit Default"
ARLNu_DevName = r"AR Flx Logit"
RLS_DevName = r"RLS"
sRLS_DevName=r"sRLS"
RMLE_DevName = r"RMLE"
sRMLE_DevName = r"sRMLE"
Bayes_DevName = r"Bayes"
BayesNu_DevName = r"Ada Bayes"
model_dev_ordered_names = [Persistence_DevName,ARL_DevName,ARLNu_DevName,
                   RLS_DevName,sRLS_DevName,RMLE_DevName,sRMLE_DevName,
                   Bayes_DevName,BayesNu_DevName]

###### Model Legend
Persitence_Name = r"Persistence"
ARL_Name = r"AR-$L$"
ARLNu_Name = r"AR-$L_\nu$"
RLS_Name = r"RLS"
sRLS_Name=r"sRLS"
RMLE_Name = r"RMLE"
sRMLE_Name = r"sRMLE"
Bayes_Name = r"Bayes"
BayesNu_Name = r"Bayes-$\nu$"
model_legend_order_name=[Persitence_Name,ARL_Name, ARLNu_Name,
                        RLS_Name,sRLS_Name,RMLE_Name,sRMLE_Name,
                        Bayes_Name,BayesNu_Name]

###### Model dev Legend map
Model_Dev_Legend_Map={
    Persistence_DevName:Persitence_Name,
    ARL_DevName:ARL_Name,
    ARLNu_DevName:ARLNu_Name,
    RLS_DevName:RLS_Name,
    sRLS_DevName:sRLS_Name,
    RMLE_DevName:RMLE_Name,
    sRMLE_DevName:sRMLE_Name,
    Bayes_DevName:Bayes_Name,
    BayesNu_DevName:BayesNu_Name
}

COLOR_PALETTE = {
    Persitence_Name: "#999999",   # gray

    ARL_Name: "#53778a",          # blue gray
    ARLNu_Name: "#2d405a",        # dark blue gray

    RLS_Name: "#64d133",          # green
    sRLS_Name: "#9be564",         # lighter green (variant)

    RMLE_Name: "#1a7a38",         # dark green
    sRMLE_Name: "#2ca25f",        # medium green

    Bayes_Name: "#ff9a00",        # orange
    BayesNu_Name: "#f86257",      # coral
}
plt.rcdefaults()
# _=plot_colortable(COLOR_PALETTE, ncols=3, sort_colors=False)


MARKER_PALETTE = {
    Persitence_Name: None,  # no marker

    ARL_Name: "o",
    ARLNu_Name: "o",

    RLS_Name: "v",
    sRLS_Name: "^",

    RMLE_Name: ">",
    sRMLE_Name: "<",

    Bayes_Name: "s",
    BayesNu_Name: "s",
}
#endregion


# # NaN Distribution

# In[ ]:


#region nan distribution
print("Figure NaN Distribution")
file_path =os.path.join(Output_Folder, f"DQ_Summary.csv")
df = pd.read_csv(file_path,index_col=[0,1])

plt.ioff()
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
fig, axes = plt.subplots(1,4, figsize=(12, 3),dpi=400)
_ = fig.tight_layout(pad=0)

binwidth = 2.5

years = [2020,2021,2022,2023]
for i, ax in enumerate(axes.flat):
    year = years[i]
    data = df[df.index.get_level_values(1) == year]["NanPcent"]
    bins=np.arange(min(data), max(data) + binwidth, binwidth)
    counts, bins, patches= ax.hist(data,bins=bins,alpha=0.5,color='gray',edgecolor='black')

    ### Find the bin that contains the mean value
    mean = np.nanmean(data)
    bin_idx = np.digitize(mean, bins) - 1  # Get the index of the bin containing the mean
    # print(f"({mean},{bin_idx})")
    if bin_idx >= 0 and bin_idx < len(patches):  # Ensure the index is valid
        _ = patches[bin_idx].set_facecolor('blue')  # Change the color of that bin to blue
        _ = patches[bin_idx].set_alpha(0.8)  # Optional: Adjust transparency
    
    _= ax.set_title(year)
    _= ax.set_xticks(np.arange(0,100.1,10))
    # _= ax.set_xticklabels(np.arange(0,100.1,10),rotation=45)
_ = axes[0].set_ylabel("Count")
_ = axes[0].set_xlabel("Percentage of Missing Data (%)")
_ = fig.savefig(os.path.join(Release_Folder,f"Fig.nan_describe.jpeg"),bbox_inches='tight')
_ = plt.close(fig)
_ = plt.rcdefaults()
_ = plt.ion()

#endregion


# # Ternary Chart

# In[ ]:


#region ternary chart
print("Figure Ternary Chart")
import ternary

dominator = (100 - df["NanPcent"])
dominator[dominator<=0.1] = 10/9
df["correct_factor"] = 100 / dominator
df["Pertlow"] = df["EplisonPcent"] * df["correct_factor"]
df["Perthigh"] = df["CleanEpsilonMaxPcent"] * df["correct_factor"]
df["Pertmid"] = 100-df["Pertlow"]-df["Perthigh"]
tdata = df[["Pertlow","Pertmid","Perthigh",]]

scale = 25
fontsize = 15
offset = 0.15


years=[2020,2021,2022,2023]

# Create a matplotlib figure with subplots
# plt.rcParams.update({'font.size': 10})
_ = plt.rcdefaults()
_ = plt.ioff()
plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
fig, axes = plt.subplots(1, 4, figsize=(20, 5),dpi=400)
_ = fig.tight_layout(pad=1)  

for i, ax in enumerate(axes.flat):
    year= years[i]
    data = [ tuple(row) for row in tdata[tdata.index.get_level_values(1)==year].values]
    _ = tax = ternary.TernaryAxesSubplot(scale=scale, ax=ax)
    _ = tax.boundary(linewidth=1.5)
    _ = tax.gridlines(color="gray", multiple=5,linewidth=0.5)
    _ = tax.set_title(f"{year}\n", fontsize=fontsize)

    if(i==0): # only for first chart
        _= tax.left_axis_label("Upper Bound Data Percentage (%)", fontsize=fontsize, offset=offset)
        _= tax.right_axis_label("Mid-Ranged Data Percentage (%)", fontsize=fontsize, offset=offset)
        _= tax.bottom_axis_label("Lower Bound Data Percentage (%)", fontsize=fontsize, offset=offset)

    _= tax.set_axis_limits({'b': [0, 25], 'l': [0, 25], 'r': [75, 100]})
    # get and set the custom ticks:
    _= tax.get_ticks_from_axis_limits(multiple=5)
    _= tax.set_custom_ticks(fontsize=10, offset=0.025,multiple=5)
    data_zoom = tax.convert_coordinates(data,axisorder='brl')
    # Draw lines parallel to the axes
    _= tax.horizontal_line(15,linewidth=2., color='orange', linestyle="--")
    _= tax.scatter(data_zoom,c="blue", s=3)
    _= tax.get_axes().axis('off')
    _= tax.clear_matplotlib_ticks()
    _= tax._redraw_labels()
    _= tax.resize_drawing_canvas() 
_ = fig.savefig(os.path.join(Release_Folder,f"Fig.bound_describe.jpeg"),bbox_inches='tight')

_ = plt.close(fig)
_ = plt.rcdefaults()
_ = plt.ion()

#endregion


# # $\nu$ distribtuion

# In[ ]:


#region nu distribution
print("Figure Nu Distribution")
###### Use ARFlx to have a feeling of distribution of $\nu$ ######
import Models.ARFlxLogit as ARFlxLogit
import Evaluation.ModelSelection as ModelSelection

###### Basic Model Configuration ######
Epsilon = 0.005
Resolution = 501

def Wrapper(year):
    data_file_path = os.path.join(Data_Folder, f"UBOR_{year}.csv")
    trainDataSetAider = DC.DFrameAider.DFrameAider().Unverbose().Load(data_file_path)

    nu_list = np.zeros(len(Domain.Constant.WINDFARMS)) * np.nan
    for i, windfarm in enumerate(Domain.Constant.WINDFARMS):
        if (not trainDataSetAider.TestWindfarmExist(windfarm)): continue
        X_train = trainDataSetAider.SelectWindFarm(windfarm).ScaleByCleanMax().BoundData(Epsilon).ToMatrixAider().GetData()
        
        ####### DriftAR Flx nu ######
        lags = ModelSelection.SelectOrder(X_train)
        ARFlxLogit_model = ARFlxLogit.ARFlxLogitModel(lags=lags, inital_nu=0.5, epsilon=Epsilon,resolution=Resolution)
        ARFlxLogit_model.Fit(X_train,nv_bounds=[0.1,5])
        nu_list[i] = ARFlxLogit_model.nu
    return nu_list

## 30s~60s for each 
nu_list_2020 = Wrapper(2020)
nu_list_2021 = Wrapper(2021)
nu_list_2022 = Wrapper(2022)
nu_list_2023 = Wrapper(2023)

###### Plotting 
plt.ioff()
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
fig, axes = plt.subplots(1,4, figsize=(12, 3),dpi=400)
fig.tight_layout(pad=0)

binwidth = 0.05
nus = (nu_list_2020,nu_list_2021,nu_list_2022,nu_list_2023)
titles=["2020","2021","2022","2023","2024"]
for i, ax in enumerate(axes.flat):
    data = nus[i]
    bins=np.arange(min(data), max(data) + binwidth, binwidth)
    counts, bins, patches= ax.hist(data,bins=bins,alpha=0.5,color='gray',edgecolor='black')

    ### Find the bin that contains the mean value
    mean = np.nanmean(data)
    bin_idx = np.digitize(mean, bins) - 1  # Get the index of the bin containing the mean
    # print(f"({mean},{bin_idx})")
    if bin_idx >= 0 and bin_idx < len(patches):  # Ensure the index is valid
        patches[bin_idx].set_facecolor('blue')  # Change the color of that bin to blue
        patches[bin_idx].set_alpha(0.8)  # Optional: Adjust transparency
    _= ax.set_xlim([0,3])
    _ = ax.set_title(titles[i])
_ = axes[0].set_ylabel("Count")
_ = axes[0].set_xlabel(r"$\nu$")
_ = fig.savefig(os.path.join(Release_Folder,f"Fig.nu_describe.jpeg"),bbox_inches='tight')
_ = plt.close(fig)
_ = plt.rcdefaults()
_ = plt.ion()
#endregion


# # CRPS Table (main paper)

# In[ ]:


#region crps table main
print("Table CRPS main")
import os
import pandas as pd

pd.options.display.float_format = "{:.5f}".format
metric_list = []

for year in [2021, 2022, 2023]:
    file_path = os.path.join(Output_Folder, f"_Metric_TestOn_{year}.csv")
    df = pd.read_csv(file_path, index_col=[0, 1])
    df = df[df.index.get_level_values(0).isin(Domain.Constant.GOOD_QUALITY_WINDFARMS)]

    res = df.groupby(level=1).mean() * 100
    res = res.rename(index=Model_Dev_Legend_Map)
    res = res.reindex(model_legend_order_name)

    temp = res[["CRPS", "Skill_Avg"]].copy()
    temp = temp.rename(columns={"CRPS": f"CRPS ({year})", "Skill_Avg": f"Skill ({year})"})

    metric_list.append(temp)

summary_df = pd.concat(metric_list, axis=1)
summary_df = summary_df.reset_index()
summary_df = summary_df.rename(columns={"index": "Model"})

column_order = ["Model"]
column_order += [f"CRPS ({year})" for year in [2021, 2022, 2023]]
column_order += [f"Skill ({year})" for year in [2021, 2022, 2023]]

summary_df = summary_df[column_order]

save_path = os.path.join(Release_Folder, "Table.CRPS_Skill_Mainpaper.csv")
summary_df.to_csv(save_path, index=False)
#endregion


# # CRPS Table (Appendice)

# In[ ]:


#region crps table appendice
print("Table CRPS Appendice")
import os
import pandas as pd

pd.options.display.float_format = "{:.5f}".format
metric_list = []

for year in [2021, 2022, 2023]:
    file_path = os.path.join(Output_Folder, f"_Metric_TestOn_{year}.csv")
    df = pd.read_csv(file_path, index_col=[0, 1])

    res = df.groupby(level=1).mean() * 100
    res = res.rename(index=Model_Dev_Legend_Map)
    res = res.reindex(model_legend_order_name)

    temp = res[["CRPS", "Skill_Avg"]].copy()
    temp = temp.rename(columns={"CRPS": f"CRPS ({year})", "Skill_Avg": f"Skill ({year})"})

    metric_list.append(temp)

summary_df = pd.concat(metric_list, axis=1)
summary_df = summary_df.reset_index()
summary_df = summary_df.rename(columns={"index": "Model"})

column_order = ["Model"]
column_order += [f"CRPS ({year})" for year in [2021, 2022, 2023]]
column_order += [f"Skill ({year})" for year in [2021, 2022, 2023]]

summary_df = summary_df[column_order]

save_path = os.path.join(Release_Folder, "Table.CRPS_Skill_Appendix.csv")
summary_df.to_csv(save_path, index=False)
#endregion


# # Rank Table

# In[ ]:


#region rank
print("Table Rank")
import os
import pandas as pd

pd.options.display.float_format = "{:.5f}".format

rank_stats_list = []

for year in [2021, 2022, 2023]:
    file_path = os.path.join(Output_Folder, rf"_Metric_TestOn_{year}.csv")
    df = pd.read_csv(file_path, index_col=[0, 1])

    df = df[df.index.get_level_values(0).isin(Domain.Constant.GOOD_QUALITY_WINDFARMS)]["CRPS"]

    rank_res = df.groupby(level=0).rank(ascending=True).astype(int)

    rank_stats = rank_res.groupby(level=1).value_counts()
    rank_stats = rank_stats.to_frame(name="Count")
    rank_stats.reset_index(inplace=True)
    rank_stats.columns = ["Method", "Rank", "Count"]

    rank_stats["Year"] = year
    rank_stats_list.append(rank_stats)

rank_stats_all = pd.concat(rank_stats_list, axis=0, ignore_index=True)

rank_stats_all["Method"] = rank_stats_all["Method"].map(Model_Dev_Legend_Map)

rank_table = rank_stats_all.groupby(["Method", "Rank"])["Count"].sum()
rank_table = rank_table.reset_index()
rank_table = rank_table.pivot_table(index="Method", columns="Rank", values="Count", fill_value=0)
rank_table = rank_table.astype(int)
rank_table = rank_table.reindex(model_legend_order_name)
save_path = os.path.join(Release_Folder, "Table.Rank.csv")
rank_table.to_csv(save_path)
#endregion


# # Violin Plot

# In[ ]:


#region violin
print("Figure Violin Plot")
import seaborn as sns
import numpy as np

def GetSkillData(year):
    file_path = os.path.join(Output_Folder,rf"_Metric_TestOn_{year}.csv")
    df = pd.read_csv(file_path,index_col=[0,1])
    data = df[df.index.get_level_values(1)!="Persistent"].copy(deep=True)
    data = data[data.index.get_level_values(0).isin(Domain.Constant.GOOD_QUALITY_WINDFARMS)]
    data["model"] = data.index.get_level_values(1)
    data["Skill"] = data["Skill_Avg"]*100
    return data

plt.ioff()
plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})
plt.rc('axes', axisbelow=True)


years = [2021,2022,2023]
fig, axes = plt.subplots(1,3,figsize=(36, 12), dpi=400)

for i, ax in enumerate(axes.flat):
    data = GetSkillData(years[i])
    _= sns.violinplot(
        data=data,
        x='model',
        y='Skill',
        split=True,
        inner="quart",
        density_norm='count',
        bw_method=0.2,
        ax=ax,
        fill=True,
        color='#87cefa',
        edgecolor='black', linewidth=1.5,
        order=model_dev_ordered_names,
    )
    _= ax.grid(color='gray', linestyle='-', axis="y",alpha=.5,zorder=0)
    _= ax.set_yticks(np.arange(-100,100,5))
    _= ax.set_xticks(ticks=[0, 1, 2,3,4,5,6,7,8],labels=model_legend_order_name, )
    _= ax.tick_params(axis='x', labelrotation=45)
    if(i==0): _= ax.set_ylim([-10,16]) 
    if(i==1): _= ax.set_ylim([-10,16]) 
    if(i==2): _= ax.set_ylim([-10,16]) 
    _= ax.set_xlabel("")
    _= ax.set_ylabel("")
    if(i==0):_= ax.set_ylabel("Skill Score (%)")
    _= ax.set_title(f"{years[i]}")
_ = fig.savefig(os.path.join(Release_Folder,f"Fig.Skill.jpeg"),bbox_inches='tight')
_ = plt.close(fig)
_ = plt.rcdefaults()
_ = plt.ion()
#endregion


# # Reliability Diagrams

# In[ ]:


#region reliability
print("Figure Reliability")
import copy
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.functional import fboxplot

### For CRPS Data
Data_Year = 2022
file_path = os.path.join(Output_Folder,f"_PPPlot_TestOn_{Data_Year}.csv")
df = pd.read_csv(file_path,index_col=[0,1])
df = df[df.index.get_level_values(0).isin(Domain.Constant.GOOD_QUALITY_WINDFARMS)]

def GetPPPlotPointss(df, windfarm):
    res = {
        Persitence_Name:df.loc[(windfarm,Persistence_DevName)].to_numpy(),
        ARL_Name:df.loc[(windfarm,ARL_DevName)].to_numpy(),
        ARLNu_Name:df.loc[(windfarm,ARLNu_DevName)].to_numpy(),
        RLS_Name:df.loc[(windfarm,RLS_DevName)].to_numpy(),
        sRLS_Name:df.loc[(windfarm,sRLS_DevName)].to_numpy(),
        RMLE_Name:df.loc[(windfarm,RMLE_DevName)].to_numpy(),
        sRMLE_Name:df.loc[(windfarm,sRMLE_DevName)].to_numpy(),
        Bayes_Name:df.loc[(windfarm,Bayes_DevName)].to_numpy(),
        BayesNu_Name:df.loc[(windfarm,BayesNu_DevName)].to_numpy(),
    }
    return res

pppoint = GetPPPlotPointss(df,"T_BRBEO-1")
pppoint2 = GetPPPlotPointss(df,"T_AKGLW-2")
pppoint3 = GetPPPlotPointss(df,"E_MINSW-1")

### For Functional Data
df2021 = pd.read_csv(os.path.join(Output_Folder,f"_PPPlot_TestOn_2021.csv"),index_col=[0,1])
df2021 = df2021[df2021.index.get_level_values(0).isin(Domain.Constant.GOOD_QUALITY_WINDFARMS)]
df2022 = pd.read_csv(os.path.join(Output_Folder,f"_PPPlot_TestOn_2022.csv"),index_col=[0,1])
df2022 = df2022[df2022.index.get_level_values(0).isin(Domain.Constant.GOOD_QUALITY_WINDFARMS)]
df2023 = pd.read_csv(os.path.join(Output_Folder,f"_PPPlot_TestOn_2023.csv"),index_col=[0,1])
df2023 = df2023[df2023.index.get_level_values(0).isin(Domain.Constant.GOOD_QUALITY_WINDFARMS)]


plt.rcParams.update({"font.size": 20, "font.family": "serif"})
_ = plt.ioff()
fig, axes = plt.subplots(3, 4, figsize=(36, 27), dpi=400)
fig.tight_layout(pad=1.0)
_ = plt.subplots_adjust(wspace=0.025, hspace=0.1)
PPPlot_Resolution = 100
probas = np.linspace(Epsilon, 1 - Epsilon, PPPlot_Resolution)
cmap_outliers = plt.colormaps["coolwarm"]
stride = 5

sample_titles = ["T_BRBEO-1", "T_AKGLW-2", "E_MINSW-1"]
sample_pppoints = [pppoint, pppoint2, pppoint3]

functional_dev_methods = [
    Persistence_DevName,
    ARL_DevName,
    ARLNu_DevName,
    RLS_DevName,
    sRLS_DevName,
    RMLE_DevName,
    sRMLE_DevName,
    Bayes_DevName,
    BayesNu_DevName,
]

def draw_sample_panel(ax, pppoint_data, title, show_y=False, need_legend_plot=False):
    _ = ax.plot([0, 1], [0, 1], color="black", label="Ideal")

    for model, res in pppoint_data.items():
        _ = ax.plot(
            probas,
            res,
            color=COLOR_PALETTE[model],
            linewidth=2,
            marker=MARKER_PALETTE[model],
            markersize=4,
            label=model,
        )

    handles, labels = ax.get_legend_handles_labels()
    handles = [copy.copy(ha) for ha in handles]
    _ = [ha.set_linewidth(6) for ha in handles]
    _ = [ha.set_markersize(16) for ha in handles if ha.get_marker() is not None]

    if need_legend_plot:
        _ = ax.legend(handles=handles, labels=labels, fontsize=20)
        need_legend_plot = False
    _ = ax.scatter(0.5, 0.5, marker="D", s=20, color="black")
    _ = ax.set_title(title, fontsize=30, fontweight="bold", fontfamily="serif")
    _ = ax.set_xlim([0, 1])
    _ = ax.set_ylim([0, 1])
    _ = ax.grid()

    if show_y:
        _ = ax.set_xticks(np.linspace(0, 1, 11), labels=[])
        _ = ax.set_yticks(np.linspace(0, 1, 11), labels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        _ = ax.tick_params(axis="both", which="both", direction="out", top=True, bottom=True, labelsize=30)
    else:
        _ = ax.set_xticks(np.linspace(0, 1, 11), labels=[])
        _ = ax.set_yticks(np.linspace(0, 1, 11), labels=[])
        _ = ax.tick_params(axis="both", which="both", direction="out", top=True, bottom=True, labelsize=30)

def draw_functional_panel(ax, method_dev_name, show_x=False, show_y=False):
    data2021 = df2021[df2021.index.get_level_values(1) == method_dev_name][::stride].to_numpy()
    data2022 = df2022[df2022.index.get_level_values(1) == method_dev_name][::stride].to_numpy()
    data2023 = df2023[df2023.index.get_level_values(1) == method_dev_name][::stride].to_numpy()

    non_empty = [x for x in [data2021, data2022, data2023] if x.size > 0]
    if len(non_empty) == 0:
        ax.axis("off")
        print(f"Skip {method_dev_name}: no data found")
        return

    func_data = np.vstack(non_empty)

    _ = ax.grid()
    _ = fboxplot(
        func_data,
        xdata=probas,
        ax=ax,
        wfactor=3,
        plot_opts=dict(cmap_outliers=cmap_outliers, draw_nonout=False),
    )
    _ = ax.plot([0, 1], [0, 1], linestyle="--", c="gray")
    _ = ax.set_xlim([0, 1])
    _ = ax.set_ylim([0, 1])
    _ = ax.set_title(Model_Dev_Legend_Map[method_dev_name], fontsize=30, fontweight="bold")

    if show_x and show_y:
        _ = ax.set_xticks(np.linspace(0, 1, 11), labels=["", "", 0.2, "", 0.4, "", 0.6, "", 0.8, "", 1.0])
        _ = ax.set_yticks(np.linspace(0, 1, 11), labels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        _ = ax.tick_params(axis="both", which="both", direction="out", top=True, bottom=True, labelsize=30)
    elif show_y:
        _ = ax.set_xticks(np.linspace(0, 1, 11), labels=[])
        _ = ax.set_yticks(np.linspace(0, 1, 11), labels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        _ = ax.tick_params(axis="both", which="both", direction="out", top=True, bottom=True, labelsize=30)
    elif show_x:
        _ = ax.set_xticks(np.linspace(0, 1, 11), labels=["", "", 0.2, "", 0.4, "", 0.6, "", 0.8, "", 1.0])
        _ = ax.set_yticks(np.linspace(0, 1, 11), labels=[])
        _ = ax.tick_params(axis="both", which="both", direction="out", top=True, bottom=True, labelsize=30)
    else:
        _ = ax.set_xticks(np.linspace(0, 1, 11), labels=[])
        _ = ax.set_yticks(np.linspace(0, 1, 11), labels=[])
        _ = ax.tick_params(axis="both", which="both", direction="out", top=True, bottom=True, labelsize=30)

draw_sample_panel(axes[0, 0], sample_pppoints[0], sample_titles[0], show_y=True, need_legend_plot=True)
draw_sample_panel(axes[0, 1], sample_pppoints[1], sample_titles[1], show_y=False)
draw_sample_panel(axes[0, 2], sample_pppoints[2], sample_titles[2], show_y=False)
draw_functional_panel(axes[0, 3], functional_dev_methods[0], show_x=False, show_y=False)

draw_functional_panel(axes[1, 0], functional_dev_methods[1], show_x=False, show_y=True)
draw_functional_panel(axes[1, 1], functional_dev_methods[2], show_x=False, show_y=False)
draw_functional_panel(axes[1, 2], functional_dev_methods[3], show_x=False, show_y=False)
draw_functional_panel(axes[1, 3], functional_dev_methods[4], show_x=False, show_y=False)

draw_functional_panel(axes[2, 0], functional_dev_methods[5], show_x=True, show_y=True)
draw_functional_panel(axes[2, 1], functional_dev_methods[6], show_x=False, show_y=False)
draw_functional_panel(axes[2, 2], functional_dev_methods[7], show_x=False, show_y=False)
draw_functional_panel(axes[2, 3], functional_dev_methods[8], show_x=False, show_y=False)

_ = fig.savefig(os.path.join(Release_Folder, "Fig.Reliability.jpeg"), bbox_inches="tight")
_ = plt.close(fig)
_ = plt.rcdefaults()
_ = plt.ion()
#endregion


# # Case Study

# In[ ]:


#region case study
print("Figure Case Study")
import Evaluation.ModelSelection as ModelSelection
import Models.BayesFlx as BayesFlx
from  DataCleaning import ResultPostProcess as RPP

###### Model HyperParameter
Epsilon = 0.005

###### Data Configuration and Loding ######
Data_Year = 2021
Data_Year_Valid = Data_Year + 1
windfarm ="E_DALSW-1"
data_file_path = os.path.join(Data_Folder, f"UBOR_{Data_Year}.csv")
data_valid_file_path = os.path.join(Data_Folder, f"UBOR_{Data_Year_Valid}.csv")

###### Data Loading ######
Train_Data_Aider = DC.DFrameAider.DFrameAider().Load(data_file_path)
Test_Data_Aider = DC.DFrameAider.DFrameAider().Load(data_valid_file_path)
def LoadTrainTest(windfarm) : 
    X_train = Train_Data_Aider.SelectWindFarm(windfarm).ScaleByCleanMax().BoundData(Epsilon).ToMatrixAider().GetData()
    clean_max = Train_Data_Aider.SelectWindFarm(windfarm).GetCleanMax()
    X_test = Test_Data_Aider.SelectWindFarm(windfarm).ScaleBy(clean_max).BoundData(Epsilon).ToMatrixAider().GetData()
    return X_train,X_test
X_train,X_test= LoadTrainTest(windfarm)

### Lag selection
lags = ModelSelection.SelectOrder(X_train,max_lags=6)

### arflx benchmark
arflx = ARFlxLogit.ARFlxLogitModel(lags=lags)
_= arflx.Fit(X_train)

### Bayesian essai
Resolution = 501
best_nu = arflx.nu
numDim= len(lags) + 1
initmu = arflx.param.flatten()
precision = np.eye(numDim)/10000
Default_Lambda = 0.9995
bayesAda_m  = BayesFlx.BayesFlxLogitModel(lags=lags,mu=initmu, precision=precision,nu=best_nu, _forget=Default_Lambda, epsilon=Epsilon,resolution=Resolution)
bayesAda_m.Fit(X_train)
bayesAda_m_pred = bayesAda_m.ProbaPred(X_test)

origin = X_test
x = np.array(range(len(X_test)))
p25 = np.array(list(map(lambda x: np.nan if x is None else x, RPP.GetQuantilePred(bayesAda_m_pred,.25))) )
p50 = np.array(list(map(lambda x: np.nan if x is None else x, RPP.GetQuantilePred(bayesAda_m_pred,.50))) )
p75 = np.array(list(map(lambda x: np.nan if x is None else x, RPP.GetQuantilePred(bayesAda_m_pred,.75))) )
arr_filled = pd.Series(p50).ffill().to_numpy()
p25[np.isnan(p25)] = arr_filled[x[np.isnan(p25)]]
p75[np.isnan(p75)] =  arr_filled[np.isnan(p75)]

bandwidth = p75 - p25
indices = (p50 < 0.2) | (p50 > 0.8)
bw_extreme = bandwidth[indices]
bw_middle = bandwidth[~indices]
q_extreme = np.nanpercentile(bw_extreme, [5, 25, 50, 75, 95])
q_middle = np.nanpercentile(bw_middle, [5, 25, 50, 75, 95])

### Plotting Time Trunc
x = np.array(range(len(X_test)))
i0=  int(len(X_test) // 52 * 5)
i1 = int(len(X_test) // 52 * 7)
i2 = int(len(X_test) // 52 * 43)
i3 = int(len(X_test) // 52 * 45)

day_label_stride = 48 # fro a week 48 * 7
tick_position = np.array(range(len(X_test)))[::day_label_stride]
day_label = np.array(range(len(np.array(range(len(X_test))[::day_label_stride])))) + 1

########## Plotting 
plt.ioff()
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
plt.rc('axes', axisbelow=True)
fig, axes = plt.subplots(1,4,figsize=(16,24), dpi=400)

###### a case in H1 Forecast
ax= axes[0]
_= ax.invert_yaxis()
_= ax.plot(p50[i0:i1],x[i0:i1], label='Prediction', color='blue',alpha=0.8,linewidth=0.5)
_= ax.plot(origin[i0:i1],x[i0:i1], label="Observed",color="k", linewidth=0.5)
_= ax.set_yticks(tick_position,day_label)
_= ax.set_ylim([i1,i0])
_= ax.grid(True, zorder=0)
### Fill
h1_non_nan_index = (~np.isnan(p25[i0:i1])) & (~np.isnan(p75[i0:i1]))
_= ax.fill_betweenx(x[i0:i1][h1_non_nan_index], p25[i0:i1][h1_non_nan_index], p75[i0:i1][h1_non_nan_index], color='blue', alpha=0.2, label='50% Prediction Interval', edgecolor='none')

# Enable x-axis ticks and labels on both top and bottom
_= ax.set_xticks(np.arange(0,1.1, 0.2))
_= ax.set_xlim([-0.01,1.01])
_=ax.xaxis.set_label_position('top')  # Set x-axis label on top
_=ax.tick_params(axis='x', which='both', direction='out', top=True, bottom=True)  # Enable ticks on both
_=ax.xaxis.set_tick_params(labeltop=True, labelbottom=True)  # Enable labels on both

_= ax.set_ylabel("Day Number")
_= ax.set_xlabel("Power Generation")

###### a case in H2 Forecast
ax= axes[2]
_= ax.invert_yaxis()
_= ax.plot(p50[i2:i3],x[i2:i3], label='Prediction', color='blue',alpha=0.8,linewidth=0.5)
_= ax.plot(origin[i2:i3],x[i2:i3], label="Observed",color="k",linewidth=0.5)
_= ax.set_yticks(tick_position,day_label)
_= ax.set_ylim([i3,i2])
_= ax.grid(True, zorder=0)
### Fill
h2_non_nan_index = (~np.isnan(p25[i2:i3])) & (~np.isnan(p75[i2:i3]))
_= ax.fill_betweenx(x[i2:i3][h2_non_nan_index], p25[i2:i3][h2_non_nan_index], p75[i2:i3][h2_non_nan_index], color='blue', alpha=0.2, label='50% Prediction Interval', edgecolor='none')


# Enable x-axis ticks and labels on both top and bottom
_= ax.set_xticks(np.arange(0,1.1, 0.2))
_= ax.set_xlim([-0.01,1.01])
_=ax.xaxis.set_label_position('top')  # Set x-axis label on top
_=ax.tick_params(axis='x', which='both', direction='out', top=True, bottom=True)  # Enable ticks on both
_=ax.xaxis.set_tick_params(labeltop=True, labelbottom=True)  # Enable labels on both

###### a case in H1 nu
ax= axes[1]
_= ax.invert_yaxis()
_= ax.plot(bayesAda_m._nus[i0:i1],x[i0:i1], label="nu",color="k", linewidth=0.5)
_= ax.scatter(bayesAda_m._nus[i0:i1],x[i0:i1],c="k",s=.5)
_= ax.set_yticks(tick_position,day_label)
_= ax.set_ylim([i1,i0])
_= ax.set_xlim([np.min(bayesAda_m._nus),np.max(bayesAda_m._nus)])
_= ax.set_xlim([0.58,0.62])
_= ax.set_xticks(np.arange(0.58,0.62,0.008))
_= ax.xaxis.set_label_position('top')  # Set x-axis label on top
_= ax.tick_params(axis='x', which='both', direction='out', top=True, bottom=True)  # Enable ticks on both
_= ax.xaxis.set_tick_params(labeltop=True, labelbottom=True)  # Enable labels on both
_= ax.grid(True, zorder=0)
_= ax.set_xlabel(r"$\nu_t$")

###### a case in H2 nu
ax= axes[3]
_= ax.invert_yaxis()
_= ax.plot(bayesAda_m._nus[i2:i3],x[i2:i3], label="nu",color="k", linewidth=0.5)
_= ax.scatter(bayesAda_m._nus[i2:i3],x[i2:i3],c="k",s=.5)
_= ax.set_yticks(tick_position,day_label)
_= ax.set_ylim([i3,i2])
# _= ax.set_xlim([np.min(bayesAda_m._nus),np.max(bayesAda_m._nus)]
_= ax.set_xlim([0.65,0.70])
_= ax.set_xticks(np.arange(0.65,0.70,0.02))
_= ax.xaxis.set_label_position('top')  # Set x-axis label on top
_= ax.tick_params(axis='x', which='both', direction='out', top=True, bottom=True)  # Enable ticks on both
_= ax.xaxis.set_tick_params(labeltop=True, labelbottom=True)  # Enable labels on both
_= ax.grid(True, zorder=0)


_= fig.savefig(os.path.join(Release_Folder,"Fig.case_study.jpeg"), bbox_inches='tight')
_ = plt.close(fig)
_ = plt.rcdefaults()
_ = plt.ion()
#endregion


# # Sensitivity

# In[ ]:


#region sensitivity
print("Figure Sensitivity")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator

RLS_mu = pd.read_csv(rf"{Output_Folder}\RLS_mu_Wrapper.csv") * 100
RLS_Var = pd.read_csv(rf"{Output_Folder}\RLS_Var_Wrapper.csv") * 100
RLS_M = pd.read_csv(rf"{Output_Folder}\RLS_M_Wrapper.csv") * 100

SRLS_mu = pd.read_csv(rf"{Output_Folder}\SRLS_mu_Wrapper.csv") * 100
SRLS_Var = pd.read_csv(rf"{Output_Folder}\SRLS_Var_Wrapper.csv") * 100
SRLS_M = pd.read_csv(rf"{Output_Folder}\SRLS_M_Wrapper.csv") * 100

RMLE_mu = pd.read_csv(rf"{Output_Folder}\RMLE_mu_Wrapper.csv") * 100
RMLE_Var = pd.read_csv(rf"{Output_Folder}\RMLE_var_Wrapper.csv") * 100
RMLE_M = pd.read_csv(rf"{Output_Folder}\RMLE_M_Wrapper.csv") * 100
RMLE_nu = pd.read_csv(rf"{Output_Folder}\RMLE_nu_Wrapper.csv") * 100

SRMLE_mu = pd.read_csv(rf"{Output_Folder}\SRMLE_mu_Wrapper.csv") * 100
SRMLE_Var = pd.read_csv(rf"{Output_Folder}\SRMLE_var_Wrapper.csv") * 100
SRMLE_M = pd.read_csv(rf"{Output_Folder}\SRMLE_M_Wrapper.csv") * 100
SRMLE_nu = pd.read_csv(rf"{Output_Folder}\SRMLE_nu_Wrapper.csv") * 100

Bayes_mu = pd.read_csv(rf"{Output_Folder}\Bayes_mu_Wrapper.csv") * 100
Bayes_Var = pd.read_csv(rf"{Output_Folder}\Bayes_var_Wrapper.csv") * 100
Bayes_M = pd.read_csv(rf"{Output_Folder}\Bayes_M_Wrapper.csv") * 100

BayesAda_mu = pd.read_csv(rf"{Output_Folder}\BayesAda_mu_Wrapper.csv") * 100
BayesAda_Var = pd.read_csv(rf"{Output_Folder}\BayesAda_var_Wrapper.csv") * 100
BayesAda_P = pd.read_csv(rf"{Output_Folder}\BayesAda_P_Wrapper.csv") * 100
BayesAda_nu = pd.read_csv(rf"{Output_Folder}\BayesAda_nu_Wrapper.csv") * 100



plt.rcParams.update({"font.size": 20, "font.family": "serif"})
plt.ioff()

class RoundedOffsetScalarFormatter(ScalarFormatter):
    def __init__(self, decimals=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decimals = decimals

    def get_offset(self):
        if len(self.locs) == 0:
            return ""
        if self.orderOfMagnitude or self.offset:
            sci_part = ""
            offset_part = ""
            if self.orderOfMagnitude:
                if self._useMathText:
                    sci_part = r"$\times\mathdefault{10^{%d}}$" % self.orderOfMagnitude
                else:
                    sci_part = f"1e{self.orderOfMagnitude}"
            if self.offset:
                offset_str = f"{self.offset:.{self.decimals}f}"
                if self._useMathText:
                    offset_part = rf"$+\mathdefault{{{offset_str}}}$"
                else:
                    offset_part = f"+{offset_str}"
            return f"{sci_part}{offset_part}"
        return ""
    
def _plot_hist(ax, df, bench, ymax, title=None, ylabel=None, sci=None):
    _= ax.hist(df["res"], bins=10, alpha=0.5, color="gray", edgecolor="black")
    _= ax.plot([bench, bench], [0, ymax], color="b", linewidth=3)

    if title is not None: _= ax.set_title(title, fontsize=24)
    if ylabel is not None:
        _= ax.text(
            -0.1,
            0.5,
            ylabel,
            rotation=90,
            transform=ax.transAxes,
            verticalalignment="center",
            horizontalalignment="right",
            fontsize=24,
            fontweight="bold",
        )

    _= ax.set_ylim(0, ymax)
    _= ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    if sci is not None:
        formatter = RoundedOffsetScalarFormatter(useMathText=True)
        _= formatter.set_scientific(True)
        _= formatter.set_useOffset(sci["offset"])
        _= formatter.set_powerlimits(sci["powerlimits"])
        _= ax.xaxis.set_major_formatter(formatter)


def _plot_na(ax, title=None):
    _= ax.axis("off")
    _= ax.text(0.5, 0.5, "N/A", fontsize=24, ha="center", va="center")
    if title is not None:
        _= ax.set_title(title, fontsize=24)


fig, axes = plt.subplots(6, 4, figsize=(24, 36), dpi=400)
_ = fig.tight_layout(pad=1.5)

bench_map = {
    RLS_Name: 2.9434617,
    sRLS_Name: 5.4873321,
    RMLE_Name: 4.3915329,
    sRMLE_Name: 4.3920194,
    Bayes_Name: 4.8886956,
    BayesNu_Name: 5.9979651,
}
Y_MAX= 35
row_map = [
    {
        "label": RLS_Name,
        "bench": bench_map[RLS_Name],
        "mu": RLS_mu,
        "var": RLS_Var,
        "M": RLS_M,
        "nu": None,
        "ymax": [Y_MAX+15,Y_MAX,75, None],
        "sci": [None, None, None, None],
    },
    {
        "label": sRLS_Name,
        "bench": bench_map[sRLS_Name],
        "mu": SRLS_mu,
        "var": SRLS_Var,
        "M": SRLS_M,
        "nu": None,
        "ymax": [Y_MAX,Y_MAX,Y_MAX+15, None],
        "sci": [None, None, None, None],
    },
    {
        "label": RMLE_Name,
        "bench": bench_map[RMLE_Name],
        "mu": RMLE_mu,
        "var": RMLE_Var,
        "M": RMLE_M,
        "nu": RMLE_nu,
        "ymax": [Y_MAX,Y_MAX,75,Y_MAX],
        "sci": [None, None, None, None],
    },
    {
        "label": sRMLE_Name,
        "bench": bench_map[sRMLE_Name],
        "mu": SRMLE_mu,
        "var": SRMLE_Var,
        "M": SRMLE_M,
        "nu": SRMLE_nu,
        "ymax": [Y_MAX,Y_MAX,Y_MAX+15,Y_MAX],
        "sci": [None, None, None, None],
    },
    {
        "label": Bayes_Name,
        "bench": bench_map[Bayes_Name],
        "mu": Bayes_mu,
        "var": Bayes_Var,
        "M": Bayes_M,
        "nu": None,
        "ymax": [Y_MAX,Y_MAX,Y_MAX+15, None],
        "sci": [None, None, {"offset": bench_map[Bayes_Name], "powerlimits": (-6, 6)}, None],
    },
    {
        "label": BayesNu_Name,
        "bench": bench_map[BayesNu_Name],
        "mu": BayesAda_mu,
        "var": BayesAda_Var,
        "M": BayesAda_P,
        "nu": BayesAda_nu,
        "ymax": [Y_MAX,Y_MAX,Y_MAX+15,Y_MAX],
        "sci": [
            None,
            None,
            {"offset": bench_map[BayesNu_Name], "powerlimits": (-6, 6)},
            None,
        ],
    },
]


titles = [
    r"Disturbance on $\mu$",
    r"Disturbance on $\sigma_z^2$",
    r"Disturbance on $M$",
    r"Disturbance on $\nu$",
]


for r, spec in enumerate(row_map):
    _plot_hist(
        axes[r, 0],
        spec["mu"],
        spec["bench"],
        spec["ymax"][0],
        title=titles[0] if r == 0 else None,
        ylabel=spec["label"],
        sci=spec["sci"][0],
    )

    _plot_hist(
        axes[r, 1],
        spec["var"],
        spec["bench"],
        spec["ymax"][1],
        title=titles[1] if r == 0 else None,
        sci=spec["sci"][1],
    )

    _plot_hist(
        axes[r, 2],
        spec["M"],
        spec["bench"],
        spec["ymax"][2],
        title=titles[2] if r == 0 else None,
        sci=spec["sci"][2],
    )

    if spec["nu"] is None:
        _plot_na(axes[r, 3], title=titles[3] if r == 0 else None)
    else:
        _plot_hist(
            axes[r, 3],
            spec["nu"],
            spec["bench"],
            spec["ymax"][3],
            title=titles[3] if r == 0 else None,
            sci=spec["sci"][3],
        )


_= fig.savefig(os.path.join(Release_Folder, "Fig.Sensitivity.jpeg"), bbox_inches="tight")
_ = plt.close(fig)
_ = plt.rcdefaults()
_ = plt.ion()
#endregion


# # Diebold-Mariano Test

# In[ ]:


#region dm test
print("Figure Diebold Mariano Test")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("orange_blue", [plt.cm.coolwarm(0.6), plt.cm.coolwarm(0.0)])
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

DMT_2020 = pd.read_csv(os.path.join(Output_Folder,"_DM_Metric_TestOn_2021.csv"),index_col=[0,1,2])
DMT_2021 = pd.read_csv(os.path.join(Output_Folder,"_DM_Metric_TestOn_2022.csv"),index_col=[0,1,2])
DMT_2022 = pd.read_csv(os.path.join(Output_Folder,"_DM_Metric_TestOn_2023.csv"),index_col=[0,1,2])
DMT = pd.concat([DMT_2020,DMT_2021,DMT_2022])

### Only use Good Windfarms result
good_windfarms = set(Domain.Constant.GOOD_QUALITY_WINDFARMS)
DMT = DMT[DMT.index.get_level_values(0).isin(good_windfarms)]

## H0 :  d = 0 (reject H0 if |DM_statistics| > 1.93), H1: d < 0
H0_rejction = -1.627 #
DMT["Significance"] = (DMT["DM_statistics"] < H0_rejction).astype(int) #here Significance means reject H0.
totals = DMT['Significance'].groupby(level=[1,2]).count()

### DMT_percent
DMT_percent = DMT.groupby(level=[1,2]).sum()
DMT_percent['Significance'] = np.round(DMT_percent['Significance'] / totals * 100).astype(int)
DMT_percent = DMT_percent.rename(index=Model_Dev_Legend_Map, level=0)
DMT_percent = DMT_percent.rename(index=Model_Dev_Legend_Map, level=1)

plt.ioff()
fig, ax = plt.subplots(figsize=(10, 8),dpi=400)
method_order= model_legend_order_name
data_for_heatmap = DMT_percent['Significance'].unstack(level=1)  # rows=level1, cols=level2
data_for_heatmap = data_for_heatmap.reindex(index=method_order, columns=method_order)

# Create a mask for the diagonal
mask = np.eye(data_for_heatmap.shape[0], dtype=bool)  # True on diagonal, False elsewhere
annot = data_for_heatmap.copy().astype(str)
annot[mask] = "N/A"

# Plot
cbar_kws={'label': 'Test Pass Rate(%)',
        'shrink': 0.9,  # shrink height
        'pad': .15    # space between heatmap and colorbar
          }
_ = sns.heatmap(data_for_heatmap, annot=True, fmt="", cmap=cmap, cbar_kws=cbar_kws, ax=ax, mask=mask)

_= ax.tick_params(
    axis='both', which='both',
    bottom=True, top=True, left=True, right=True,
    labelbottom=True, labeltop=True, labelleft=True, labelright=True
)
# Rotate y-axis tick labels on both sides
_ = ax.yaxis.set_tick_params(rotation=0)
_ = ax.xaxis.set_tick_params(rotation=0)


_ = plt.tight_layout()


_ = fig.savefig(os.path.join(Release_Folder,"Fig.DM_Metric_Heatmap.png"), bbox_inches='tight', dpi=400)
_ = plt.close(fig)
_ = plt.rcdefaults()
_ = plt.ion()
#endregion

