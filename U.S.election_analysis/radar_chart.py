#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 22:27:12 2017

@author: yewonkaitlynpark
"""

import pandas as pd
import matplotlib.pyplot as plt
from math import pi

polarraw = pd.read_excel('polarpolar.xlsx')
polarrawt = polarraw.transpose()
polarrawt.columns = ["cluster 1", "cluster 2", "cluster 3","cluster 4","cluster 5"]
polarrawt = polarrawt.drop(['group'])
# ------- PART 1: Create background
 
# number of variable
N = 49
categories=list(["Age_20to29",
"Age_30to39",
"Age_40to49",
"Age_50to59",
"Age_60to80over",
"Educational_Attainment_Nohighschool",
"Educational_Attainment_Highschool",
"Educational_Attainment_Somecollege",
"Educational_Attainment_Bachelors_Degree",
"FamilyIncome_USD0_USD24999",
"FamilyIncome_USD25000_USD47499",
"FamilyIncome_USD47500_USD62499",
"FamilyIncome_USD62500_USD79999",
"FamilyIncome_USD80000_USD99999",
"FamilyIncome_USD100000_orMore",
"Gender_Male",
"Gender_Female",
"Race_White",
"Race_White_Male",
"Race_White_Female",
"Race_Asian",
"Race_Hispanic",
"Race_Black",
"Race_Black_Male",
"Race_Black_Female",
"MaritalStatus_Married",
"MaritalStatus_NeverMarried",
"MaritalStatus_Divorced/Separated",
"MaritalStatus_Widowed",
"Health_Insurance_Coverage_Insured",
"Health_Insurance_Coverage_Uninsured",
"LabourForce_StatusRecord_Working",
"LabourForce_StatusRecord_Unemployed",
"Turn_Out_Rate_Total",
"Turn_Out_Rate_Male",
"Turn_Out_Rate_Female",
"Turn_Out_Rate_White",
"Turn_Out_Rate_Black",
"Turn_Out_Rate_Asian",
"Turn_Out_Rate_Hispanic",
"Urbanity_by_NumberofCounty",
"Urbanity_State_in_Total",
"Trump_Num_of_Campaign_sinceAug",
"Clinton_Num_of_Campaign_sinceAug",
"Trump_Num_of_Campaign_last50d",
"Clinton_Num_of_Campaign_last50d",
"Trump_Num_of_Campaign_last2w",
"Clinton_Num_of_Campaign_last2w",
"Electoral_Votes_Number"])

categories2 = list(["20to29",
"30to39",
"40to49",
"50to59",
"60to80over",
"Nohighschool",
"Highschool",
"Somecollege",
"Bachelors_Degree",
"USD0_USD24999",
"USD25000_USD47499",
"USD47500_USD62499",
"USD62500_USD79999",
"USD80000_USD99999",
"USD100000_orMore",
"Gender_Male",
"Gender_Female",
"White",
"White_Male",
"White_Female",
"Asian",
"Hispanic",
"Black",
"Black_Male",
"Black_Female",
"Married",
"NeverMarried",
"Divorced/Separated",
"Widowed",
"Insured",
"Uninsured",
"Working",
"Unemployed",
"Total",
"Male",
"Female",
"White",
"Black",
"Asian",
"Hispanic",
"by_NumberofCounty",
"State_in_Total",
"Trump_sinceAug",
"Clinton_sinceAug",
"Trump_last50d",
"Clinton_last50d",
"Trump_last2w",
"Clinton_last2w",
"Electoral_Votes"])

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
fig = plt.figure(figsize=(25, 20)) 
ax = plt.subplot(111, polar=True)
plt.title("Attributes Polar Chart for Swing States Cluster (z-score)", size=30)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories2, size=13,  color ="#b30000")
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0], ["-2.0", "-1.5", "-1.0", "-0.5", "0", "0.5", "1.0", "1.5", "2.0"], color="grey", size=15)
plt.ylim(-2,2)
 
 
# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't do a loop, because plotting more than 3 groups makes the chart unreadable
 
# Ind1
values=polarrawt["cluster 1"].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=4, linestyle='solid', label="Cluster 1", color="#3DE5A6")
ax.fill(angles, values, "#3DE5A6", alpha=0)
 
# Ind2
values=polarrawt["cluster 2"].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', label="Cluster 2", color="#ff6600")
ax.fill(angles, values, "#ff6600", alpha=0)

# Ind3
values=polarrawt["cluster 3"].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', label="Cluster 3", color="#6699ff")
ax.fill(angles, values, "#6699ff", alpha=0)

# Ind4
values=polarrawt["cluster 4"].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', label="Cluster 4", color = "#d580ff")
ax.fill(angles, values, "#d580ff", alpha=0.2)

# Ind5
values=polarrawt["cluster 5"].values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=2, linestyle='solid', label="Cluster 5", color ="#ace600")
ax.fill(angles, values, "#ace600", alpha=0)


# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.05, 0.05))
