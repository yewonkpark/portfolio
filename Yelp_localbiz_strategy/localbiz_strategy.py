

### data preparation
# the dataset was obtained from http://yelp.com/dataset_challenge (2017)

#%% import packages
import pandas as pd
import numpy as np
import scipy.stats as stats
import gc
import matplotlib.pyplot as plt
import seaborn as sns

#%% converting business.json file into pandas dataframe
with open('business.json', 'r') as f: businessdata = f.readlines()
businessdf = pd.read_json('[' + ','.join(map(lambda x: x.rstrip(), businessdata)) + ']')
print(businessdf.columns)

#%% slicing 'attributes' columns which was dictionary
#slice the dataframe and remerge it to the original dataframe with new columns with name of each attributes.

attribute_slice = pd.DataFrame(businessdf['attributes'].tolist())
business_df = pd.concat([businessdf, attribute_slice],axis = 1)

del businessdata
del businessdf
gc.collect()

#%% parsing tip.json to extract only necessary info.(business_id and date)
import json
import itertools
import gc
tipid = []
with open('tip.json', 'r') as f: 
    for line in itertools.islice(f,0,None,1):
        tipdata = json.loads(line)
        value_id = tipdata["business_id"]
        tipid.append(value_id)


tip_df = pd.DataFrame(tipid,columns = ["business_id"])


del value_id
del tipid
del line
del tipdata
gc.collect()

#%% parsing checkin.json to extract only necessary info.


from pandas.io.json import json_normalize

with open('checkin.json') as f: cidata = f.readlines()
cidf = pd.read_json('[' + ','.join(map(lambda x: x.rstrip(), cidata)) + ']')        
citime = cidf.time
cinormal = json_normalize(citime)
checkin_df = pd.concat([cidf.business_id,cinormal],axis=1)
#replace Nan to 0 (no check-in occured)
checkin_df = checkin_df.fillna(0)
del cidf
del cidata
del citime
del cinormal
gc.collect()

# now, calculate the sum of the all check-in time plot and store the value in new column
checkin_df["checkin_count"] = checkin_df.iloc[:,1:].sum(axis=1)



#%% now we can merge refined business name, final cafe yes/no, and local/franchise label
label = pd.read_excel('cleansing_result.xlsx')
business_merged = pd.merge(business_df,label[['business_id','refined_finalname','refined_in/out','refined_franchise/local']], left_on='business_id', right_on='business_id', how='outer')
cafe_df = business_merged[business_merged['refined_in/out'] == 'in']
#ANALYSE ONLY FOR LOCAL !! (AVOID BIAS ESPECIALLY IN ATTIRIBUTES ANALYSES)
cafe_df = cafe_df[cafe_df['refined_franchise/local'] == 'Local']
cafe_df = cafe_df.reset_index(drop=True)

#%%add additional handlabel attribute for MCA
handlabel = pd.read_excel('handlabel_a.xlsx')
cafe_df = pd.merge(cafe_df, handlabel, on = "business_id", how = 'left')

#%%check the error
print(business_merged["business_id"].isnull().sum(axis=0))
print(label["business_id"].isnull().sum(axis=0))
##reserved for the case in need (initial observation)
## label the each business with 'y/n' if the business is cafe or not
#careful the list is not to be sorted since those two dataframe is iterated/merged on default index
#use lambda/map/any for sort out business any of the keyword

#searchcafe = np.array(['Cafes','Coffee & Tea','Coffee Roasteries','Coffeeshops']) #keywords

#cafeyesorno = []
#for everyitem in business_df["categories"] :
#    cafeyesorno.append(any(map(lambda x : x in searchcafe, everyitem))) 
    
#cafeyn = pd.DataFrame(cafeyesorno, columns = ["cafey/n"])
    
#business_df["cafey/n"] = cafeyn
    
#cafe_business = business_df[business_df["cafey/n"] == True]

##store the full list of the coffee business name - franchise/local label excluding repeated ones
fl_labellist = cafe_df[["refined_finalname","refined_franchise/local"]]
fl_labellist=fl_labellist.drop_duplicates(keep='first')

##merge / integrate check-in and tip data
#count the number of tip like countifs on key business id
tiptemp = np.array(tip_df)
tip_count = pd.DataFrame(stats.itemfreq(tiptemp), columns= ["business_id","tip_count"])
cafe_df = pd.merge(cafe_df, tip_count, on = "business_id", how = 'left')
cafe_df.tip_count = cafe_df.tip_count.fillna(0)

#match the checkin figrue we calculated above 
cafe_df = pd.merge(cafe_df, checkin_df[["business_id","checkin_count"]], on = "business_id", how = 'left')
cafe_df.checkin_count = cafe_df.checkin_count.fillna(0)

##making pivot table for individual brand/business : 
business_table = cafe_df.pivot_table(index = "refined_finalname" , values = ["business_id","review_count","checkin_count","tip_count","stars"], aggfunc= {"business_id" : 'count',"review_count" : ['sum','mean'],"checkin_count" : ['sum','mean'],"tip_count" : ['sum','mean'], "stars" : ['mean','std']})
business_table.columns = ["Number of Business","checkin_Average","checkin_Total","review_Average","review_Total","starrating_Average","starrating_std","tip_Average","tip_Total"]
business_table["refined_finalname"] = business_table.index.values
business_table["starrating_std"] = business_table["starrating_std"].fillna(0)
business_table.sort_values("Number of Business", inplace = True, ascending = False)
indv_df = pd.merge(business_table, fl_labellist, on = "refined_finalname", how = 'left')
column_titles = ['refined_finalname','refined_franchise/local','Number of Business', 'checkin_Average', 'checkin_Total','review_Average', 'review_Total', 'starrating_Average','starrating_std', 'tip_Average', 'tip_Total']
indv_df = indv_df.reindex(columns=column_titles)
# sorting the cafe list by the number of the store by name


compare_table = cafe_df.pivot_table(index="refined_franchise/local", values = ["business_id","review_count","checkin_count","tip_count","stars"], aggfunc= {"business_id" : 'count',"review_count" : ['sum','mean'],"checkin_count" : ['sum','mean'],"tip_count" : ['sum','mean'], "stars" : ['mean','std']})
compare_table.columns = ["Number of Business","checkin_Average","checkin_Total","review_Average","review_Total","starrating_Average","starrating_std","tip_Average","tip_Total"]



#%% PART 1 : starrating distribution (caution : NOT correlation coz these are agg not avg)

#plot the distribution : starrating vs review_count

plt.figure(1)
plt.suptitle('star rating vs number of review')
plt.xlabel('star rating')
ax = sns.distplot(cafe_df["stars"])


compare_table2 = cafe_df.pivot_table(index="stars", values = ["business_id","review_count","checkin_count","tip_count"], aggfunc= {"business_id" : 'count',"review_count": 'mean',"checkin_count": 'mean',"tip_count": 'mean'})
compare_table2["stars"] = compare_table2.index.values
#ax = sns.stripplot(x=compare_table2["stars"], y=compare_table2["review_count"] , size = 5, alpha = 0.5, hue=cafe_df["refined_franchise/local"], jitter=True, palette="Set2", dodge=True)

#%% PART2 : corr observation 1 : prob because star rating is 'categorical'
#here can be mitigated by using avg (grouped by brands)
#check the column name : cafe_df.columns.values

#by brand

#view as matrix
#should check some p-values
#a few outliers

#used the original indiv. raw data
e = pd.DataFrame(cafe_df["stars"])
f = pd.DataFrame(cafe_df["review_count"])
g = pd.DataFrame(cafe_df["checkin_count"])
h = pd.DataFrame(cafe_df["tip_count"])
spearman = pd.concat([e,f,g,h],axis=1).corr(method="spearman")
pearson = pd.concat([e,f,g,h],axis=1).corr(method="pearson")


sns.jointplot(x=cafe_df["stars"], y=cafe_df["review_count"],kind="reg",stat_func=stats.pearsonr,color="orange");
sns.jointplot(x=cafe_df["stars"], y=cafe_df["review_count"],kind="reg",stat_func=stats.spearmanr,color="orange");

sns.jointplot(x=cafe_df["stars"], y=cafe_df["tip_count"],kind="reg",stat_func=stats.pearsonr,color="purple");
sns.jointplot(x=cafe_df["stars"], y=cafe_df["tip_count"],kind="reg",stat_func=stats.spearmanr,color="purple");

sns.jointplot(x=cafe_df["stars"], y=cafe_df["checkin_count"],kind="reg",stat_func=stats.pearsonr);
sns.jointplot(x=cafe_df["stars"], y=cafe_df["checkin_count"],kind="reg",stat_func=stats.spearmanr);



sns.jointplot(x=cafe_df["review_count"], y=cafe_df["tip_count"],kind="reg",stat_func=stats.pearsonr, color="m");
sns.jointplot(x=cafe_df["review_count"], y=cafe_df["tip_count"],kind="reg",stat_func=stats.spearmanr, color="m");

sns.jointplot(x=cafe_df["review_count"], y=cafe_df["checkin_count"],color="g",kind="reg",stat_func=stats.pearsonr);
sns.jointplot(x=cafe_df["review_count"], y=cafe_df["checkin_count"],color="g",kind="reg",stat_func=stats.spearmanr);

sns.jointplot(x=cafe_df["tip_count"], y=cafe_df["checkin_count"], color = "r",kind="reg",stat_func=stats.pearsonr);
sns.jointplot(x=cafe_df["tip_count"], y=cafe_df["checkin_count"], color = "r",kind="reg",stat_func=stats.spearmanr);
#thought in this stage no need to cal corr for each local/franchise

#%% PART3 index development 1) normalisation

#Here normalisation is done (since review/tip/check-in range differs) to make new index

# option 1 : normalization to [0, 1] interval is done by subtracting the min value and diving by (maxVal - minVal).
# cons : outliers make other values to small
tonormalise = ["review_count","checkin_count","tip_count","stars"]
nornewcolumn = ["norm_review_count","norm_checkin_count","norm_tip_count","norm_stars"]
for i in range(0,len(tonormalise)) :
    minVal = cafe_df[tonormalise[i]].min()
    maxVal = cafe_df[tonormalise[i]].max()
    normCol = (cafe_df[tonormalise[i]] - minVal) / (maxVal - minVal)
    cafe_df[nornewcolumn[i]] = normCol

# option 2: standardization (z-score)
# pro: more stable cons: handle (-)value , and unbounded
tostandardise = ["review_count","checkin_count","tip_count","stars"]
standnewcolumn = ["zscore_review_count","zscore_checkin_count","zscore_tip_count","zscore_stars"]   

for i in range(0,len(tostandardise)) :
    zscore = stats.zscore(cafe_df[tostandardise[i]])
    cafe_df[standnewcolumn[i]] = zscore
    
decisionoption = cafe_df.describe()
    

    
#%% PART3 index development 2) index function
    


cafe_df["zscore_popsum"] = cafe_df["zscore_review_count"]+cafe_df["zscore_checkin_count"]+cafe_df["zscore_tip_count"]
cafe_df["index_levelofsuccess"] = cafe_df["zscore_popsum"]/2.873599429 + cafe_df["zscore_stars"]
popularity = pd.DataFrame(cafe_df["zscore_popsum"])
satisfaction = pd.DataFrame(cafe_df["zscore_stars"])

checkcheck=pd.concat([popularity,cafe_df["zscore_review_count"],cafe_df["zscore_checkin_count"],cafe_df["zscore_tip_count"]],axis=1).corr(method="pearson") 

ind = pd.DataFrame(cafe_df["index_levelofsuccess"])

indexdv = pd.concat([ind,popularity,satisfaction],axis=1).corr(method="pearson") 
#so far it 0.956  0.927. will adjust it!
#%% rank
cafe_df["index_levelofsuccess_rank"] = cafe_df["index_levelofsuccess"].rank(ascending=False)
cafe_df["index_levelofsuccess_rank_percentile"] = cafe_df["index_levelofsuccess_rank"] / len(cafe_df["index_levelofsuccess_rank"]) * 100

ranklabel = ["top0-10%","top10-20%","top20-30%","top30-40%","top40-50%","top50-60%","top60-70%","top70-80%","top80-90%","top90-100%"]
cafe_df["index_levelofsuccess_tag"] = pd.cut(cafe_df["index_levelofsuccess_rank_percentile"],10,labels=ranklabel)


#%% PART4 : attribute selection (MDA in R)



#attribute_list = attribute_slice.columns.values.tolist()
#reconstruct df as potential attribute (sliced/cateogrical) and index
feature_selection = ['index_levelofsuccess_tag','RestaurantsPriceRange2','BusinessAcceptsCreditCards','OutdoorSeating',
'RestaurantsTakeOut','WiFi','BikeParking','RestaurantsDelivery','RestaurantsGoodForGroups','RestaurantsReservations','GoodForKids',
'HasTV','NoiseLevel','DogsAllowed',
'AB_romantic','AB_intimate','AB_classy','AB_touristy','AB_trendy','AB_casual','AB_upscale',
'AB_hipster','AB_divey','BusinessParking_y','GoodForDessert',
'GoodForLatenight','GoodForLunch','GoodForDinner','GoodForBreakfast','GoodForBrunch']
cafe_df2 = cafe_df[feature_selection]

#handle the missing value
missing_fill = [0,1,'FALSE','FALSE','FALSE','no','FALSE','FALSE','FALSE','FALSE','FALSE','FALSE','average','FALSE',
                0,0,0,0,0,0,0,0,0,'FALSE',0,0,0,0,0,0]
#should justify why I handle the missing value as False(0)
#basically for 0/false, but price range (1), restaurantattrie to casual, noise level(average) which are highest frequency

cafe_df2.iloc[:,1] = cafe_df2.iloc[:,1].fillna(missing_fill[1])
cafe_df2.iloc[:,2] = cafe_df2.iloc[:,2].fillna(missing_fill[2])
cafe_df2.iloc[:,3] = cafe_df2.iloc[:,3].fillna(missing_fill[3])
cafe_df2.iloc[:,4] = cafe_df2.iloc[:,4].fillna(missing_fill[4])
cafe_df2.iloc[:,5] = cafe_df2.iloc[:,5].fillna(missing_fill[5])
cafe_df2.iloc[:,6] = cafe_df2.iloc[:,6].fillna(missing_fill[6])
cafe_df2.iloc[:,7] = cafe_df2.iloc[:,7].fillna(missing_fill[7])
cafe_df2.iloc[:,8] = cafe_df2.iloc[:,8].fillna(missing_fill[8])
cafe_df2.iloc[:,9] = cafe_df2.iloc[:,9].fillna(missing_fill[9])
cafe_df2.iloc[:,10] = cafe_df2.iloc[:,10].fillna(missing_fill[10])
cafe_df2.iloc[:,11] = cafe_df2.iloc[:,11].fillna(missing_fill[11])
cafe_df2.iloc[:,12] = cafe_df2.iloc[:,12].fillna(missing_fill[12])
cafe_df2.iloc[:,13] = cafe_df2.iloc[:,13].fillna(missing_fill[13])
cafe_df2.iloc[:,14] = cafe_df2.iloc[:,14].fillna(missing_fill[14])
cafe_df2.iloc[:,15] = cafe_df2.iloc[:,15].fillna(missing_fill[15])
cafe_df2.iloc[:,16] = cafe_df2.iloc[:,16].fillna(missing_fill[16])
cafe_df2.iloc[:,17] = cafe_df2.iloc[:,17].fillna(missing_fill[17])
cafe_df2.iloc[:,18] = cafe_df2.iloc[:,18].fillna(missing_fill[18])
cafe_df2.iloc[:,19] = cafe_df2.iloc[:,19].fillna(missing_fill[19])
cafe_df2.iloc[:,20] = cafe_df2.iloc[:,20].fillna(missing_fill[20])
cafe_df2.iloc[:,21] = cafe_df2.iloc[:,21].fillna(missing_fill[21])
cafe_df2.iloc[:,22] = cafe_df2.iloc[:,22].fillna(missing_fill[22])
cafe_df2.iloc[:,23] = cafe_df2.iloc[:,23].fillna(missing_fill[23])
cafe_df2.iloc[:,24] = cafe_df2.iloc[:,24].fillna(missing_fill[24])
cafe_df2.iloc[:,25] = cafe_df2.iloc[:,25].fillna(missing_fill[25])
cafe_df2.iloc[:,26] = cafe_df2.iloc[:,26].fillna(missing_fill[26])
cafe_df2.iloc[:,27] = cafe_df2.iloc[:,27].fillna(missing_fill[27])
cafe_df2.iloc[:,28] = cafe_df2.iloc[:,28].fillna(missing_fill[28])
cafe_df2.iloc[:,29] = cafe_df2.iloc[:,29].fillna(missing_fill[29])




#export to excel for R analyses
writer = pd.ExcelWriter('mcatotal6.xlsx')
cafe_df2.to_excel(writer,'Sheet1',index=False)
writer.save()


#%% PART4-2 : attribute selection (heatmap frequency... scoring)


#unify the number format and normalize 
cafe_df_heatmap = cafe_df2.replace([False,'FALSE',True,'True','no','free','paid','average','quiet','loud','very_loud'],[0,0,1,1,0,1,0.5,0.5,0.25,0.75,1])
cafe_df_heatmap["RestaurantsPriceRange2"] = cafe_df_heatmap["RestaurantsPriceRange2"].replace([1,2,3,4],[0.25,0.5,0.75,1])
#wi-fi, casual, noise-level is categorical
#price-range is not 0 or 1

heatmap_pivot = cafe_df_heatmap.pivot_table(index = "index_levelofsuccess_tag" , aggfunc= 'sum')
heatmap_divide = cafe_df2["index_levelofsuccess_tag"].value_counts()
heatmap_pivot_f = heatmap_pivot.div(heatmap_divide, axis=0)
    


h_cmap="YlOrRd"
f, ax = plt.subplots(figsize=(22,8))
sns.heatmap(heatmap_pivot_f,cmap =h_cmap ,linewidths=.2, annot=True, fmt=".0%", ax = ax)
plt.title("Business Feature HeatMap according to Success level (% within the group)")
plt.ylabel("Success level")
plt.xlabel("Business Features")
plt.show()


