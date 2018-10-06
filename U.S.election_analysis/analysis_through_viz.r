#load the package
library(readxl)
library(statebins)
library(MASS)
library(corrplot)
library(ggcorrplot)
library(ggplot2)
library(FactoMineR)
library(factoextra)
library(dplyr)
library(GGally)
library(reshape2)
library(RColorBrewer)
library(scales)
library(ggrepel)


#load the data
rawdata <- read_excel("electionswing2.xlsx", sheet="states")
rawdataagg <- read_excel("electionswing2.xlsx", sheet="swingyn")
View(rawdata)


#shape check
dim(rawdata) 
#data type check
sapply(rawdata,class)

#change the column name
names(rawdata)[names(rawdata) == "X__1"] <- "Swing_States"

#practice corrplot package
#slice the data with only numerical column
corrdata <- rawdata[,2:53]
rownames(corrdata) <- rawdata$Swing_States

#generate correlation matrix
electionresult <- corrdata[,c("Vote_Clinton","Vote_Trump","Vote_Gap")]
variables = corrdata[,!(names(corrdata) %in% c("Vote_Clinton","Vote_Trump","Vote_Gap"))]
corr <- cor(x=electionresult, y=variables,  method="pearson")

#regarding p-value
#corr_pvalue <- cor_pmat(corrdata, method="pearson")
#corr_pvalue <- corr_pvalue[c("Vote_Clinton","Vote_Trump","Vote_Gap"),]
#corr_pvalue <- corr_pvalue[,colnames(corr_pvalue) != "Vote_Clinton"]
#corr_pvalue <- corr_pvalue[,colnames(corr_pvalue) != "Vote_Trump"]
#corr_pvalue <- corr_pvalue[,colnames(corr_pvalue) != "Vote_Gap"]



#melt correlation matrix using reshape package
melted_corrp <-melt(corr)


fig1 <- ggplot(data = melted_corrp, aes(x=Var1, y=Var2, fill=value)) + 
geom_tile(color=melted_corrp$value) +
labs(title = "Correlation btw election result and attributes") +
geom_text(aes(label=round(melted_corrp$value,2)), color="black", size=2.8) +
scale_fill_gradient2(low = "red", high = muted("blue"), mid = "white", midpoint = 0, limit = c(-1,1), 
space = "Lab", name="Pearson Correlation Analyses", guide="colourbar") +
theme_minimal() +
coord_fixed(ratio=0.2) +
theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  plot.title = element_text(size = rel(1)),
  panel.grid.major = element_blank(),
  panel.border = element_blank(),
  panel.background = element_blank(),
  axis.ticks = element_blank(),
  legend.position = "top",
  legend.title =element_text(size = 8.5)) +
guides(fill = guide_colorbar(direction = "horizontal", barwidth = 13, barheight=1, title.position = "top" ))
fig1

#additional wrangling
melted_corrp$tag <- ifelse(melted_corrp$value >= 0.4, "strong+", ifelse(melted_corrp$value <= -0.4, "strong-","-"))  
melted_corrp$tag2 <- ifelse(melted_corrp$value >= 0.4, "strong+",ifelse(melted_corrp$value >=0.3, "moderate+", ifelse(melted_corrp$value <= -0.4, "strong-", ifelse(melted_corrp$value <= -0.3, "moderate-","-"))))



#scatter plot for selected attributes
#differ the size by the number of electoral vote
fig3 <-ggplot(data=rawdata, mapping=aes(x=Turn_Out_Rate_Asian, y=Vote_Clinton)) +
geom_point(mapping = aes(color=Swing_States, size=Electoral_Votes_Number))+ 
stat_smooth(method=lm, se=FALSE, size=1, linetype = "longdash", color="black")+
theme_minimal() +
geom_label_repel(aes(label = Swing_States), size=2.7, color='#566573')+
labs(title = "Correlation between Asian Turn-out rate and Vote% Clinton (r=0.69)") +
theme(axis.title.y = element_text(size=9, face= "bold"),
axis.title.x = element_text(size =9, face="bold"),
axis.text = element_text(size=8),legend.title =element_text(size = 10))
fig3

fig4 <-ggplot(data=rawdata, mapping=aes(x=Educational_Attainment_Bachelors_Degree, y=Vote_Gap)) +
geom_point(mapping = aes(color=Swing_States, size=Electoral_Votes_Number))+ 
stat_smooth(method=lm, se=FALSE, size=1, linetype = "longdash", color="black")+
theme_minimal() +
geom_label_repel(aes(label = Swing_States), size=2.7, color='#566573')+
labs(title = "Correlation between Bachelor's Degree attainment and Vote Gap (r=0.48)") +
theme(axis.title.y = element_text(size=9, face= "bold"),
axis.title.x = element_text(size =9, face="bold"),
axis.text = element_text(size=8),legend.title =element_text(size = 10))
fig4


fig5 <-ggplot(data=rawdata, mapping=aes(x=Age_60to80over, y=Vote_Trump)) +
geom_point(mapping = aes(color=Swing_States, size=Electoral_Votes_Number))+ 
stat_smooth(method=lm, se=FALSE, size=1, linetype = "longdash", color="black")+
theme_minimal() +
geom_label_repel(aes(label = Swing_States), size=2.7, color='#566573')+
labs(title = "Correlation between Age 60 + and Vote% Trump (r=0.55)") +
theme(axis.title.y = element_text(size=9, face= "bold"),
axis.title.x = element_text(size =9, face="bold"),
axis.text = element_text(size=8),legend.title =element_text(size = 10))
fig5

#the second heatmap considering impact


#agg2 <- rawdataagg[3,]
#agg2 <- t(agg2)
#colnames(agg2) <- c("figure")
#agg2 <- as.data.frame(agg2)
#agg2$Var2 <- rownames(agg2)
#agg2 <- agg2[c("Var2","figure")]
agg3 <- read.csv('agg2.csv')
agg3 <- select(agg3, -X)

melted_corr_impact <- melted_corrp[,c("Var1","Var2","value")]
melted_corr_impact <- merge(melted_corr_impact, agg3, by=c("Var2"))
melted_corr_impact$figure <- as.numeric(levels(melted_corr_impact$figure))[melted_corr_impact$figure]
melted_corr_impact$figure2 <- ifelse(melted_corr_impact$figure >= 1,1,melted_corr_impact$figure )
melted_corr_impact$figurefinal = melted_corr_impact$value * melted_corr_impact$figure2



fig2 <- ggplot(data = melted_corr_impact, aes(x=Var1, y=Var2, fill=figurefinal)) + 
geom_tile(color=melted_corr_impact$figurefinal) +
labs(title = "Impact of attributes to election result") +
geom_text(aes(label=round(melted_corr_impact$figurefinal,2)), color="black", size=2.8) +
scale_fill_gradient2(low = "yellow", high = muted("green"), mid = "white", midpoint = 0, limit = c(-1,1), 
space = "Lab", name="Impact of Attributes", guide="colourbar") +
theme_minimal() +
coord_fixed(ratio=0.2) +
theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  plot.title = element_text(size = rel(1)),
  panel.grid.major = element_blank(),
  panel.border = element_blank(),
  panel.background = element_blank(),
  axis.ticks = element_blank(),
  legend.position = "top",
  legend.title =element_text(size = 8.5)) +
guides(fill = guide_colorbar(direction = "horizontal", barwidth = 13, barheight=1, title.position = "top" ))
fig2


#additional table : write.csv()

#initial approach_pca

library(FactoMineR)
library(factoextra)
library(rgl)
library(plot3D)
library(car)
library(plotly)

#PCA용 로데이터 다시 생성

pcaraw <- rawdata[,1:49]
rownames(pcaraw) <- pcaraw$Swing_States
pcaraw <- select(pcaraw, -Swing_States)


res.pca <- PCA(pcaraw, scale.unit = TRUE, ncp = 3, graph = TRUE)
print(res.pca)

#check the eigenvalue table for each level of dimension
eig.val <- get_eigenvalue(res.pca)
eig.val

#The scree plot can be produced using the function fviz_eig() or fviz_screeplot() [factoextra package]
fviz_eig(res.pca, addlabels = TRUE, ylim=c(0,45), barfill = "#ff704d", barcolor="#ff704d", linecolor = "#800000", title = "Scree plot : % of explained variance by each PCA dimension")

#extract pca result for variables
var <- get_pca_var(res.pca)
var

#how well "each attribute" is explained by dimension 1,2,3
#how much variations are captured by each dimension !! (important)
variationcaptured <- var$cos2
print(variationcaptured)
corrplot(var$cos2, is.corr=FALSE) 
fviz_cos2(res.pca, choice="var", axes = 1:3, fill ="#70dbdb", color ="#70dbdb", ggtheme = theme_minimal(), 
xtickslab.rt = 90, title = "Bar chart : sum of cos2 (Dim1 + Dim2 + Dim3) for each attribute", labels = TRUE)

# Contributions of variables to PC1 - PC3
fviz_contrib(res.pca, choice = "var", axes = 1:3, top = 20)

#now move on to individual level
indvcorr <- res.pca$ind$coord
print(indvcorr)
indvcorr <- as.data.frame(indvcorr)

#pca plotting

#if plot_ly failed, alternatives.
x <- plot.x <- indvcorr$Dim.1
y <- plot.y <- indvcorr$Dim.2
z <- plot.z <- indvcorr$Dim.3 
scatter3d(x,y,z, surface=FALSE, grid=FALSE, xlab = "PC1 (37.4%)", ylab = "PC2 (16.9%)",
        zlab = "PC3 (13.6%)", label=rownames(indvcorr), id.n=nrow(indvcorr), sphere.size = 1)

# plotly
fig6 <- plot_ly(indvcorr, type= "scatter3d", x = ~Dim.1, y = ~Dim.2, z = ~Dim.3,mode ='markers+text', text = rownames(indvcorr)) %>% 
  add_annotations(x = indvcorr$Dim.1, y = indvcorr$Dim.2, z = indvcorr$Dim.3 , text = rownames(indvcorr), xref="x", yref="y", zref="z") %>%
  layout(scene = list(xaxis = list(title = 'PC1 (35.7%)'),
                     yaxis = list(title = 'PC2 (17.0%)'),
                     zaxis = list(title = 'PC3 (14.2%)')))
fig6
                     
#참고 : how to drop columns in R
indvcorr <- select(indvcorr, -states)

#save for 2d plotting later
#fviz_pca_ind(res.pca$ind, geom.ind=c("point","text")) +
#labs(title = "Principal Component Analysis",  x = "PC1", y = "PC2")

#k-mean clustering 
#clusterdf <- scale(indvcorr)

#find the optimal k
library(NbClust)
nbtest <- NbClust(indvcorr, distance = "euclidean", method = "kmeans", min.nc=2, max.nc=6)
fviz_nbclust(nbtest, barfill = "#ff3385", barcolor="#ff3385")

#do clustering with k=6
set.seed(200)
km.res <- kmeans(indvcorr, 6, nstart=50)
print(km.res)

#do clustering with k=5
set.seed(200)
km.res2 <- kmeans(indvcorr, 5, nstart=50)
print(km.res2)

#do clustering with k=2
set.seed(200)
km.res3 <- kmeans(indvcorr, 2, nstart=50)
print(km.res3)

#merge the cluster result to the original data and do some cleansing
clusterresultk5<- as.data.frame(km.res2$cluster)
clusterresultk5$Swing_States <- rownames(clusterresultk5)
rawwithcluster <- merge(rawdata, clusterresultk5, by=c("Swing_States"))

pcawithcluster <- indvcorr 
pcawithcluster$Swing_States<- rownames(pcawithcluster)

pcawithcluster <- merge(pcawithcluster, clusterresultk5, by=c("Swing_States"))
names(pcawithcluster)[names(pcawithcluster) == "km.res2$cluster"] <- "cluster_result"
pcawithcluster$cluster <- paste("cluster", pcawithcluster$cluster_result)



pcawithcluster <- merge(pcawithcluster,rawdata[,c("Electoral_Votes_Number","Swing_States")], by=c("Swing_States"))

#plot again with different colour for each cluster


fig7 <- plot_ly(pcawithcluster, type= "scatter3d", x = ~Dim.1, y = ~Dim.2, z = ~Dim.3,mode ='markers+text', color=~cluster, text = ~Swing_States,
marker = list(symbol = 'circle', sizemode = 'diameter'), sizes = 15, textfont = list(size=15)) %>% 
  add_annotations(x = pcawithcluster$Dim.1, y = pcawithcluster$Dim.2, z = pcawithcluster$Dim.3 , text = pcawithcluster$Swing_States, xref="x", yref="y", zref="z") %>%
  layout(title = 'Clustering result for US Swing States', scene = list(xaxis = list(title = 'PC1 (35.7%)'),
                     yaxis = list(title = 'PC2 (17.0%)'),
                     zaxis = list(title = 'PC3 (14.2%)')))
fig7

#plot clustering result into statebins
library(statebins)
clustermap <- read_excel("clustermap.xlsx")

fig8 <- statebins(state_data = clustermap, state_col = "st", value_col = "ing", text_color = "white", breaks=6, 
labels = 0:5, font_size=4, brewer_pal="Paired", plot_title = "US Swing States Cluster Map (k=5)", title_position = "top", legend_position="none")
fig8




#raw for polar radar chart (not weighted ver.)

names(rawwithcluster)[names(rawwithcluster) == "km.res2$cluster"] <- "cluster_result"
rawwithcluster$cluster <- paste("cluster", rawwithcluster$cluster_result)
rawforpolar_test <- aggregate(rawwithcluster[,2:53], by=list(rawwithcluster$cluster), FUN=mean, narm=TRUE) 
rawforpolar_test <- select(rawforpolar_test, -c(Vote_Clinton,Vote_Trump,Vote_Gap))


#z-score standardisation 

rawforpolarz <- as.data.frame(scale(rawforpolar_test[,2:50]))
rownames(rawforpolarz) <- c("cluster 1", "cluster 2", "cluster 3","cluster 4","cluster 5")

write.csv(rawforpolarz, 'polarpolar.csv')