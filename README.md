# analysis

The analysis section is based using the merged data from three different sources including Properati open dataset, city of Buenos Aires dataset, and census. The merged data puts a wide range of features suitable for conducting unique analysis at individual home level. The analysis has three sections:
1) Feature selection
2) Mislabeled homes: homes that have been mislabeled by sellers
3) Price gap: the gap between the advertised price and the final price

Feature selection:
MIR_Feature_Selection_Results is a directory containing Mutual Info Regression feature selection for each year. The yearly directories contain csv files of ranked features with their respective scores for each barrio. File names are in the format: year_bid_rankedFeatures.csv
barrio_feat_select.py generated the csv files in MIR_Feature_Selection Results using the final file csv with all merged data. [authored by Breanna]
price_pred.py generated the file for price gap prediction, or rather, ranked features by importance in driving the price gap on a yearly basis. Results are of the format: Year_rankedFeatures_gapPred.csv. 
Files of the format: year_gap_feat_influence are based from the price gap prediction feature selection results. Rather than scores, features are assigned a percentage of influence based on the scores. 

Mislabeled homes:
This section includes correlation analysis on mislabeled homes at the barrio level. The purpose of this analysis is to show what features cause higher mislabeling rate at the barrio level. Sellers might mislabel their barrio name by purpose or mistake, but our analysis reveals interesting findings about sellers' behaviour. In this process, we had some chalanges, for example, people entered same barrio names with different proununciation such as "NUÃ‘EZ" or "NUNEZ", or diffrent style like "VILLA GRAL. MITRE" and "Villa General Mitre". Fnally we compute the Mislabaling (Cheat Rates) percentage over different Barrios, and we have a study on the correlation between this rates with average of other features over barrios. the result is showing in pic (slide 1.jpg):
as you can see the correlation of Education and uninhabited percent and %of highest price were more significant equal to -0.33 (pic slide 2, 3.jpg), it means in general when education and "price pre m2" and rate of "uninhabited" are higher we expect that rate of mislabeling decrease. 
"%of highest price" for each barrio means (average price per m2 for that barrio / highest average price per m2)*100 
we also observe that features of "computer Percent" and "Education Percent" has high positive correlation with each other with value 0.85, it means when a barrio has a higher education level we can expect higher rate of computer usage which is make sense. 
As you can see in pic (PriceEducation_Analys_pics.jpg) we also see significant positive correlation of features "computer Percent" and "Education Percent" on "%of highest price", Education correlation value on property price is 0.76 and correlation value for computer usage on property price is 0.47. 
When we compare "%of highest price" with the "%of higest distanceToHealth" correlation value is -0.61 which means that when average distance to helth centers are less, the price per m2 would be higher. Inverse when we compare "%of highest price" with the "%of higest distanceTotransportation" correlation value is 0.55 which means that when average distance to transportations is higher, the price per m2 would be higher, so there is possitive correlation between these two which is interesting.



Price gap:
This section covers some analysis on identifying how price gap changes across different barrios and different years. This is a unique dataset that has been created by matching longitude/latitude of homes in Properati and the city of Buenos Aires datasets. The price gap analysis enables us to see how the price gap in some barrios is higher than the others.
