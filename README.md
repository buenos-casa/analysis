# analysis

MIR_Feature_Selection_Results is a directory containing Mutual Info Regression feature selection for each year. The yearly directories contain csv files of ranked features with their respective scores for each barrio. File names are in the format: year_bid_rankedFeatures.csv

barrio_feat_select.py generated the csv files in MIR_Feature_Selection Results using the final file csv with all merged data. [authored by Breanna]

price_pred.py generated the file for price gap prediction, or rather, ranked features by importance in driving the price gap on a yearly basis. Results are of the format: Year_rankedFeatures_gapPred.csv. note that 2018 was not created because there are was only one data point for this year after undergoing filtering. [authored by Breanna]


