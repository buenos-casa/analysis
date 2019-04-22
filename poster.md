### Description of the system

In general, we created a web application that visualized our analysis of Buenos Aires' real estate market using Properati's rent and sales dataset and open data from Autonomous City of Buenos Aires' (CABA) portal. We also:
- created heatmaps that characterized Buenos Aires' neighborhoods based on average rental, property sale and property value prices. 
- created charts that showed the characteristics of each neighborhood and the ranking of relevant features used in price prediction 
- created a map showing households that mislabelled the location of their properties within Properati; a Argentian version of the "Buckhead Effect" 


### Our goal

Characterize the different neighborhoods of Buenos Aires in order to visualize and analyze the housing inequality prevalent in the city. 

### Data Cleanup and Analysis
We characterized the neighborhoods or _"barrios"_ using the following data:
- Property rental and sale listings on Properati between January 2015 and March 2018
- Argentina's 2010 Census
- Buenos Aires' recorded property sales 
- Buenos Aires' list of landmarks, social care centers, sport amenities and health providers. 
  
We used Python scripts and QGIS to merge perform geospatial merges between the different datasets. 

We also did:
- Linear Regression for Price Prediction using the advertised price on Properati
- Used Mutual Information Regression to determine the information gain for the different features from the price prediction
- Identified properties that misclassified their location on Properati per neighborhood
- Analyzed the gaps between advertised sale price on Properati and the city's recorded purchase price and performed additional feature selections on properties that had these gaps.


### Conclusions
#### Feature selection:
##### Price gap analysis
Actual price of each property is the most driving factor of the price gap followed by public wifi access and cellular percent. 

##### Advertised price
Public transportation is the most important feature across time in all regions of Buenos Aires and is most observed in Eastern barrios. In addition, Southern barrios produce notable MIR scores with public wifi and health. Interestingly, Western Buenos Aires produce higher scores pertaining to education percent despite most schools and univsierites being positioned in the East. Nothern barrios tend to place emphasis on nearby education percent, and health and sports facilities while resulting in the smallest vairance amongst feature importance.     


### Mislabelled Homes:
We found that barrios that had lower percentiles of Education and Computer literacy had a much higher likelihood of mislabelling their property location. Also, we noted that the majority of the mislabelled households per barrio were found on the edges of the barrios.




