## HackerEarth - Machine Learning 3

**Problem Statement:**
A leading affiliate network company from Europe wants to leverage machine learning to improve (optimise) their conversion rates and eventually their topline. Their network is spread across multiple countries in europe such as Portugal, Germany, France, Austria, Switzerland etc.

Affiliate network is a form of online marketing channel where an intermediary promotes products / services and earns commission based on conversions (click or sign up). The benefit companies sees in using such affiliate channels is that, they are able to reach to audience which doesn’t exist in their marketing reach.

The company wants to improve their CPC (cost per click) performance. A future insight about an ad performance will give them enough headstart to make changes (if necessary) in their upcoming CPC campaigns.

In this challenge, you have to predict the probability whether an ad will get clicked or not. 


**Program Flow:**
- Visualized data using Tableau: visualizations_train.twb
- Pre-Processing 
    - Pre_Processing1.1.py : Predicted missing values for browser and devid
	- Pre_Processing2.1.py : Built on top of O/P of Pre_Processing1.1. Created more features using aggregation
- Building Models
    - Model_XGBoost2.py : Ran a XGBoost classifier over the preprocessed file
    - Model_CatBoost.py : Ran a CatBoost classifier over the preprocessed file
    - Parameter_Tuning.py : Used GridSearch to get best parameters for XGBoost


**Files not used:**
- Pre_Processing1.py : Failed to correctly predict missing values
- SiteID_Pred.py, Pre_Processing_siteid_top100.py : Performed poorly owing to number of categories.
- Model_XGBoost.py : Version discarded after paramter tuning and changing files
- Model_Keras_NN.py : Did not get time to work on it
- Ensembling: Tried a simple averaging model as well as a combined model of XGBoost and CatBoost. Neither gave great results.


**Overall Rank:** 9

    
