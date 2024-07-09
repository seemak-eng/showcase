**Title: House Price of King County**

**Executive Summary**

*Objective:*

The goal of this data project is to develop a predictive model for housing prices using a dataset containing various features
related to home sales. Our primary focus is to build a regression model to predict house price effectively.

*Data Overview:*

The dataset encompasses diverse variables providing insights into home characteristics, including the number of bedrooms
and bathrooms, square footage, location, condition, and grade, among others. The target variable, “price,” represents the sale
price of the homes for the Washington state King County between beginning of May 2014 to end of May 2015.

*Key Predictors:*

Essential metrics for assessing the size and functionality of a home include number of bedrooms, number of bathrooms,
footage of livable area, and lot size. Property’s features and desirability are presented by waterfront and a good view of
surroundings. Condition and grade are metrics that reflect the overall condition and quality of the home. number of years
since built and number of years since renovated are factors that indicate the age and renovation history of the property.
Location-based features are average lot size of the nearest 15 houses and average footage of the nearest 15 houses.

*Approaches:*

We create and test different regression models on both train and test data sets. Neural Networks gives best results among all
models, followed by LASSO and multiple linear regression. Diagnostics and remedial measures are conducted thoroughly
for every models we try. Most of the R packages and knowledge we gain from the present course CSCI-E106. Additional
reference websites are GeeksforGeeks (GeeksforGeeks 2008), Wikipedia etc.

**Conclusion:**

This data project aims to build an accurate model for predicting housing prices. The results obtained will not only contribute
to a better understanding of the factors influencing home prices but also offer practical guidance for individuals involved in
the real estate market.

**I. Introduction**

*A brief literature review*

Before we dive into the data analysis of house sales prices in King County, Washington, we look into a couple of review
papers. The first is a survey on methods and input data types (Geerts, M.; vanden Broucke, S.; De Weerdt, J. 2023) that
summarizes ninety-three publications on modeling methods and types of input data. The methods range from basic techniques
such as multiple regression analysis or spatial econometric models, to time series, fuzzy logic, decision trees, and also novel
methods like artificial neural networks and deep learning. Input data categories vary as well. Data of standard features
includes structural, temporal, socioeconomic, and environment. Then, there is standard and advanced level spatial data.
Some publications use graphs and other unstructured data such as image and text.
We also review an assessment (Chwiałkowski, C.; Zydron, A.; Kayzer, D. 2022) on factors that affect dwelling price which
summarizes the categories of attributes that highly related with house price based on literature review (MIRANDA CRACE
2023). It draws our attention to not only the physical characteristics, but also the characteristics of neighborhood, environment,
and economic conditions.

*Focus of present analysis*

Although there are many factors associated with a house’s price and there could be countless methods to be applied to
house valuation and price prediction, the present report focuses on building linear regression models to predict sales prices of
real estate properties using the house sales data collected in King County, Washington, USA between the years 2014 and
2015. Since the data we use for modeling is limited to certain physical characteristics and neighborhood comparison, the
interpretation of estimates, statistics and prediction is bound to these physical characteristics and neighborhood data.
We divide the whole data set into a random training data set accounting for 70% of the raw data set and the remaining is
used as a test data set to test the performance of the models we build here. We start with exploratory data analysis (EDA)
to summarize the uni-variate distribution and bi-variate correlation. EDA helps us seek the potential remedial measures.
Following EDA, we conduct multiple linear regression on selected predictors to build a full model which is the starting point
of our modeling process. With the residuals and fitted values available from the full model, we are able to carry out the
model diagnostics, then the remedial measures, such as transformation on response and predictors, detecting influential data
points and checking heteroscedasticity. Following diagnostics, we conduct variable selections, such as best subset and stepwise
selection to identify significant predictors. We also try other methods such as the weighted least square (WLS), ridge, lasso,
elasticNet, iterated re-weighted least square regression, etc. At last, we transform the house price from continuous type to
a binary variable so that we could build a logistic regression model to work around the normality assumptions of linear
(Gaussian) regression.
