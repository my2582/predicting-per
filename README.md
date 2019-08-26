# Forecasting P/E Ratios in Small-Cal Tech Sector using deep neural networks
- **Course name:** Applied Machine Learning for Financial Modeling in Spring 2019
- **Topic:** Can machine learning better forecast Price to Earnings ratios than current methods in small-cap technology sector?
- **Full report:** [here](https://github.com/my2582/predicting_per/raw/master/Can%20machine%20learning%20better%20forecast%20Price%20to%20Earnings%20ratios%20than%20current%20methods%20in%20small-cap%20technology%20sector_Minsu%20Yeom_Spring%202019.pdf)
- **Contributors**
  - Ling He (in alphabetical order)
  - Kitae Kum
  - Patrick Kwon
  - Dongoh Shin
  - Miao Wang
  - Minsu Yeom
  
# Summary of the full report
## Data sets
 - Fundamental values in financial statements such as balance sheets and income statements, and stock price momentum (3m, 6 and 9 months)
 - **Frequency:** Quarterly
 - **Source:** Wharton Research Data Services (WRDS), Center for Research in Security Prices (CRSP) and Compustat
 - **Period**
   - Originally, it's from January 1970 to the latest recorded date (May 2018)
   - We later decided to discard data from 1970 to 2005 since we want to avoid over-generalizing our model for periods that do not appear repeated in the future.
 - **Size:** Our original dataset contains 161,852 rows and 34 columns
 - **Missing values:** some companies may become inactive on the market due to bankrupt or delisting over some period. Therefore, as mentioned in previous section, we handled those discontinuous data by *filtering out* the rows where the companies are inactive indicated by the variable “active”.
 
 ### Features
 - **Dependent:** Market capitalization (Mkvaltq), Income Before Extraordinary Items (Ibq)
 - **Independent:** Sales/Turnover (Saleq), Cost of Goods Sold (Coqsq), Selling, General and Administrative Expenses (Xsgaq), Operating Income after Depreciation (Oiadpq), Cash and Short-Term Investment (Cheq), Receivables (Rectq), Inventories (invtq), Other Current Assets (Acoq), Property Plant and Equipment (Ppentq), Other Assets (Aoq), Debt in Current Liabilities (Dlcq), Account Payable/Creditors (Apq), Income Tax Payable (Txpq), Other Current Liabilities (Ltq), Stock price momentum for 3m, 6m and 9 months (Mom3m, Mom6m, Mom9m)
 
## Data standardization
- We normalize all features by *market capitalization* so that we have zero mean for the following two reasons:
  - efficient convergence speed
  - fair comparison
- We use the natural log to counter large values.
- Lastly, we applied Robust Scaler on all the preprocessed values.

## The model used
- We chose RNN as our model to generate the largest value-add because
  - both dependent and independent variables have typical characteristics of sequential and repetitive data, and
  - RNN automatically learns impactful feature representations during training as any deep learning model can approximate universal functions.
- **Cell type:** LSTM is chosed vs. GRU becausse LSTM is able to control the exposure of memory content.
- Developed using TensorFlow
- **Batches:**
  - The dataset was divided into many ‘batches’ depending on company ID and dates. Each batch contains approximately similar number of rows so that they will generally have the same dimension.
  - We splitted the batches into training and validation sets, where the training set contains 90% of the batches and the rest 10% are in the validation set.
- **Early stopping:** For each training epoch, the model will collect the training and validation MSE generated from the epoch. If the model starts to have very little marginal improvement on the validation MSE after certain number of epochs, the model stops and generates the model based on the minimum validation MSE.
- **A key difference in training:** Unlike traditional machine learning methodology where we split our data into training, validation, and test sets, we actually use *the same dataset for training and test*. Due to the sequential nature of our data, the past quarter’s feature values can be used to predict the present/current quarter’s P/E ratio, and the present/current quarter’s features values can be used to predict next quarter’s P/E ratio.

## Performances
