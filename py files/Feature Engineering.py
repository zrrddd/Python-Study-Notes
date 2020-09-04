# 网址这里：https://www.kaggle.com/pavansanagapati/comprehensive-feature-engineering-tutorial

# Feature Engineering can lead to the following benefits:
# 1. Enable us to achieve good model performances using simpler Machine Learning models.
# 2. Using simpler Machine Learning models, increases the transparency of our model, therefore making easier for us to understand how is making its predictions.
# 3. Reduced need to use Ensemble Learning techniques.
# 4. Reduced need to perform Hyperparameters Optimization.

################################## 1. Missing Data Imputation ##################################
# 1. Dropping observations that have missing values
# 2. Imputing the missing values based on other observations

######## Type of missingness ########
# 1. MCAR (Missing Completely At Random)
# 2. MAR (Missing At Random)
# 3. MNAR (Missing not at Random)

######## Missing numeric data ########
# - Use flaging and filling:
# - Flag the observation with an indicator variable of missingness
# - Fill the original missing value with 0 just to meet the technical requirement of no missing data
# - IMPUTE MEAN: Suitable for continuous data without outliers
# - IMPUTE MEDIAN: Suitable for continuous data with outliers

######## Missing categorical data ########
# - The best way to handle missing data is to simply label them as 'Missing' (这个有点意思)
# - IMPUTE MODE

######## Complete Cast Analyasis (CCA) ########
# - 很简单，就是只用那些没有missingness的observation
# - CCA works well when the data are MCAR
# Advantages: 
# - Easy to implement
# - 保持distribution不变（因为MCAR）
# Disadvantages:
# - 可能large fraction data被删掉了，损失了很多信息
# - 如果不是MCAT就糟糕了，因为你的distribution会变化

## 注意!In many real life datasets, the amount of missing data is never small, and therefore CCA is typically never an option.

######## Random Sample Imputation ########
# - 就是random sample一些observation然后impute到NA里，这样我们perserve mean and sd
# - Assume that data are MCAR
# - Variance is perserved: 因为出现的多的一些data会被选中的几率更大，比如说mean 还有 mode, median
# - 不是经常用诶，compared to impute mean/median/mode，因为randomness所以不经常用

######## Replacement by Arbitrary Value ########
# - 首先要明白如果impute mean/median/mode，说明你认为他不是MNAR，因为impute以后你让NA have data similar to observed data
# - 如果是MNAR,那你没有出现的data很可能有原来data所没有的value

# You can:
# - Add an additional binary variable indicating the missingness
# - Replace the NA by a value at a far end of the distribution
# - 懂了，就是比如原来的distribution是0-100的normal，你replace一个100，那大家都知道这是missing value了， 因为100不常见
# - Used in real life, 比如在金融行业，如果信用卡记录有缺失，那我肯定假设是MNAR，会impute一个不常见的值（就是假设这个人不是好人？


################################## 2. Categorical Encoding ##################################

######## Data Types ########
# 1. Nominal Data
  # - Categorical, 然后没有order
# 2. Ordinal Data
  # - Categorical, 但是有order
# 3. Interval Data
# 4. Ratio Data

######## 1. Classic Encoders ########
# - One Hot
# - Binary: onvert each integer to binary digits. Each binary digit gets one column. Some info loss but fewer dimensions. Ordinal.
# - Ordinal 
# - BaseN
# - Hashing: Like OneHot but fewer dimensions, some info loss due to collisions. 
# - Sum: Just like OneHot except one value is held constant and encoded as -1 across all columns.

######## 2. Contrast Encoders ########
# - Helmert (reverse)
# - Backward Difference
# - Polynomial 

######## 3. Bayesian Encoders ########
# - Target
# - LeaveOneOut
# - WeightOfEvidence
# - James-Stein
# - M-estimator



######## 1. Classic Encoders ########

######## OneHot Encoder ########
import pandas as 
pd.get_dummies(df, drop_first = True, dummy_na = True) # 这样missing也会算作一个category

# 我去如果是feature selection就不要drop第一个！因为你有可能把很重要的feature给drop了
# Advantages:
# 1. straightforward
# 2. makes no assumption
# 3. keeps all the information of categorical variable
# Disadvantages:
# 1. Does not add any information that may make the variable more predictive
# 2. 如果category很多的话会增加很多column

######## Count and Frequency Encoder ########
# 就是not replace with 1 but with count/frequency
# Simple, does not expand the feature space（在原来的column上直接修改）
# 不好的就是如果两个count一样，那就搞不清谁是谁了，就会lose infomation
frequency_map = X_train[col].value_counts().to_dict()
X_train.col = X_train.col.map(frequency_map) #直接map就好了，或者replace？

# 注意如果有的label是test里有的但是train没有，那你就会有missingness
# 有一种方法是把几个比较频繁出现的encode一下，剩下的category全部归为“Rare"

######## Binary Encoder ########
# Hybrid of One-hot and hashing encoders
# Create fewer features than one-hot, while preserving some uniqueness of values in the column

# 1. Encoded by Ordinal Encoder first
# 2. Covert integers to binary code, for example 5 becomes 101 and 10 becomes 1010
# 3. 然后有几位就几个column，比如你有7个category，那就是3个column，因为(2**3 = 8!)

import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder

binary_encoder = cd.BinaryEncoder(cols = ['col'])
binary_encoder.fit_transform(X, y)

# Binary Encoder很适合如果你有很多种类，比如50个US state
# 但如果种类很少的话，就onehot好了


######## Ordinal Encoder ######## 
ordinal_encoder = ce.OrdinalEncoder(cols = ['col'])
ordinal_encoder.fit_transform(X, y['col'])


######## BaseN Encoder ######## 
# When BaseN = 1, one hot encoding
# When BaseN = 2, binary encoding
BaseN_encoder = ce.BaseNEncoder(cols = ['col'])
BaseN_encoder.fit_transform(X, y)

# default is BaseN = 2

######## Hashing Encoder ######## 
# Hashing Encoder implements the hashing trick. It is similar to one-hot encoding but with fewer new dimensions and some info loss due to collisions
Hashing_encoder = ce.HashingEncoder(cols = ['col'])
Hashing_encoder.fit_transform(X, y)


######## Sum Encoder ######## 
Sum_encoder = ce.SumEncoder(cols = ['col'])
Sum_encoder.fit_transform(X, y)



######## 2. Contrast Encoders ########

######## Backward Encoder ######## 
# the mean of the dependent variable for a level is compared with the mean of the dependent variable for the prior level
ce_backward = ce.BackwardDifferenceEncoder(cols = ['col'])
ce_backward.fit_transform(X, y)


######## Helmert (reverse) Encoder ######## 
# - The opposite of Backward Encoder 
# - instead of comparing each level of categorical variable to the mean of the previous level, it is compared to the mean of the subsequent levels.
ce_helmert = ce.HelmertEncoder(cols = ['col'])
ce_helmert.fit_transform(X, y)

######## Polynomial Encoder ######## 
ce_poly = ce.PolynomialEncoder(cols = ['col'])
ce_poly.fit_transform(X, y)


######## 3. Bayesian Encoders ########
# - The Bayesian encoders use information from the dependent variable in their encodings
# - They output one column and can work well with high cardinality data.


######## Target/Mean Encoder ######## 
# - 一个category encode的value based on the mean of target
ce_target = ce.TargetEncoder(cols = ['col'], smoothing = 10) #这个smoothing是做什么的？
ce_target.fit_transform(X, y)




################################## 3. Gaussian Encoding ##################################










