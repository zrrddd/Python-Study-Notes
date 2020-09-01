#网址这里https://www.kaggle.com/saisatish09/a-beginner-s-guide-to-feature-selection-methods
################################# A. Filter Methods ############################################
#### 1. Constant Features ####
# 非常简单，对于所有data都是一样的column，直接drop掉就好
df.std() == 0

#### 2. Quasi - Constant Features ####
# 大部分data都是same value
# 所以不能直接看std是不是等于0，要设置一个threshold
# Threshold: 比如是99%，那就是data要99%都是一样的才会被drop掉

#### 3. Duplicated Features ####
# 这个应该就是说如果有两个column是重复了，那drop掉就好

#### 4. Correlation ####
# Correlation就是代表的how 2 variables are linearly related
# There are several different correlation techniques
# 今天要介绍的是 —— Pearson correlation, which works best with 线性关系
# 等下这个不就是课上学的吗？？？
df.corr()

# 如果两个variables highly correlated， 比如 > 0.9
# 那么你把两个都加进去没什么用，加一个就可以了

#### 5. Mutual Information (MI) ####
# MI is a measure of the mutual dependence between 2 variables
# 就是 it quantifies the "amount of information" obtained about one variable if you observe another variable
# 1. MI measures the mutual depency of two variables
# 2. MI determines how similar p(X, Y) is to product of p(X) and p(Y)
# 3. If X and Y are independent, then MI is zero

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile

# 5.1 MI for classification 
# get the MI info between each feature and the target
mutual_info_classif(X_train, y_train) # 注意这个是专门classification

# Use the SelectKBest method to select the TOP K variables 
selector = SelectKBest(mutual_info_classif, k = 10).fit(X_train, y_train)
X_train.columns[selector.get_support()]

# 5.2 MI for regression
mutual_info_regression(X_train, y_train)
# Select the top 10 percentile 注意这里是比例上面是个数
selector = SelectPercentile(mutual_info_regression, percentile = 10).fit(X_train, y_train)
X_train.columns[selector.get_support()]


#### 6. Fischer Score | Chi Square

# 1. Measure the dependecy of 2 variables
# 2. Suited for Categoriacl Variables
# 3. Target should be binary
# 4. Variable values should be non-negative, and typically boolean, frequencies or count
# 5. It compares the observed distribution class with the different labels against the expected one, would there be no labels
# 没怎么看懂

from sklearn.feature_selection import chi2

f_score = chi2(X_train, y_train)
# return了两个array
# 第二个是p-value， the smaller the p_value, the more significant the feature is 
f_score[1]

#### 7. Univariate Feature Selection #### 
# Select the best features based on univariate statistical tests (ANOVA)
# Based on F-test, estimate the degree of linear dependency between 2 random variables
# Assume features are linearly related and normally distributed
# 所以说如果不是线性也不是高斯分布，就不能用？

# 7.1 ANOVA for classification
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile

univaraite = f_classif(X_train, y_train)
# 一样的 两个array第二个是pvalue
univaraite[1]

# lower the p-value, more predictive the feature is 
# 有的时候高的pvalue并不代表没用，因为这里我们assume的线性关系
SelectKBest(f_classif, k=10).fit(X_train, y_train)

# ANOVA for regression
univariate = f_regression(X_train, y_train)
univariate[1]

SelectPercentile(f_regression, percentile = 10).fit(X_train, y_train)

#### Univariate roc-auc or mse ####
# 1. Build one decision tree per feature, to predict the target
# 2. Make predictions using the decision tree and the mentioned feature
# 3. Rank the features according to the ML metric (roc-auc or mse)
# 4. Select the highest ranked features

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, mean_square_error

# 8.1 Univariate roc-auc for Classification
# 原来要自己手动作啊！
roc_values = []
for feature in X_train.columns:
	clf = DecisionTreeClassifier()
	clf.fit(X_train[feature].to_frame(), y_train)
	y_scored = clf.predict_proba(X_test[feature].to_frame())
	roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))

# 之后rank一下选分数高的就好了

# 8.2 Univariate roc-auc for Regression
mse_values = []
for feature in X_train.columns:
	clf = DecisionTreeRegressor()
	clf.fit(X_train[feature].to_frame(), y_train)
	y_scored = clf.predict(X_test[feature].to_frame())
	mse_values.append(mean_square_error(y_test, y_scored))

	# Rank it!

################################# B. Wrapper Methods ############################################

# 1. Forward Selection: add one feature at a time recursively
# 2. Backward Selection: remove one feature at a time recursively
# 3. Exhaustive Search: searches across all possible feature combinations

## Procedure
# 1. Search for the subset of features
# 2. Build the Machine Learning Model on the selected feature subset
# 3. Evaluate Model Performance
# 4. Repeat

# 9. Forward feature selection
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# 9.1 Forward feature selection for Classification
sfs1 = SFS(
	RandomForestClassifier(),
	k_features = 5,
	forward = True,
	floating = False,
	verbose = 2,
	scoring = 'roc_auc'
	cv = 5
)
sfs1 = sfs1.fit(X_train, y_train)
# Get the selected feature
selected_feat = X_train.columns[list(sfs1.k_features_idx_)]
selected_feat

# 9.2 Forward feature selection for Regression
sfs1 = SFS(
	RandomForestRegressor(),
	k_features = 5,
	forward = True,
	floating = False,
	verbose = 2,
	scoring = 'roc_auc'
	cv = 5
)
sfs1 = sfs1.fit(X_train, y_train)
# Get the selected feature
selected_feat = X_train.columns[list(sfs1.k_features_idx_)]
selected_feat

################################# C. Embeded Methods ############################################

# Regularization Methods
# 1. The L1 Regularization (LASSO)
# 2. The L2 Regularization (Ridge)
# 3. The L1/L2 Regularization (Elastic Net)

# 10. Lasso Method
from skelarn.linear_Model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

scaler = StandardScaler()
scaler.fit(X_train)

sel_ = SelectFromModel(LogisticRegression(C = 1, penalty = 'l1', solver = 'liblinear'))
sel_.fit(scaler.transform(X_train), y_train)

# Get the selected feature
sel_.get_support()

selected_feat = X_train.columns[(sel_.get_support())]

# 11. Tree Method (random forest)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from skleanr.metrics import roc_auc_score

sel_ = SelectFromModel(
	RandomForestClassifier()
)
sel_.fit(X_train, y_train)

sel_.get_support()




