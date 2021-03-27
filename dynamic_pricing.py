#######################################################
# DYNAMIC PRICING FOR A NEW PRODUCT WITH AB TESTING
########################################################

# PROJECT STEPS:
# 1. Import libraries & dataset
# 2. Data preprocessing
# 3. Descriptive statistics
# 4. Answers for the questions about pricing:
    # Q1: Does the price of the item differ by categories?
    # Q2: Depending on the first question, what should be the item price?
    # Q3: It is desired to "be able to move" about the price. Create a decision support system for the price strategy.
    # Q4: Simulate item purchases and income for possible price changes.

#######################################################
# STEP 1: IMPORT LIBRARIES & DATA SET
#######################################################

import numpy as np
import pandas as pd
from scipy import stats
import itertools
from matplotlib import pyplot as plt

pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Import set:
df = pd.read_csv("pricing.csv", sep=";")
df.head()
#######################################################
# STEP 2: DATA PREPROCESSING
#######################################################

# Rounding prices:
df["price"] = round(df["price"])

# Checking for abnormal observations and values:
df["price"].describe([0.01,0.25,0.5,0.75,0.99]).T

# Defining functions for replacing outliers with min-max threshold values:
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Replacing outliers with thresholds:
replace_with_thresholds(df,"price")

#######################################################
# STEP 3: DESCRIPTIVE STATISTICS
#######################################################

# Looking the new price distribution:
df["price"].describe([0.01,0.25,0.5,0.75,0.99]).T

# Check medain value:
df["price"].median()

# Descriptive statistics of item price by categories:
df_describe = df.groupby(["category_id"], as_index=False)["price"].agg({"sum","mean","median","count"})
df_describe.reset_index(inplace=True)
df_describe.sort_values("mean")

#######################################################
# STEP 4: ANSWERS for THE QUESTIONS ABOUT PRICING:
#######################################################

####################################################
# Q1. DOES THE PRICE OF THE ITEM DIFFER BY CATEGORY?
####################################################

# Since we compare prices of the item by categories, we can conduct a two-sample t-test to compare mean prices.

# STEP1: DEFINE ALTERNATIVE AND NULL HYPOTHESIS

# H0: There is no significant difference in the price averages of all categories.
# H1: There is a significant difference in the price averages of all categories.

# STEP2: ASSUMPTIONS TESTING FOR TWO SAMPLE T-TEST

# NORMALITY OF DATA DISTRIBUTION

# H0: All category's price metric is normally distributed
# H1: All category's price metric is not normally distributed

for i in df["category_id"].unique():
    test_statistics, pvalue = stats.shapiro(df.loc[df["category_id"] == i, "price"])
    print('Test Statistics = %.4f, p-values = %.4f' % (test_statistics, pvalue))
    if pvalue <= 0.05:
        print(f"The p-value {round(pvalue,4)} is lower than the chosen alpha level (0.05), then the null hypothesis is rejected."
              f"There is evidence that the price of {i} category is not normally distributed")
    else:
        print(f"The p-value {round(pvalue,4)} is higher than the chosen alpha level (0.05), then the null hypothesis is not rejected."
              f"There is evidence that the price of {i} category is normally distributed")

# For all categories, the p-value is lower than the chosen alpha level (0.05), then the null hypothesis is rejected.
# Price metric is not normally distributed.
# VARIABLES NOT MET THE ASSUMPTIONS FOR PARAMETRIC TWO SAMPLE T TES
# Therefore, we will use non-parametric two sample t-test (mannwhitneyu)

# HYPOTHESIS TESITNG - MANNWHITNEYU TEST

# H0: μ1=μ2
# H1: μ1≠μ2

# We need to create two combinations of all categories. (5-2 combination)
category_list = list(itertools.combinations(df_describe['category_id'], 2))

not_rejected = []
rejected = []

for i in category_list:
    test_statistics, pvalue = stats.mannwhitneyu(df.loc[df["category_id"] == i[0], "price"],
                                                 df.loc[df["category_id"] == i[1], "price"])
    if pvalue >= 0.05:
        not_rejected.append(i)
        print(f"categories: {i}, p-value: {round(pvalue,2)}, then the null hypothesis is not rejected.")
    else:
        rejected.append(i)
        print(f"categories: {i}, p-value: {round(pvalue,2)}, then the null hypothesis is rejected.")

# for 201436, 361254, 326584, 675201 categories:
# H0: There is no significant difference in the price averages of all categories. --> NOT REJECTED.

# for 874521, 489756 categories:
# H0: There is no significant difference in the price averages of all categories. --> REJECTED.

########################################################################
# Q2. DEPENDING ON THE FIRST QUESTION, WHAT SHOULD BE THE ITEM PRICE?
########################################################################

df_describe.sort_values("mean")

# When we checkout the descriptive statistics through categories,
# we can observe differences in mean, sum and count statistics. Median values are close to each other.
# If there is no hierarchical or participation fee difference between the categories,
# items price should be equal for all categories.

df["price"].median()
# Median Price: 35.0 can be used as price.

########################################################################################################
# Q3. IT IS DESIRED TO "BE ABLE TO MOVE" ABOUT THE PRICE. CREATE A DECISION SUPPORT SYSTEM FOR THE PRICE STRATEGY.
########################################################################################################

# According price and marketing strategy, different prices can be applied. For example:

# Sorting item prices ascendin order:

df.sort_values("price",inplace=True)
df.reset_index(inplace=True)
del(df["index"])

# Finding least 0.025 value and highest 0.025 value

treshold = 0.95
max_min_index = len(df) * (1-treshold)/2
min_price = df.loc[df.index == round(max_min_index),"price"]
max_price = df.loc[df.index == (len(df) - round(max_min_index)), "price"]

print(f" {round(treshold*100)}% of customers paid for item between {round(min_price.iloc[0])} and {round(max_price.iloc[0])}")

# Profit = Quantity * Price
profit_for_min_price = len(df.loc[df["price"]>=min_price.iloc[0]]) * min_price.iloc[0]
profit_for_max_price = len(df.loc[df["price"]>=max_price.iloc[0]]) * max_price.iloc[0]
cust_for_min_price = len(df.loc[df["price"]>=min_price.iloc[0]])
cust_for_max_price = len(df.loc[df["price"]>=max_price.iloc[0]])

print(f" If we set minimum price as new price our profit will be {round(profit_for_min_price)}")
print(f" If we set maximum price as new price our profit will be {round(profit_for_max_price)}")

# CUMULATIF DISTRUBUTION GRAPH:

data = df["price"]
# calculate the proportional values of samples
p = 1. * np.arange(len(data)) / (len(data) - 1)

# plot the sorted data:
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(data, p)
ax.set_xlabel('$x$')
ax.set_ylabel('$p$')

plt.show()

# WHAT IS THE OPTIUMUM POINT FOR PROFIT?

# 10 * len(df)
a = [34480]

for i in range(1,len(df.index)):
    if df["price"].loc[i] == df["price"].loc[i-1]:
        a.append(a[i-1])
    else:
        a.append((df["price"].loc[i])*((df.index.max() - i) + 1))
print(a)

# Maximum profit:

maximum_profit = max(a)
maxiumum_index = a.index(max(a))
maximum_profit_price = df["price"].loc[maxiumum_index]
num_of_customers = len(df) - maxiumum_index

print(f"maximum profit is {maximum_profit} for item.")

# PROFIT FOR MEAN VALUE OF ITEM PRICE

median_value = df["price"].median()
cust_for_median = len(df.loc[df["price"] >= median_value])
median_profit = len(df.loc[df["price"] >= median_value]) * median_value

print(f"Profit for the median price is {median_profit}")

#####################################################################
# Q4. SIMULATE ITEM PURCHASES AND INCOME FOR POSSIBLE PRICE CHANGES.
#####################################################################

print(f"Minimum Price: {min_price.iloc[0]}, Profit: {round(profit_for_min_price)}, Number of customers: {cust_for_min_price} \n"
      f"Maximum Price: {round(max_price.iloc[0])}, Profit: {round(profit_for_max_price)}, Number of customers: {cust_for_max_price} \n"
      f"Optimum Price: {round(maximum_profit_price)}, Profit: {round(maximum_profit)}, Number of customers: {num_of_customers} \n"
      f"Median Price: {median_value}, Profit: {median_profit}, Number of customers: {cust_for_median}")

# FOR THE MAXIMUM REVENUE AND CUSTOMER SATISFACTION, THE PRICE SHOULD DE 30.00