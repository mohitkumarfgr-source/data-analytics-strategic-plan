import pandas as pd
from scipy.stats import ttest_ind, pearsonr

df = pd.read_csv("data/customer_analytics.csv")
print("Rows:", len(df))
print("Total revenue:", round(df.revenue.sum(),2))
print("Churn rate:", round(df.churn.mean()*100,2), "%")
print("Average satisfaction:", round(df.satisfaction_score.mean(),2))
print("Average conversion:", round(df.conversion_rate.mean()*100,2), "%")

m = df.delivery_delay_days.median()
high = df.loc[df.delivery_delay_days > m, "satisfaction_score"]
low = df.loc[df.delivery_delay_days <= m, "satisfaction_score"]
t,p = ttest_ind(high,low,equal_var=False)
r,pr = pearsonr(df.delivery_delay_days,df.satisfaction_score)
print("Welch t-test:", round(t,4), "p=", p)
print("Pearson r:", round(r,4), "p=", pr)
