
# ============================================================
# BUSINESS ANALYTICS PROJECT
# Electronics Retail Business Dataset
# ============================================================
# Dataset: electronics_retail_business_dataset_clean.csv
# Records: 5,000 | Columns: 25 | Period: 2024-01-01 to 2025-12-31
#
# Install once if needed:
# pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
#
# Optional:
# pip install openpyxl
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
FILE = "electronics_retail_business_dataset_clean.csv"

df = pd.read_csv(FILE)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 2. DATASET DESCRIPTION / BASIC CHECKS
# ------------------------------------------------------------
print("\nFIRST 5 RECORDS")
print(df.head())

print("\nLAST 5 RECORDS")
print(df.tail())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nUNIQUE VALUES")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

print("\nDATE RANGE")
print(df["Date"].min(), "to", df["Date"].max())

# ------------------------------------------------------------
# 3. DATA PREPROCESSING
# ------------------------------------------------------------
# Remove exact duplicate records if any
df = df.drop_duplicates().copy()

# Convert date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Check invalid numeric values
numeric_cols = df.select_dtypes(include=np.number).columns
print("\nINVALID NUMERIC VALUES")
print((df[numeric_cols] < 0).sum())

# Business-rule checks
print("\nBUSINESS-RULE CHECKS")
print("Quantity <= 0:", (df["Quantity"] <= 0).sum())
print("Sales <= 0:", (df["Sales"] <= 0).sum())
print("Cost <= 0:", (df["Cost"] <= 0).sum())
print("Inventory < 0:", (df["Inventory"] < 0).sum())
print("Demand <= 0:", (df["Demand"] <= 0).sum())

# Missing-value treatment
# Dataset is already complete, but this makes the code reusable.
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# ------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

# 4.1 Sales by product category
category_sales = df.groupby("Product_Category")["Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(9,5))
category_sales.plot(kind="bar")
plt.title("Total Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Sales")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 4.2 Profit by region
region_profit = df.groupby("Region")["Profit"].sum().sort_values(ascending=False)

plt.figure(figsize=(9,5))
region_profit.plot(kind="bar")
plt.title("Total Profit by Region")
plt.xlabel("Region")
plt.ylabel("Profit")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 4.3 Sales distribution
plt.figure(figsize=(9,5))
plt.hist(df["Sales"], bins=30, edgecolor="black")
plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4.4 Profit distribution
plt.figure(figsize=(9,5))
plt.hist(df["Profit"], bins=30, edgecolor="black")
plt.title("Distribution of Profit")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4.5 Sales box plot
plt.figure(figsize=(8,5))
plt.boxplot(df["Sales"], vert=True)
plt.title("Box Plot of Sales")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# 4.6 Product-category pie chart
category_quantity = df.groupby("Product_Category")["Quantity"].sum()

plt.figure(figsize=(7,7))
plt.pie(category_quantity, labels=category_quantity.index, autopct="%1.1f%%")
plt.title("Quantity Sold by Product Category")
plt.show()

# 4.7 Quantity vs Sales scatter plot
plt.figure(figsize=(9,5))
plt.scatter(df["Quantity"], df["Sales"], alpha=0.5)
plt.title("Quantity vs Sales")
plt.xlabel("Quantity")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------
important_vars = [
    "Age", "Quantity", "Unit_Price", "Discount", "Sales",
    "Cost", "Profit", "Delivery_Days",
    "Customer_Satisfaction", "Demand", "Inventory", "Profit_Ratio"
]

desc = pd.DataFrame({
    "Mean": df[important_vars].mean(),
    "Median": df[important_vars].median(),
    "Mode": [df[c].mode().iloc[0] for c in important_vars],
    "Minimum": df[important_vars].min(),
    "Maximum": df[important_vars].max(),
    "Range": df[important_vars].max() - df[important_vars].min(),
    "Variance": df[important_vars].var(),
    "Std_Deviation": df[important_vars].std()
})

print("\nDESCRIPTIVE STATISTICS")
print(desc.round(3))

# ------------------------------------------------------------
# 6. DISTRIBUTION ANALYSIS
# ------------------------------------------------------------
print("\nSALES SKEWNESS:", df["Sales"].skew())

plt.figure(figsize=(9,5))
plt.hist(df["Sales"], bins=40, edgecolor="black")
plt.title("Sales Distribution and Density")
plt.xlabel("Sales")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.boxplot(df["Sales"], vert=False)
plt.title("Sales Box Plot")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. CORRELATION ANALYSIS
# ------------------------------------------------------------
corr_cols = [
    "Age", "Quantity", "Unit_Price", "Discount", "Sales",
    "Cost", "Profit", "Visitors", "Conversions",
    "Delivery_Days", "Customer_Satisfaction",
    "Demand", "Inventory", "Profit_Ratio"
]

corr = df[corr_cols].corr()

print("\nCORRELATION MATRIX")
print(corr.round(3))

plt.figure(figsize=(13,9))
plt.imshow(corr.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. LINEAR REGRESSION
# ------------------------------------------------------------
# Business question:
# How strongly does Quantity affect Sales?

X = df[["Quantity"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nREGRESSION RESULTS")
print("Intercept:", model.intercept_)
print("Coefficient for Quantity:", model.coef_[0])
print("R-squared:", r2_score(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("MAE:", mean_absolute_error(y_test, pred))

print(
    f"Regression equation: Sales = {model.intercept_:.2f} "
    f"+ ({model.coef_[0]:.2f} × Quantity)"
)

# Example prediction
example_quantity = np.array([[3]])
print("Predicted sales for quantity = 3:",
      model.predict(example_quantity)[0])

plt.figure(figsize=(9,5))
plt.scatter(df["Quantity"], df["Sales"], alpha=0.35)
x_line = np.linspace(df["Quantity"].min(), df["Quantity"].max(), 100)
y_line = model.intercept_ + model.coef_[0] * x_line
plt.plot(x_line, y_line)
plt.title("Linear Regression: Quantity vs Sales")
plt.xlabel("Quantity")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9. BUSINESS INTELLIGENCE / DASHBOARD DATA
# ------------------------------------------------------------
# Use these tables in Power BI/Tableau.

kpi = {
    "Total Sales": df["Sales"].sum(),
    "Total Profit": df["Profit"].sum(),
    "Total Orders": df["Order_ID"].nunique(),
    "Average Order Sales": df["Sales"].mean(),
    "Average Satisfaction": df["Customer_Satisfaction"].mean(),
    "Average Profit Ratio": df["Profit_Ratio"].mean()
}
print("\nDASHBOARD KPIs")
for k, v in kpi.items():
    print(k, ":", round(v, 3))

monthly = (
    df.set_index("Date")
      .resample("MS")
      .agg(Sales=("Sales", "sum"),
           Profit=("Profit", "sum"),
           Orders=("Order_ID", "nunique"))
      .reset_index()
)

channel = (
    df.groupby("Marketing_Channel")
      .agg(Sales=("Sales","sum"),
           Profit=("Profit","sum"),
           Visitors=("Visitors","sum"),
           Conversions=("Conversions","sum"))
      .reset_index()
)

category_dashboard = (
    df.groupby("Product_Category")
      .agg(Sales=("Sales","sum"),
           Profit=("Profit","sum"),
           Quantity=("Quantity","sum"))
      .reset_index()
)

print("\nMonthly dashboard data:")
print(monthly.head())

# ------------------------------------------------------------
# 10. PROBABILITY
# ------------------------------------------------------------
# Probability of a return
return_probability = (df["Return"] == "Yes").mean()

# Probability of conversion among website visitors
total_visitors = df["Visitors"].sum()
total_conversions = df["Conversions"].sum()
conversion_probability = total_conversions / total_visitors

print("\nPROBABILITY RESULTS")
print("P(Return = Yes):", return_probability)
print("Website conversion probability:", conversion_probability)

# Probability distribution of demand
demand_prob = df["Demand"].value_counts(normalize=True).sort_index()
print("\nDemand probability distribution:")
print(demand_prob)

plt.figure(figsize=(9,5))
demand_prob.plot(kind="bar")
plt.title("Empirical Probability Distribution of Demand")
plt.xlabel("Demand")
plt.ylabel("Probability")
plt.tight_layout()
plt.show()

# Normal, Uniform and Exponential distributions
rng = np.random.default_rng(42)

normal_values = rng.normal(
    loc=df["Sales"].mean(),
    scale=df["Sales"].std(),
    size=5000
)

uniform_values = rng.uniform(
    low=df["Sales"].min(),
    high=df["Sales"].max(),
    size=5000
)

exponential_values = rng.exponential(
    scale=df["Delivery_Days"].mean(),
    size=5000
)

plt.figure(figsize=(9,5))
plt.hist(normal_values, bins=40)
plt.title("Normal Distribution Simulation")
plt.xlabel("Simulated Sales")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(9,5))
plt.hist(uniform_values, bins=40)
plt.title("Uniform Distribution Simulation")
plt.xlabel("Simulated Sales")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(9,5))
plt.hist(exponential_values, bins=40)
plt.title("Exponential Distribution Simulation")
plt.xlabel("Simulated Delivery Days")
plt.ylabel("Frequency")
plt.show()

# ------------------------------------------------------------
# 11. DECISION MAKING UNDER UNCERTAINTY
# ------------------------------------------------------------
# Decision: choose inventory quantity for a representative product.
# Demand states are Low / Medium / High based on historical demand.
# Payoff = revenue - purchasing cost - leftover holding cost.
# This is a modelling assumption; replace unit economics if your
# instructor provides product-specific inventory costs.

demand_q = df["Demand"].quantile([1/3, 2/3]).values

def demand_state(x):
    if x <= demand_q[0]:
        return "Low"
    elif x <= demand_q[1]:
        return "Medium"
    return "High"

df["Demand_State"] = df["Demand"].apply(demand_state)

state_probs = df["Demand_State"].value_counts(normalize=True).reindex(
    ["Low", "Medium", "High"]
).fillna(0)

# Representative unit economics
avg_unit_price = df["Unit_Price"].mean()
avg_unit_cost = df["Cost"].div(df["Quantity"]).mean()
holding_cost = 0.05 * avg_unit_cost

state_demand = {
    "Low": df.loc[df["Demand_State"]=="Low", "Demand"].mean(),
    "Medium": df.loc[df["Demand_State"]=="Medium", "Demand"].mean(),
    "High": df.loc[df["Demand_State"]=="High", "Demand"].mean()
}

inventory_choices = [3, 5, 7, 9, 12]

decision_rows = []

for Q in inventory_choices:
    expected_value = 0
    for state in ["Low", "Medium", "High"]:
        demand = state_demand[state]
        sold = min(Q, demand)
        leftover = max(Q - demand, 0)
        payoff = sold * avg_unit_price - Q * avg_unit_cost - leftover * holding_cost
        expected_value += state_probs[state] * payoff

        decision_rows.append({
            "Inventory_Choice": Q,
            "Demand_State": state,
            "Probability": state_probs[state],
            "Expected_Demand": demand,
            "Payoff": payoff
        })

decision_df = pd.DataFrame(decision_rows)
ev_table = decision_df.groupby("Inventory_Choice").apply(
    lambda g: np.sum(g["Probability"] * g["Payoff"])
).reset_index(name="Expected_Value")

print("\nDECISION TABLE")
print(ev_table)

best_decision = ev_table.loc[ev_table["Expected_Value"].idxmax()]
print("\nBest inventory choice:", best_decision["Inventory_Choice"])
print("Maximum expected value:", best_decision["Expected_Value"])

# ------------------------------------------------------------
# 12. SAMPLING AND SAMPLING DISTRIBUTION
# ------------------------------------------------------------
# Population = all 5,000 orders
# Sample = random orders
# Technique = simple random sampling

rng = np.random.default_rng(42)

def sampling_distribution(sample_size, repetitions=1000):
    means = []
    for _ in range(repetitions):
        sample = df["Sales"].sample(
            n=sample_size, replace=False,
            random_state=int(rng.integers(0, 1_000_000_000))
        )
        means.append(sample.mean())
    return np.array(means)

for n in [30, 100, 300]:
    means = sampling_distribution(n, 1000)

    print(f"\nSample size = {n}")
    print("Mean of sample means:", means.mean())
    print("SD of sample means:", means.std(ddof=1))
    print("Theoretical standard error:",
          df["Sales"].std(ddof=1) / np.sqrt(n))

    plt.figure(figsize=(9,5))
    plt.hist(means, bins=30)
    plt.axvline(df["Sales"].mean(), linestyle="--")
    plt.title(f"Sampling Distribution of Sales Mean (n={n})")
    plt.xlabel("Sample Mean")
    plt.ylabel("Frequency")
    plt.show()

# ------------------------------------------------------------
# 13. CONFIDENCE INTERVALS
# ------------------------------------------------------------
sales = df["Sales"].dropna()
n = len(sales)
mean_sales = sales.mean()
std_sales = sales.std(ddof=1)
se = std_sales / np.sqrt(n)

confidence_results = []

for confidence in [0.90, 0.95, 0.99]:
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    margin = t_critical * se

    confidence_results.append({
        "Confidence_Level": f"{confidence*100:.0f}%",
        "Mean": mean_sales,
        "Lower": mean_sales - margin,
        "Upper": mean_sales + margin,
        "Margin_of_Error": margin
    })

ci_table = pd.DataFrame(confidence_results)
print("\nCONFIDENCE INTERVALS")
print(ci_table.round(2))

# ------------------------------------------------------------
# 14. BOOTSTRAPPING
# ------------------------------------------------------------
# Bootstrap statistic = median Sales
bootstrap_medians = []

rng = np.random.default_rng(42)

for _ in range(2000):
    sample = rng.choice(sales.values, size=len(sales), replace=True)
    bootstrap_medians.append(np.median(sample))

bootstrap_medians = np.array(bootstrap_medians)
boot_ci = np.percentile(bootstrap_medians, [2.5, 97.5])

print("\nBOOTSTRAP RESULTS")
print("Original median:", np.median(sales))
print("95% bootstrap CI:", boot_ci)

plt.figure(figsize=(9,5))
plt.hist(bootstrap_medians, bins=40)
plt.axvline(boot_ci[0], linestyle="--")
plt.axvline(boot_ci[1], linestyle="--")
plt.title("Bootstrap Distribution of Median Sales")
plt.xlabel("Bootstrap Median")
plt.ylabel("Frequency")
plt.show()

# ------------------------------------------------------------
# 15. HYPOTHESIS TESTING
# ------------------------------------------------------------
# Question:
# Is average customer satisfaction different from 3.5?
#
# H0: mean satisfaction = 3.5
# H1: mean satisfaction != 3.5

test_value = 3.5

t_stat, p_value = stats.ttest_1samp(
    df["Customer_Satisfaction"], test_value
)

alpha = 0.05

print("\nHYPOTHESIS TEST")
print("H0: Mean satisfaction =", test_value)
print("H1: Mean satisfaction !=", test_value)
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < alpha:
    print("Decision: Reject H0")
else:
    print("Decision: Fail to reject H0")

# ------------------------------------------------------------
# 16. A/B TESTING
# ------------------------------------------------------------
# Website Version A vs B
ab = df.groupby("Website_Version").agg(
    Visitors=("Visitors", "sum"),
    Conversions=("Conversions", "sum")
)

ab["Conversion_Rate"] = ab["Conversions"] / ab["Visitors"]
print("\nA/B TEST SUMMARY")
print(ab)

A = ab.loc["A"]
B = ab.loc["B"]

p1 = A["Conversion_Rate"]
p2 = B["Conversion_Rate"]

pooled_p = (A["Conversions"] + B["Conversions"]) / (
    A["Visitors"] + B["Visitors"]
)

se_ab = np.sqrt(
    pooled_p * (1-pooled_p) *
    (1/A["Visitors"] + 1/B["Visitors"])
)

z_stat = (p2 - p1) / se_ab
p_ab = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print("Conversion rate A:", p1)
print("Conversion rate B:", p2)
print("Z-statistic:", z_stat)
print("p-value:", p_ab)

if p_ab < 0.05:
    print("Decision: statistically significant difference.")
else:
    print("Decision: no statistically significant difference.")

plt.figure(figsize=(7,5))
plt.bar(["A", "B"], [p1, p2])
plt.title("A/B Test: Website Conversion Rates")
plt.xlabel("Website Version")
plt.ylabel("Conversion Rate")
plt.show()

# ------------------------------------------------------------
# 17. TIME SERIES ANALYSIS
# ------------------------------------------------------------
monthly_ts = (
    df.set_index("Date")
      .resample("MS")["Sales"]
      .sum()
      .asfreq("MS")
)

plt.figure(figsize=(12,5))
plt.plot(monthly_ts)
plt.title("Monthly Sales Time Series")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(alpha=0.2)
plt.show()

# Monthly seasonality diagnostic (average sales by calendar month)
monthly_seasonality = df.assign(Month=df["Date"].dt.month).groupby("Month")["Sales"].mean()
print("\nAVERAGE SALES BY CALENDAR MONTH (SEASONALITY CHECK)")
print(monthly_seasonality.round(2))

# Train/test split: last 6 months as test set
train = monthly_ts.iloc[:-6]
test = monthly_ts.iloc[-6:]

# Exponential smoothing
ets_model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal=None,
    initialization_method="estimated"
).fit()

ets_forecast = ets_model.forecast(len(test))

# ARIMA
arima_model = ARIMA(train, order=(1,1,1)).fit()
arima_forecast = arima_model.forecast(len(test))

forecast_comparison = pd.DataFrame({
    "Actual": test,
    "Exponential_Smoothing": ets_forecast,
    "ARIMA": arima_forecast
})

print("\nFORECAST COMPARISON")
print(forecast_comparison)

# Forecast accuracy
def forecast_metrics(actual, forecast):
    return {
        "MAE": mean_absolute_error(actual, forecast),
        "MSE": mean_squared_error(actual, forecast),
        "RMSE": np.sqrt(mean_squared_error(actual, forecast))
    }

ets_metrics = forecast_metrics(test, ets_forecast)
arima_metrics = forecast_metrics(test, arima_forecast)

accuracy_table = pd.DataFrame({
    "Exponential Smoothing": ets_metrics,
    "ARIMA": arima_metrics
}).T

print("\nFORECAST ACCURACY")
print(accuracy_table)

plt.figure(figsize=(12,5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, ets_forecast, label="Exponential Smoothing")
plt.plot(test.index, arima_forecast, label="ARIMA")
plt.legend()
plt.title("Forecast Comparison")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

# Future 6-month forecasts using full dataset
final_ets = ExponentialSmoothing(
    monthly_ts,
    trend="add",
    seasonal=None,
    initialization_method="estimated"
).fit()

future_ets = final_ets.forecast(6)

final_arima = ARIMA(monthly_ts, order=(1,1,1)).fit()
future_arima = final_arima.forecast(6)

print("\nNEXT 6 MONTHS - EXPONENTIAL SMOOTHING")
print(future_ets)

print("\nNEXT 6 MONTHS - ARIMA")
print(future_arima)

# ------------------------------------------------------------
# 18. OPTIMIZATION MODELING
# ------------------------------------------------------------
# Goal: choose quantities of product categories to maximize profit
# subject to an inventory/capacity budget.
#
# Decision variables:
# x1..x5 = units of Smartphones, Laptops, Audio,
#          Smart Home, Accessories.
#
# Objective:
# maximize average profit per unit.
#
# Constraint:
# total units <= capacity.
#
# This is a linear programming model based on historical
# category-level unit economics.

category_profit = df.groupby("Product_Category")["Profit"].sum()
category_qty = df.groupby("Product_Category")["Quantity"].sum()

opt_categories = category_profit.index.tolist()
profit_per_unit = (category_profit / category_qty).reindex(opt_categories).values

capacity = 1000

# maximize profit => minimize negative profit
c = -profit_per_unit

# Sum quantities <= capacity
A_ub = np.ones((1, len(opt_categories)))
b_ub = [capacity]

bounds = [(0, None) for _ in opt_categories]

lp_result = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=bounds,
    method="highs"
)

print("\nOPTIMIZATION")
if lp_result.success:
    optimal = pd.DataFrame({
        "Product_Category": opt_categories,
        "Optimal_Units": lp_result.x,
        "Profit_per_Unit": profit_per_unit
    })
    optimal["Expected_Profit"] = (
        optimal["Optimal_Units"] * optimal["Profit_per_Unit"]
    )

    print(optimal)
    print("Maximum estimated profit:", optimal["Expected_Profit"].sum())
else:
    print("Optimization failed:", lp_result.message)

# ------------------------------------------------------------
# 19. SIMULATION MODELING
# ------------------------------------------------------------
# Inventory simulation under 3 policies:
# Conservative, Balanced, Aggressive.
#
# Demand is sampled from the empirical Demand distribution.

rng = np.random.default_rng(42)
demand_values = df["Demand"].values

policies = {
    "Conservative": 5,
    "Balanced": 8,
    "Aggressive": 12
}

selling_price = avg_unit_price
unit_cost = avg_unit_cost
holding_cost_per_unit = 0.05 * unit_cost
shortage_cost_per_unit = 0.10 * unit_cost

simulation_results = []

for policy, inventory_level in policies.items():
    profits = []
    shortages = []
    ending_inventory = []

    for _ in range(10000):
        demand = rng.choice(demand_values)

        sold = min(inventory_level, demand)
        shortage = max(demand - inventory_level, 0)
        leftover = max(inventory_level - demand, 0)

        revenue = sold * selling_price
        purchase_cost = inventory_level * unit_cost
        holding_cost = leftover * holding_cost_per_unit
        shortage_cost = shortage * shortage_cost_per_unit

        profit = revenue - purchase_cost - holding_cost - shortage_cost

        profits.append(profit)
        shortages.append(shortage)
        ending_inventory.append(leftover)

    simulation_results.append({
        "Policy": policy,
        "Average_Profit": np.mean(profits),
        "Average_Shortage": np.mean(shortages),
        "Average_Ending_Inventory": np.mean(ending_inventory),
        "Service_Level": np.mean(np.array(shortages) == 0)
    })

simulation_table = pd.DataFrame(simulation_results)
print("\nSIMULATION RESULTS")
print(simulation_table.round(3))

plt.figure(figsize=(9,5))
plt.bar(
    simulation_table["Policy"],
    simulation_table["Average_Profit"]
)
plt.title("Inventory Policy Simulation: Average Profit")
plt.xlabel("Policy")
plt.ylabel("Average Profit")
plt.show()

# ------------------------------------------------------------
# 20. ANOVA
# ------------------------------------------------------------
# Compare Sales across the three marketing channels.
#
# H0: all channel mean sales are equal
# H1: at least one mean is different

anova_model = ols(
    "Sales ~ C(Marketing_Channel)",
    data=df
).fit()

anova_table = sm.stats.anova_lm(anova_model, typ=2)

print("\nANOVA: SALES BY MARKETING CHANNEL")
print(anova_table)

# Optional group means
print("\nGROUP MEANS")
print(df.groupby("Marketing_Channel")["Sales"].agg(["mean","count","std"]))

# ------------------------------------------------------------
# 21. EXPERIMENTAL DESIGN
# ------------------------------------------------------------
# Factors:
# Factor 1 = Website Version (A/B)
# Factor 2 = Discount Level (Low/High)
# Response = Sales
#
# Discount level is created from the median discount.
# This is an observational factorial analysis of the available
# dataset, not a controlled randomized experiment.

discount_median = df["Discount"].median()

df["Discount_Level"] = np.where(
    df["Discount"] <= discount_median, "Low", "High"
)

factorial_model = ols(
    "Sales ~ C(Website_Version) * C(Discount_Level)",
    data=df
).fit()

factorial_anova = sm.stats.anova_lm(factorial_model, typ=2)

print("\nFACTORIAL EXPERIMENT / ANOVA")
print(factorial_anova)

factor_means = (
    df.groupby(["Website_Version", "Discount_Level"])["Sales"]
      .mean()
      .reset_index()
)

print("\nFACTORIAL CELL MEANS")
print(factor_means)

plt.figure(figsize=(9,5))
pivot_means = factor_means.pivot(index="Website_Version", columns="Discount_Level", values="Sales")
pivot_means.plot(kind="bar")
plt.title("Website Version × Discount Level: Average Sales")
plt.xlabel("Website Version")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 22. ADDITIONAL BUSINESS INSIGHTS TABLES
# ------------------------------------------------------------

# Top products
top_products = (
    df.groupby("Product")
      .agg(
          Sales=("Sales","sum"),
          Profit=("Profit","sum"),
          Quantity=("Quantity","sum"),
          Orders=("Order_ID","nunique")
      )
      .sort_values("Sales", ascending=False)
)

print("\nTOP PRODUCTS BY SALES")
print(top_products.head(10))

# Region performance
region_summary = (
    df.groupby("Region")
      .agg(
          Sales=("Sales","sum"),
          Profit=("Profit","sum"),
          Avg_Satisfaction=("Customer_Satisfaction","mean"),
          Orders=("Order_ID","nunique")
      )
      .sort_values("Profit", ascending=False)
)

print("\nREGION SUMMARY")
print(region_summary)

# Return analysis
return_summary = (
    df.groupby("Return")
      .agg(
          Orders=("Order_ID","count"),
          Sales=("Sales","sum"),
          Avg_Satisfaction=("Customer_Satisfaction","mean")
      )
)

print("\nRETURN SUMMARY")
print(return_summary)

# ------------------------------------------------------------
# 23. AUTOMATIC PROJECT SUMMARY VALUES
# ------------------------------------------------------------
print("\n" + "="*60)
print("PROJECT SUMMARY VALUES")
print("="*60)

print(f"Total records: {len(df):,}")
print(f"Total sales: ₹{df['Sales'].sum():,.2f}")
print(f"Total profit: ₹{df['Profit'].sum():,.2f}")
print(f"Average order sales: ₹{df['Sales'].mean():,.2f}")
print(f"Average satisfaction: {df['Customer_Satisfaction'].mean():.2f}")
print(f"Return rate: {return_probability:.2%}")
print(f"Website A conversion: {p1:.2%}")
print(f"Website B conversion: {p2:.2%}")
print(f"Highest-sales category: {category_sales.idxmax()}")
print(f"Highest-profit region: {region_profit.idxmax()}")
print(f"Best simulated inventory policy: "
      f"{simulation_table.loc[simulation_table['Average_Profit'].idxmax(), 'Policy']}")

print("\nIMPORTANT:")
print("Use the actual outputs generated by your run in the written report.")
print("Do not copy example interpretations without checking the output.")
