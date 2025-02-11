import streamlit as st
import pandas as pd
import zipfile
import altair as alt
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

# Extract & Load Data
@st.cache_data
def load_data():
    with zipfile.ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extractall()  # Extracts data.csv

    df = pd.read_csv("data.csv", encoding="ISO-8859-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    return df

df = load_data()

# Streamlit App Title
st.title("📊 Customer Purchase Behavior Dashboard")
st.write("🔹 **Explore customer purchase patterns, top-selling products, and trends.**")

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")

# Multi-select for Country
selected_countries = st.sidebar.multiselect(
    "Select Country", df["Country"].unique(), default=["United Kingdom"]
)

# Multi-select for Product
selected_products = st.sidebar.multiselect(
    "Select Product", df["Description"].unique(), default=df["Description"].unique()[:5]
)

# Date Range Picker
start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [df["InvoiceDate"].min(), df["InvoiceDate"].max()],
    df["InvoiceDate"].min(),
    df["InvoiceDate"].max(),
)

# Apply Filters
filtered_df = df[
    (df["Country"].isin(selected_countries)) &
    (df["Description"].isin(selected_products)) &
    (df["InvoiceDate"] >= pd.to_datetime(start_date)) &
    (df["InvoiceDate"] <= pd.to_datetime(end_date))
]

st.write(f"✅ Showing data for {len(filtered_df)} transactions from {start_date} to {end_date}")

# KPI Metrics
st.subheader("📊 Key Performance Indicators (KPIs)")
total_revenue = (filtered_df["Quantity"] * filtered_df["UnitPrice"]).sum()
total_sales = filtered_df["Quantity"].sum()
total_customers = filtered_df["CustomerID"].nunique()
col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
col2.metric("📦 Total Sales", f"{total_sales:,}")
col3.metric("👥 Unique Customers", f"{total_customers}")

# 💰 Fix Revenue Breakdown by Country & Product
st.subheader("💰 Revenue Breakdown by Country & Product")

# Calculate Total Revenue (Quantity * UnitPrice)
revenue_by_country = (
    filtered_df.groupby("Country").apply(lambda x: (x["Quantity"] * x["UnitPrice"]).sum())
    .reset_index()
    .rename(columns={0: "Total Revenue"})
    .sort_values(by="Total Revenue", ascending=False)
)

# Debugging Step: Show Data Table to Verify Revenue Calculation
st.write("🔍 **Debugging: Revenue Data Check**")
st.dataframe(revenue_by_country)

# Fix Chart to Show Revenue for Multiple Countries
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(revenue_by_country["Country"], revenue_by_country["Total Revenue"], color="lightblue")
ax.set_title("Total Revenue by Country")
ax.set_ylabel("Revenue ($)")
ax.set_xlabel("Country")
ax.set_xticklabels(revenue_by_country["Country"], rotation=45)

st.pyplot(fig)


# Profit Margin Analysis
st.subheader("💰 Profit Margin Analysis")
df["CostPrice"] = df["UnitPrice"] * 0.6  # Assume cost is 60% of selling price
df["ProfitMargin"] = df["UnitPrice"] - df["CostPrice"]
profit_by_product = df.groupby("Description")["ProfitMargin"].mean().sort_values(ascending=False).head(10)
st.bar_chart(profit_by_product)

# Customer Segmentation (RFM Analysis)
st.subheader("👥 Customer Segmentation (RFM Analysis)")
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (df["InvoiceDate"].max() - x.max()).days,  # Recency
    "InvoiceNo": "count",  # Frequency
    "UnitPrice": "sum"  # Monetary Value
}).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "UnitPrice": "Monetary"})
st.write(rfm.sort_values(by="Monetary", ascending=False).head(10))

# Top Selling Products (Interactive Bar Chart)
st.subheader("🏆 Top 10 Best-Selling Products")
top_products = filtered_df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
chart = (
    alt.Chart(top_products.reset_index())
    .mark_bar()
    .encode(
        x=alt.X("Quantity:Q", title="Total Quantity Sold"),
        y=alt.Y("Description:N", title="Product", sort="-x"),
        tooltip=["Description", "Quantity"],
    )
    .interactive()
)
st.altair_chart(chart, use_container_width=True)

# Monthly Sales Trend (Interactive Line Chart)
st.subheader("📈 Monthly Sales Trend")
monthly_sales = df.groupby("YearMonth")["Quantity"].sum()
fig, ax = plt.subplots()
monthly_sales.plot(marker="o", ax=ax, color="darkorange")
ax.set_ylabel("Total Quantity Sold")
st.pyplot(fig)

# Top Countries by Sales
st.subheader("🌍 Top 10 Countries by Sales")
top_countries = df.groupby("Country")["Quantity"].sum().sort_values(ascending=False).head(10)
st.table(top_countries)

# Sales Forecasting - Long-Term Prediction
st.subheader("📊 Predict Future Sales")

# User Input: Select Forecast Length
n_months = st.slider("Select months to predict:", min_value=6, max_value=36, value=12, step=3)

# Prepare Data for Forecasting
df_sales = df.groupby(df["InvoiceDate"].dt.to_period("M"))["Quantity"].sum().reset_index()
df_sales["InvoiceDate"] = df_sales["InvoiceDate"].astype(str).str.replace("-", "").astype(int)

# Train Model
X = df_sales["InvoiceDate"].values.reshape(-1, 1)
y = df_sales["Quantity"].values
model = LinearRegression()
model.fit(X, y)

# Predict Future Months
future_months = np.array([df_sales["InvoiceDate"].max() + i for i in range(1, n_months + 1)]).reshape(-1, 1)
predicted_sales = model.predict(future_months)

# Combine Past Data & Predictions
forecast_df = pd.DataFrame({"InvoiceDate": df_sales["InvoiceDate"], "Sales": df_sales["Quantity"]})
forecast_future = pd.DataFrame({"InvoiceDate": future_months.flatten(), "Sales": predicted_sales})

# Plot Both Historical & Predicted Data
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(forecast_df["InvoiceDate"], forecast_df["Sales"], marker="o", label="Actual Sales", color="blue")
ax.plot(forecast_future["InvoiceDate"], forecast_future["Sales"], marker="o", linestyle="dashed", label="Predicted Sales", color="red")

ax.set_title(f"Sales Forecast (Next {n_months} Months)")
ax.set_xlabel("YearMonth")
ax.set_ylabel("Total Sales")
ax.legend()
ax.grid(True)

st.pyplot(fig)


# Download Data
st.subheader("📥 Download Filtered Data")
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(label="Download CSV", data=csv, file_name="filtered_data.csv", mime="text/csv")

st.write("📌 **Insights:** This dashboard provides an interactive way to analyze purchase trends, identify top products, track revenue, and predict future sales.")
