import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    file_path = "data.csv"  # Ensure this file is in the same folder as `app.py`
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    return df

df = load_data()

# Streamlit App Title
st.title("📊 Customer Purchase Behavior Dashboard")

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")
selected_country = st.sidebar.selectbox("Select Country", df["Country"].unique())
selected_year_month = st.sidebar.selectbox("Select Year-Month", df["YearMonth"].astype(str).unique())

# Filter Data
filtered_df = df[(df["Country"] == selected_country) & (df["YearMonth"].astype(str) == selected_year_month)]

# Top Selling Products
st.subheader("🏆 Top 10 Best-Selling Products")
top_products = filtered_df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
top_products.plot(kind="bar", ax=ax, color="royalblue")
ax.set_ylabel("Total Quantity Sold")
st.pyplot(fig)

# Monthly Sales Trend
st.subheader("📈 Monthly Sales Trend")
monthly_sales = df.groupby("YearMonth")["Quantity"].sum()
fig, ax = plt.subplots()
monthly_sales.plot(marker="o", ax=ax, color="darkorange")
ax.set_ylabel("Total Quantity Sold")
st.pyplot(fig)

# Top Countries
st.subheader("🌍 Top 10 Countries by Sales")
top_countries = df.groupby("Country")["Quantity"].sum().sort_values(ascending=False).head(10)
st.table(top_countries)

st.write("📌 **Insights:** This dashboard helps businesses understand purchase behavior trends over time, identify top-performing products, and analyze customer demand by region.")
