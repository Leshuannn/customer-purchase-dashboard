import streamlit as st
import pandas as pd
import zipfile
import altair as alt
import matplotlib.pyplot as plt
import datetime

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

st.write("📌 **Insights:** This dashboard provides an interactive way to analyze purchase trends, identify top products, and track revenue.")
