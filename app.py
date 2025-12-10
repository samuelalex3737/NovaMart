import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Marketing Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data():
    """Load all CSV files"""
    try:
        customers = pd.read_csv('customers.csv')
        engagement = pd.read_csv('engagement_data.csv')
        products = pd.read_csv('products.csv')
        journey = pd.read_csv('customer_journey.csv')
        reviews = pd.read_csv('customer_reviews.csv')
        geography = pd.read_csv('geography.csv')
        campaigns = pd.read_csv('campaigns.csv')
        campaign_customers = pd.read_csv('campaign_customers.csv')
        campaign_products = pd.read_csv('campaign_products.csv')
        campaign_engagement = pd.read_csv('campaign_engagement.csv')
        campaign_journey = pd.read_csv('campaign_journey.csv')
        
        return {
            'customers': customers,
            'engagement': engagement,
            'products': products,
            'journey': journey,
            'reviews': reviews,
            'geography': geography,
            'campaigns': campaigns,
            'campaign_customers': campaign_customers,
            'campaign_products': campaign_products,
            'campaign_engagement': campaign_engagement,
            'campaign_journey': campaign_journey
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load all data
data = load_data()

if data is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/business-report.png", width=100)
    st.title("📊 Marketing Analytics")
    st.markdown("---")
    
    # Dataset selector
    st.subheader("Select Analysis View")
    analysis_type = st.selectbox(
        "Choose Analysis Type",
        ["Overview", "Customer Analysis", "Product Analysis", "Campaign Analysis", 
         "Engagement Analysis", "Geographic Analysis", "Customer Journey"]
    )
    
    st.markdown("---")
    st.subheader("Filters")
    
    # Dynamic filters based on analysis type
    if analysis_type == "Customer Analysis":
        income_range = st.slider(
            "Income Range",
            int(data['customers']['Income'].min()),
            int(data['customers']['Income'].max()),
            (int(data['customers']['Income'].min()), int(data['customers']['Income'].max()))
        )
        
        education_filter = st.multiselect(
            "Education Level",
            options=data['customers']['Education'].unique(),
            default=data['customers']['Education'].unique()
        )
    
    elif analysis_type == "Campaign Analysis":
        campaign_filter = st.multiselect(
            "Select Campaigns",
            options=data['campaigns']['CampaignName'].unique(),
            default=data['campaigns']['CampaignName'].unique()
        )
    
    st.markdown("---")
    st.info("💡 **Tip**: Use filters to drill down into specific segments")

# Main content
st.title("🎯 Marketing Analytics Dashboard")
st.markdown("### Comprehensive Marketing Performance Analysis")

# Overview Section
if analysis_type == "Overview":
    st.header("📈 Executive Summary")
    
    # KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_customers = len(data['customers'])
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        total_revenue = data['products']['Revenue'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    
    with col3:
        avg_income = data['customers']['Income'].mean()
        st.metric("Avg Customer Income", f"${avg_income:,.0f}")
    
    with col4:
        total_campaigns = len(data['campaigns'])
        st.metric("Active Campaigns", total_campaigns)
    
    with col5:
        avg_engagement = data['engagement']['EngagementScore'].mean()
        st.metric("Avg Engagement Score", f"{avg_engagement:.2f}")
    
    st.markdown("---")
    
    # Two column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Product Category")
        revenue_by_category = data['products'].groupby('Category')['Revenue'].sum().reset_index()
        fig = px.pie(revenue_by_category, values='Revenue', names='Category', 
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Customer Distribution by Education")
        education_dist = data['customers']['Education'].value_counts().reset_index()
        education_dist.columns = ['Education', 'Count']
        fig = px.bar(education_dist, x='Education', y='Count', 
                     color='Count', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    # Full width chart
    st.subheader("Campaign Performance Overview")
    campaign_perf = data['campaigns'].merge(
        data['campaign_customers'].groupby('CampaignID')['CustomerID'].count().reset_index(),
        on='CampaignID', how='left'
    )
    campaign_perf.columns = ['CampaignID', 'CampaignName', 'StartDate', 'EndDate', 
                             'Budget', 'Channel', 'Customers']
    
    fig = px.bar(campaign_perf, x='CampaignName', y='Customers', color='Channel',
                 title='Customer Reach by Campaign and Channel')
    st.plotly_chart(fig, use_container_width=True)

# Customer Analysis Section
elif analysis_type == "Customer Analysis":
    st.header("👥 Customer Analysis")
    
    # Apply filters
    filtered_customers = data['customers'][
        (data['customers']['Income'] >= income_range[0]) &
        (data['customers']['Income'] <= income_range[1]) &
        (data['customers']['Education'].isin(education_filter))
    ]
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filtered Customers", f"{len(filtered_customers):,}")
    
    with col2:
        avg_age = filtered_customers['Age'].mean()
        st.metric("Average Age", f"{avg_age:.1f}")
    
    with col3:
        avg_income_filtered = filtered_customers['Income'].mean()
        st.metric("Avg Income", f"${avg_income_filtered:,.0f}")
    
    with col4:
        married_pct = (filtered_customers['Marital_Status'] == 'Married').sum() / len(filtered_customers) * 100
        st.metric("Married %", f"{married_pct:.1f}%")
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Demographics", "Income Analysis", "Segmentation"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            fig = px.histogram(filtered_customers, x='Age', nbins=30, 
                             color_discrete_sequence=['#1f77b4'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Marital Status Distribution")
            marital_dist = filtered_customers['Marital_Status'].value_counts().reset_index()
            marital_dist.columns = ['Status', 'Count']
            fig = px.pie(marital_dist, values='Count', names='Status', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Income by Education Level")
            fig = px.box(filtered_customers, x='Education', y='Income', 
                        color='Education', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Income vs Age")
            fig = px.scatter(filtered_customers, x='Age', y='Income', 
                           color='Education', size='Income', 
                           hover_data=['Marital_Status'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Customer Segmentation Matrix")
        
        # Create income segments
        filtered_customers['Income_Segment'] = pd.cut(filtered_customers['Income'], 
                                                      bins=3, labels=['Low', 'Medium', 'High'])
        filtered_customers['Age_Segment'] = pd.cut(filtered_customers['Age'], 
                                                   bins=3, labels=['Young', 'Middle', 'Senior'])
        
        segment_matrix = filtered_customers.groupby(['Income_Segment', 'Age_Segment']).size().reset_index()
        segment_matrix.columns = ['Income_Segment', 'Age_Segment', 'Count']
        
        fig = px.density_heatmap(segment_matrix, x='Age_Segment', y='Income_Segment', 
                                z='Count', color_continuous_scale='YlOrRd')
        st.plotly_chart(fig, use_container_width=True)

# Product Analysis Section
elif analysis_type == "Product Analysis":
    st.header("🛍️ Product Analysis")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_products = len(data['products'])
        st.metric("Total Products", total_products)
    
    with col2:
        total_revenue = data['products']['Revenue'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    
    with col3:
        avg_price = data['products']['Price'].mean()
        st.metric("Avg Product Price", f"${avg_price:.2f}")
    
    with col4:
        total_quantity = data['products']['QuantitySold'].sum()
        st.metric("Total Units Sold", f"{total_quantity:,}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Revenue Analysis", "Product Performance", "Category Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Products by Revenue")
            top_products = data['products'].nlargest(10, 'Revenue')[['ProductName', 'Revenue']]
            fig = px.bar(top_products, x='Revenue', y='ProductName', 
                        orientation='h', color='Revenue', 
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Revenue by Category")
            category_revenue = data['products'].groupby('Category')['Revenue'].sum().reset_index()
            fig = px.pie(category_revenue, values='Revenue', names='Category', 
                        color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Price vs Quantity Sold Analysis")
        fig = px.scatter(data['products'], x='Price', y='QuantitySold', 
                        color='Category', size='Revenue', 
                        hover_data=['ProductName'], 
                        title='Product Performance Matrix')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Selling Products (by Quantity)")
            top_selling = data['products'].nlargest(10, 'QuantitySold')[['ProductName', 'QuantitySold']]
            fig = px.bar(top_selling, x='QuantitySold', y='ProductName', 
                        orientation='h', color_discrete_sequence=['#ff7f0e'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Price Distribution by Category")
            fig = px.violin(data['products'], x='Category', y='Price', 
                          color='Category', box=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Category Performance Metrics")
            category_stats = data['products'].groupby('Category').agg({
                'Revenue': 'sum',
                'QuantitySold': 'sum',
                'Price': 'mean'
            }).reset_index()
            category_stats.columns = ['Category', 'Total Revenue', 'Total Quantity', 'Avg Price']
            st.dataframe(category_stats.style.format({
                'Total Revenue': '${:,.0f}',
                'Total Quantity': '{:,.0f}',
                'Avg Price': '${:.2f}'
            }), use_container_width=True)
        
        with col2:
            st.subheader("Revenue Contribution by Category")
            fig = go.Figure(data=[go.Pie(
                labels=category_stats['Category'],
                values=category_stats['Total Revenue'],
                hole=.3,
                marker_colors=px.colors.qualitative.Set2
            )])
            st.plotly_chart(fig, use_container_width=True)

# Campaign Analysis Section
elif analysis_type == "Campaign Analysis":
    st.header("📢 Campaign Analysis")
    
    # Filter campaigns
    filtered_campaigns = data['campaigns'][data['campaigns']['CampaignName'].isin(campaign_filter)]
    
    # Merge campaign data
    campaign_metrics = filtered_campaigns.merge(
        data['campaign_customers'].groupby('CampaignID').size().reset_index(name='Customers'),
        on='CampaignID', how='left'
    )
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Campaigns", len(filtered_campaigns))
    
    with col2:
        total_budget = filtered_campaigns['Budget'].sum()
        st.metric("Total Budget", f"${total_budget:,.0f}")
    
    with col3:
        total_reach = campaign_metrics['Customers'].sum()
        st.metric("Total Customer Reach", f"{total_reach:,}")
    
    with col4:
        avg_budget = filtered_campaigns['Budget'].mean()
        st.metric("Avg Campaign Budget", f"${avg_budget:,.0f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Campaign Performance", "Channel Analysis", "ROI Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Reach by Campaign")
            fig = px.bar(campaign_metrics, x='CampaignName', y='Customers', 
                        color='Channel', title='Campaign Reach')
            fig.update_xaxis(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Budget Allocation by Campaign")
            fig = px.pie(filtered_campaigns, values='Budget', names='CampaignName', 
                        hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Budget by Channel")
            channel_budget = filtered_campaigns.groupby('Channel')['Budget'].sum().reset_index()
            fig = px.bar(channel_budget, x='Channel', y='Budget', 
                        color='Budget', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Campaign Count by Channel")
            channel_count = filtered_campaigns['Channel'].value_counts().reset_index()
            channel_count.columns = ['Channel', 'Count']
            fig = px.pie(channel_count, values='Count', names='Channel')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Campaign Efficiency Metrics")
        
        # Calculate cost per customer
        campaign_metrics['Cost_Per_Customer'] = campaign_metrics['Budget'] / campaign_metrics['Customers']
        
        fig = px.scatter(campaign_metrics, x='Budget', y='Customers', 
                        size='Cost_Per_Customer', color='Channel',
                        hover_data=['CampaignName'],
                        title='Budget vs Customer Reach (Size = Cost per Customer)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Campaign Performance Table")
        display_df = campaign_metrics[['CampaignName', 'Channel', 'Budget', 'Customers', 'Cost_Per_Customer']]
        st.dataframe(display_df.style.format({
            'Budget': '${:,.0f}',
            'Customers': '{:,.0f}',
            'Cost_Per_Customer': '${:.2f}'
        }), use_container_width=True)

# Engagement Analysis Section
elif analysis_type == "Engagement Analysis":
    st.header("💬 Engagement Analysis")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_engagement = data['engagement']['EngagementScore'].mean()
        st.metric("Avg Engagement Score", f"{avg_engagement:.2f}")
    
    with col2:
        total_interactions = data['engagement']['Interactions'].sum()
        st.metric("Total Interactions", f"{total_interactions:,}")
    
    with col3:
        avg_time = data['engagement']['TimeSpent'].mean()
        st.metric("Avg Time Spent (min)", f"{avg_time:.1f}")
    
    with col4:
        total_reviews = len(data['reviews'])
        st.metric("Total Reviews", f"{total_reviews:,}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Engagement Metrics", "Customer Reviews", "Interaction Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Engagement Score Distribution")
            fig = px.histogram(data['engagement'], x='EngagementScore', 
                             nbins=30, color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Time Spent Distribution")
            fig = px.box(data['engagement'], y='TimeSpent', 
                        color_discrete_sequence=['#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Engagement Score vs Interactions")
        fig = px.scatter(data['engagement'], x='Interactions', y='EngagementScore', 
                        color='TimeSpent', size='Interactions',
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rating Distribution")
            rating_dist = data['reviews']['Rating'].value_counts().sort_index().reset_index()
            rating_dist.columns = ['Rating', 'Count']
            fig = px.bar(rating_dist, x='Rating', y='Count', 
                        color='Count', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Average Rating")
            avg_rating = data['reviews']['Rating'].mean()
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_rating,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [None, 5]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 2], 'color': "lightgray"},
                           {'range': [2, 4], 'color': "gray"},
                           {'range': [4, 5], 'color': "lightgreen"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 
                                   'thickness': 0.75, 'value': 4.5}}))
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Recent Reviews Sample")
        st.dataframe(data['reviews'][['CustomerID', 'ProductID', 'Rating', 'ReviewText']].head(10), 
                    use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 20 Most Engaged Customers")
            top_engaged = data['engagement'].nlargest(20, 'EngagementScore')[['CustomerID', 'EngagementScore']]
            fig = px.bar(top_engaged, x='CustomerID', y='EngagementScore', 
                        color='EngagementScore', color_continuous_scale='Plasma')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Interaction Patterns")
            fig = px.scatter(data['engagement'], x='TimeSpent', y='Interactions', 
                           color='EngagementScore', 
                           color_continuous_scale='Turbo',
                           title='Time Spent vs Interactions')
            st.plotly_chart(fig, use_container_width=True)

# Geographic Analysis Section
elif analysis_type == "Geographic Analysis":
    st.header("🌍 Geographic Analysis")
    
    # Merge geography with customers
    geo_customers = data['geography'].merge(data['customers'], on='CustomerID', how='left')
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        unique_countries = geo_customers['Country'].nunique()
        st.metric("Countries", unique_countries)
    
    with col2:
        unique_cities = geo_customers['City'].nunique()
        st.metric("Cities", unique_cities)
    
    with col3:
        avg_income_geo = geo_customers['Income'].mean()
        st.metric("Avg Income", f"${avg_income_geo:,.0f}")
    
    with col4:
        total_customers_geo = len(geo_customers)
        st.metric("Total Customers", f"{total_customers_geo:,}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Country Analysis", "City Analysis", "Regional Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Distribution by Country")
            country_dist = geo_customers['Country'].value_counts().reset_index()
            country_dist.columns = ['Country', 'Customers']
            fig = px.bar(country_dist, x='Country', y='Customers', 
                        color='Customers', color_continuous_scale='Blues')
            fig.update_xaxis(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Market Share by Country")
            fig = px.pie(country_dist, values='Customers', names='Country', 
                        hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 15 Cities by Customer Count")
            city_dist = geo_customers['City'].value_counts().head(15).reset_index()
            city_dist.columns = ['City', 'Customers']
            fig = px.bar(city_dist, x='Customers', y='City', 
                        orientation='h', color='Customers', 
                        color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Average Income by Top Cities")
            city_income = geo_customers.groupby('City')['Income'].mean().nlargest(15).reset_index()
            fig = px.bar(city_income, x='Income', y='City', 
                        orientation='h', color='Income', 
                        color_continuous_scale='Oranges')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Geographic Performance Matrix")
        
        geo_summary = geo_customers.groupby('Country').agg({
            'CustomerID': 'count',
            'Income': 'mean',
            'Age': 'mean'
        }).reset_index()
        geo_summary.columns = ['Country', 'Customer Count', 'Avg Income', 'Avg Age']
        
        st.dataframe(geo_summary.style.format({
            'Customer Count': '{:,.0f}',
            'Avg Income': '${:,.0f}',
            'Avg Age': '{:.1f}'
        }), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Income Distribution by Country")
            fig = px.box(geo_customers, x='Country', y='Income', 
                        color='Country')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Age Distribution by Country")
            fig = px.violin(geo_customers, x='Country', y='Age', 
                          color='Country', box=True)
            st.plotly_chart(fig, use_container_width=True)

# Customer Journey Section
elif analysis_type == "Customer Journey":
    st.header("🛤️ Customer Journey Analysis")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_touchpoints = len(data['journey'])
        st.metric("Total Touchpoints", f"{total_touchpoints:,}")
    
    with col2:
        unique_customers_journey = data['journey']['CustomerID'].nunique()
        st.metric("Customers Tracked", f"{unique_customers_journey:,}")
    
    with col3:
        conversion_rate = (data['journey']['Conversion'] == 'Yes').sum() / len(data['journey']) * 100
        st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
    
    with col4:
        avg_touchpoints = data['journey'].groupby('CustomerID').size().mean()
        st.metric("Avg Touchpoints/Customer", f"{avg_touchpoints:.1f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Touchpoint Analysis", "Conversion Funnel", "Journey Patterns"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Touchpoint Distribution")
            touchpoint_dist = data['journey']['Touchpoint'].value_counts().reset_index()
            touchpoint_dist.columns = ['Touchpoint', 'Count']
            fig = px.bar(touchpoint_dist, x='Touchpoint', y='Count', 
                        color='Count', color_continuous_scale='Purples')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Conversion by Touchpoint")
            conversion_by_tp = data['journey'].groupby('Touchpoint')['Conversion'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).reset_index()
            conversion_by_tp.columns = ['Touchpoint', 'Conversion Rate']
            fig = px.bar(conversion_by_tp, x='Touchpoint', y='Conversion Rate', 
                        color='Conversion Rate', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Conversion Funnel")
        
        # Create funnel data
        touchpoint_order = ['Website', 'Email', 'Social Media', 'Store Visit', 'Purchase']
        funnel_data = []
        
        for tp in touchpoint_order:
            if tp in data['journey']['Touchpoint'].values:
                count = len(data['journey'][data['journey']['Touchpoint'] == tp])
                funnel_data.append({'Stage': tp, 'Count': count})
        
        funnel_df = pd.DataFrame(funnel_data)
        
        fig = go.Figure(go.Funnel(
            y=funnel_df['Stage'],
            x=funnel_df['Count'],
            textinfo="value+percent initial",
            marker={"color": ["deepskyblue", "lightsalmon", "tan", "teal", "silver"]}
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Overall Conversion Status")
            conversion_status = data['journey']['Conversion'].value_counts().reset_index()
            conversion_status.columns = ['Status', 'Count']
            fig = px.pie(conversion_status, values='Count', names='Status', 
                        color_discrete_sequence=['#ff6b6b', '#51cf66'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Conversion Metrics")
            total_journeys = len(data['journey'])
            conversions = (data['journey']['Conversion'] == 'Yes').sum()
            non_conversions = total_journeys - conversions
            
            metrics_df = pd.DataFrame({
                'Metric': ['Total Journeys', 'Conversions', 'Non-Conversions', 'Conversion Rate'],
                'Value': [total_journeys, conversions, non_conversions, f"{conversion_rate:.2f}%"]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Customer Journey Patterns")
        
        # Journey length analysis
        journey_length = data['journey'].groupby('CustomerID').size().reset_index()
        journey_length.columns = ['CustomerID', 'Touchpoints']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Journey Length Distribution")
            fig = px.histogram(journey_length, x='Touchpoints', 
                             nbins=20, color_discrete_sequence=['#9b59b6'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Avg Touchpoints by Conversion")
            journey_conversion = data['journey'].merge(
                data['journey'].groupby('CustomerID')['Conversion'].first().reset_index(),
                on='CustomerID', suffixes=('', '_final')
            )
            
            avg_tp_conversion = journey_conversion.groupby(['CustomerID', 'Conversion_final']).size().reset_index()
            avg_tp_conversion.columns = ['CustomerID', 'Converted', 'Touchpoints']
            
            fig = px.box(avg_tp_conversion, x='Converted', y='Touchpoints', 
                        color='Converted', color_discrete_sequence=['#e74c3c', '#2ecc71'])
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p>📊 Marketing Analytics Dashboard | Built with Streamlit & Plotly</p>
        <p>Data-driven insights for better marketing decisions</p>
    </div>
    """, unsafe_allow_html=True)