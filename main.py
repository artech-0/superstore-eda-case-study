import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from io import StringIO

warnings.filterwarnings('ignore')

def switch_plot_mode(mode):
    if mode == 'dark':
        plt.style.use('dark_background')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
    elif mode == 'light':
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['text.color'] = 'black'
        plt.rcParams['axes.labelcolor'] = 'black'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'

@st.cache_data
def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)

    rows_before = len(data)
    orders_before = data['Order ID'].nunique()
    data.drop_duplicates(inplace=True)
    rows_after = len(data)
    orders_after = data['Order ID'].nunique()
    st.session_state['duplicate_impact'] = {
        'rows_removed': rows_before - rows_after,
        'orders_affected': orders_before - orders_after
    }

    data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%y', errors='coerce')
    data['Ship Date'] = pd.to_datetime(data['Ship Date'], errors='coerce')

    inconsistent_year_mask = (data['Order ID'].str.split('-').str[1] != data['Order Date'].dt.year.astype(str))
    inconsistent_years = data.loc[inconsistent_year_mask]
    st.session_state['date_inconsistency_impact'] = {
        'rows_affected': len(inconsistent_years),
        'orders_affected': inconsistent_years['Order ID'].nunique()
    }
    if not inconsistent_years.empty:
        data.loc[inconsistent_year_mask, 'Order ID'] = \
            'CA-' + data['Order Date'].dt.year.astype(str) + '-' + data['Order ID'].str.split('-').str[2]

    data['Days to Ship'] = (data['Ship Date'] - data['Order Date']).dt.days

    impossible_shipping_mask = data['Days to Ship'] < 0
    extreme_outlier_mask = data['Days to Ship'] > 90
    st.session_state['impossible_shipping_impact'] = {
        'rows_nullified': impossible_shipping_mask.sum() + extreme_outlier_mask.sum(),
        'orders_affected_impossible': data.loc[impossible_shipping_mask, 'Order ID'].nunique(),
        'orders_affected_extreme': data.loc[extreme_outlier_mask, 'Order ID'].nunique()
    }
    data.loc[impossible_shipping_mask | extreme_outlier_mask, ['Ship Date', 'Days to Ship']] = np.nan

    missing_ship_mode_initial_count = data['Ship Mode'].isna().sum()
    st.session_state['missing_ship_mode_impact'] = {
        'rows_missing': missing_ship_mode_initial_count,
        'orders_affected': data.loc[data['Ship Mode'].isna(), 'Order ID'].nunique()
    }

    imputable_ship_mode_mask = data['Ship Mode'].isna() & data['Days to Ship'].notna()
    data.loc[imputable_ship_mode_mask & (data['Days to Ship'] == 0), 'Ship Mode'] = 'Same Day'
    data.loc[imputable_ship_mode_mask & (data['Days to Ship'] == 7), 'Ship Mode'] = 'Standard Class'

    missing_quantity_mask = data['Quantity'].isna()
    st.session_state['missing_quantity_impact'] = {
        'rows_missing': missing_quantity_mask.sum(),
        'orders_affected': data.loc[missing_quantity_mask, 'Order ID'].nunique()
    }
    quantity_median = data['Quantity'].median()
    data['Quantity'].fillna(quantity_median, inplace=True)

    def get_initials(full_name):
        try:
            name_parts = full_name.split()
            initials = ".".join([part[0].upper() for part in name_parts])
            return initials
        except:
            return np.nan
    data['Customer Name Masked'] = data['Customer Name'].apply(get_initials)
    data.drop(columns=['Customer Name'], inplace=True)

    st.session_state['postal_code_impact'] = {
        'rows_affected': data[data['Postal Code'].astype(str).str.len() < 5].shape[0]
    }
    data['Postal Code'] = data['Postal Code'].astype(str).str.zfill(5)

    data['Quantity'] = data['Quantity'].astype(int)
    data['Sales Price'] = data['Sales Price'].astype(float)

    data['State'] = data['State'].str.strip()
    state_map = {
        'CA': 'California',
        'NY': 'New York',
        'TX': 'Texas',
        'NJ': 'New Jersey',
        'WA\\': 'Washington'
    }
    states_before_map = data['State'].copy()
    data['State'] = data['State'].replace(state_map)
    st.session_state['state_cleaning_impact'] = {
        'rows_affected': (states_before_map != data['State']).sum(),
        'unique_states_before': states_before_map.nunique(),
        'unique_states_after': data['State'].nunique()
    }

    negative_price_mask = data['Sales Price'] < 0
    st.session_state['negative_sales_price_impact'] = {
        'rows_affected': negative_price_mask.sum(),
        'orders_affected': data.loc[negative_price_mask, 'Order ID'].nunique()
    }
    if not negative_price_mask.empty:
        data['Sales Price'] = data['Sales Price'].abs()

    data['Original Price'] = data['Sales Price'] / (1 - data['Discount'])
    data['Total Sales'] = data['Sales Price'] * data['Quantity']
    data['Total Profit'] = data['Profit'] * data['Quantity']
    data['Discount Price'] = data['Original Price'] * data['Discount']
    data['Total Discount'] = data['Discount Price'] * data['Quantity']

    conditions = [
        (data['Days to Ship']==0),
        (data['Days to Ship']>=1) & (data['Days to Ship']<=3),
        (data['Days to Ship']>3)
    ]
    choices = ["Immediate","Urgent","Standard"]
    data['Shipping Urgency'] = np.select(conditions,choices,"Standard")

    data.sort_values(['Customer ID', 'Order Date'], inplace=True)
    data['Days Since Last Order'] = data.groupby('Customer ID')['Order Date'].diff().dt.days

    customer_summary_df = data.groupby('Customer ID').agg(
        Customer_Total_Sales=('Total Sales', 'sum'),
        Customer_Total_Quantity=('Quantity', 'sum'),
        Customer_Total_Profit=('Total Profit', 'sum'),
        Customer_Avg_Discount=('Discount', 'mean')
    )
    customer_summary_df.reset_index(inplace=True)
    data = pd.merge(data, customer_summary_df, on='Customer ID', how='left')

    def remove_outliers_func(df_in, col):
        q1 = df_in[col].quantile(0.25)
        q3 = df_in[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        outlier_mask = (df_in[col] < lower_bound) | (df_in[col] > upper_bound)
        return df_in[~outlier_mask]

    rows_before_outliers = len(data)
    orders_before_outliers = data['Order ID'].nunique()

    sales_data_cleaned = remove_outliers_func(data, 'Sales Price')
    sales_and_profit_data_cleaned = remove_outliers_func(sales_data_cleaned, 'Profit')

    st.session_state['outlier_impact'] = {
        'rows_removed': rows_before_outliers - len(sales_and_profit_data_cleaned),
        'orders_affected': orders_before_outliers - sales_and_profit_data_cleaned['Order ID'].nunique()
    }
    data = sales_and_profit_data_cleaned

    customer_summary_for_quintiles = data.groupby('Customer ID').agg(
        Customer_Total_Sales=('Total Sales', 'sum'),
        Customer_Total_Profit=('Total Profit', 'sum')
    ).reset_index()

    data = data.drop(columns=['Customer_Total_Sales', 'Customer_Total_Quantity', 'Customer_Total_Profit', 'Customer_Avg_Discount'], errors='ignore')
    data = pd.merge(data, customer_summary_for_quintiles, on='Customer ID', how='left')

    data['Customer Sales Quintile'] = pd.qcut(
        data['Customer_Total_Sales'],
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    )
    data['Customer Profit Quintile'] = pd.qcut(
        data['Customer_Total_Profit'],
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    )

    return data

st.set_page_config(layout="wide", page_title="Superstore Sales EDA Dashboard")

st.title("🛍️ Superstore Sales EDA Dashboard")

st.sidebar.header("Dashboard Controls")
plot_mode = st.sidebar.radio("Select Plot Theme:", ('dark', 'light'), index=0)
switch_plot_mode(plot_mode)

st.sidebar.markdown("---")
st.sidebar.header("About This Dashboard")
st.sidebar.markdown(
    """
    This dashboard provides an interactive Exploratory Data Analysis (EDA) of the Superstore Sales dataset.
    It covers data cleaning, feature engineering, and key insights into sales, profit, customer behavior,
    shipping, regional performance, and discount impact.
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Data Source: `SuperStore_Dataset.csv`")

file_path = "SuperStore_Dataset.csv"
data = load_and_clean_data(file_path)

if data is not None:
    st.success("Data loaded and cleaned successfully!")
    st.write(f"Final dataset shape: {data.shape[0]} rows, {data.shape[1]} columns")

    with st.expander("View Data Cleaning Impact Summary"):
        st.subheader("Data Cleaning Impact Summary")
        st.write(f"- **Duplicates Removed:** {st.session_state.get('duplicate_impact', {}).get('rows_removed', 0)} rows, affecting {st.session_state.get('duplicate_impact', {}).get('orders_affected', 0)} unique Order IDs.")
        st.write(f"- **Date Inconsistencies Corrected:** {st.session_state.get('date_inconsistency_impact', {}).get('rows_affected', 0)} rows, affecting {st.session_state.get('date_inconsistency_impact', {}).get('orders_affected', 0)} unique Order IDs.")
        st.write(f"- **Impossible Shipping Dates Nullified:** {st.session_state.get('impossible_shipping_impact', {}).get('rows_nullified', 0)} rows. Orders affected: {st.session_state.get('impossible_shipping_impact', {}).get('orders_affected_impossible', 0)} (negative) and {st.session_state.get('impossible_shipping_impact', {}).get('orders_affected_extreme', 0)} (extreme).")
        st.write(f"- **Missing Ship Mode Imputed:** {st.session_state.get('missing_ship_mode_impact', {}).get('rows_missing', 0)} initially missing values, affecting {st.session_state.get('missing_ship_mode_impact', {}).get('orders_affected', 0)} unique Order IDs.")
        st.write(f"- **Missing Quantity Imputed:** {st.session_state.get('missing_quantity_impact', {}).get('rows_missing', 0)} initially missing values, affecting {st.session_state.get('missing_quantity_impact', {}).get('orders_affected', 0)} unique Order IDs. (Using Median: {data['Quantity'].median():.0f})")
        st.write(f"- **Postal Code Formatted:** {st.session_state.get('postal_code_impact', {}).get('rows_affected', 0)} codes padded with leading zeros.")
        st.write(f"- **Negative Sales Price Corrected:** {st.session_state.get('negative_sales_price_impact', {}).get('rows_affected', 0)} rows, affecting {st.session_state.get('negative_sales_price_impact', {}).get('orders_affected', 0)} unique Order IDs.")
        st.write(f"- **State Abbreviations Corrected:** Unique states reduced from {st.session_state.get('state_cleaning_impact', {}).get('unique_states_before', 'N/A')} to {st.session_state.get('state_cleaning_impact', {}).get('unique_states_after', 'N/A')}.")
        st.write(f"- **Outliers Removed (Sales Price & Profit):** {st.session_state.get('outlier_impact', {}).get('rows_removed', 0)} rows removed, affecting {st.session_state.get('outlier_impact', {}).get('orders_affected', 0)} unique Order IDs.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview",
        "📈 Sales & Profit",
        "👥 Customer Segmentation",
        "🚚 Shipping & Regional",
        "🏷️ Discount & Pricing",
        "⏰ Temporal Analysis"
    ])

    with tab1:
        st.header("1. Data Overview & Initial Exploration")
        st.subheader("First Few Rows")
        st.dataframe(data.head())

        st.subheader("Data Information (df.info())")
        buffer = StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("Descriptive Statistics (df.describe())")
        st.dataframe(data.describe(include='all'))

        st.subheader("Missing Values Count")
        st.dataframe(data.isnull().sum().to_frame(name='Missing Count'))

        st.subheader("Unique Values Count")
        st.dataframe(data.nunique().to_frame(name='Unique Count'))

    with tab2:
        st.header("2. Sales and Profit Analysis")

        st.subheader("2.1.1 Top 10 Most Profitable Products")
        profit_df = data.groupby('Product Name')['Total Profit'].sum().sort_values(ascending=False)
        top_n = st.slider("Select Top N Profitable Products", 5, 20, 10, key='top_profit')
        fig, ax = plt.subplots(figsize=(12, 8))
        def truncate_label(label, max_len=40):
            if len(label) > max_len:
                return label[:max_len] + '...'
            return label
        sns.barplot(x=profit_df.head(top_n).values, y=profit_df.head(top_n).index.map(truncate_label), palette='viridis', ax=ax)
        ax.set_title(f'Top {top_n} Most Profitable Products')
        ax.set_xlabel('Total Profit')
        ax.set_ylabel('Product Name')
        st.pyplot(fig)

        st.subheader("2.1.2 Top 10 Most Loss-Making Products")
        loss_df = data.groupby('Product Name')['Total Profit'].sum().sort_values(ascending=True)
        top_n_loss = st.slider("Select Top N Loss-Making Products", 5, 20, 10, key='top_loss')
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=loss_df.head(top_n_loss).values, y=loss_df.head(top_n_loss).index.map(truncate_label), palette='Reds_r', ax=ax)
        ax.set_title(f'Top {top_n_loss} Most Loss-Making Products')
        ax.set_xlabel('Total Loss')
        ax.set_ylabel('Product Name')
        st.pyplot(fig)

        st.subheader("2.1.3 Sales vs. Profit Correlation")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.regplot(x='Total Sales', y='Total Profit', data=data,scatter_kws={'alpha':0.3, 'color':'skyblue'}, line_kws={'color':'red', 'linestyle':'--'}, ax=ax)
        ax.set_title('Total Sales vs. Total Profit')
        ax.set_xlabel('Total Sales')
        ax.set_ylabel('Total Profit')
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

        st.subheader("2.1.4 Joint Distribution of Sales and Profit")
        j = sns.jointplot(x='Total Sales', y='Total Profit', data=data,
                        kind='reg',
                        height=10,
                        scatter_kws={'alpha':0.3, 'color':'skyblue'},
                        line_kws={'color':'red', 'linestyle':'--'})
        j.fig.suptitle('Joint Distribution of Sales and Profit', fontsize=16, y=1.02)
        st.pyplot(j.fig)

        st.subheader("2.1.5 Profit & Loss Concentration Overview")
        total_positive_profit = profit_df[profit_df > 0].sum()
        total_negative_profit = loss_df[loss_df < 0].sum()
        profit_from_top_n = profit_df.head(top_n).sum()
        loss_from_top_n = loss_df.head(top_n_loss).sum()

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=f"Total Positive Profit (All Products)", value=f"${total_positive_profit:,.2f}")
            st.metric(label=f"Profit from Top {top_n} Profitable Products", value=f"${profit_from_top_n:,.2f}")
            st.metric(label=f"% Profit from Top {top_n}", value=f"{(profit_from_top_n / total_positive_profit) * 100:.2f}%")
        with col2:
            st.metric(label=f"Total Loss (All Products)", value=f"${total_negative_profit:,.2f}")
            st.metric(label=f"Loss from Top {top_n_loss} Loss-Making Products", value=f"${loss_from_top_n:,.2f}")
            st.metric(label=f"% Loss from Top {top_n_loss}", value=f"{(loss_from_top_n / total_negative_profit) * 100:.2f}%")

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Concentration of Profits and Losses', fontsize=20)
        profit_slices = [profit_from_top_n, total_positive_profit - profit_from_top_n]
        profit_labels = [f'Top {top_n} Products', 'All Other Profitable Products']
        loss_slices = [abs(loss_from_top_n), abs(total_negative_profit - loss_from_top_n)]
        loss_labels = [f'Top {top_n_loss} Products', 'All Other Loss-Making Products']
        axes[0].pie(profit_slices, labels=profit_labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'skyblue'], explode=(0.1, 0), wedgeprops={'edgecolor': 'white'})
        axes[0].set_title('Share of Total Positive Profit', fontsize=16)
        axes[1].pie(loss_slices, labels=loss_labels, autopct='%1.1f%%', startangle=90, colors=['salmon', 'lightcoral'], explode=(0.1, 0), wedgeprops={'edgecolor': 'white'})
        axes[1].set_title('Share of Total Losses', fontsize=16)
        plt.gca().set_aspect('equal')
        st.pyplot(fig)

    with tab3:
        st.header("3. Customer Segmentation and Analysis")

        st.subheader("3.1 Customer Sales Quintile vs. Customer Profit Quintile Heatmap")
        customer_segmentation_grid = pd.crosstab(
            data['Customer Sales Quintile'],
            data['Customer Profit Quintile']
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(customer_segmentation_grid, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Customer Segmentation: Sales Quintile vs. Profit Quintile')
        ax.set_xlabel('Customer Profit Quintile (1=Low, 5=High)')
        ax.set_ylabel('Customer Sales Quintile (1=Low, 5=High)')
        st.pyplot(fig)

        st.subheader("3.2 Product Category Performance Across Customer Segments")
        sales_profit_pivot = pd.pivot_table(
            data,
            values=['Total Sales', 'Total Profit'],
            index=['Segment', 'Category'],
            aggfunc='sum'
        )
        sales_profit_pivot['Profit to Sales Ratio'] = sales_profit_pivot['Total Profit'] / sales_profit_pivot['Total Sales']
        sorted_pivot = sales_profit_pivot.sort_values(by='Total Profit', ascending=False)
        st.dataframe(sorted_pivot)

        st.subheader("Profit and Sales by Category and Segment")
        plot_data_pivot = sorted_pivot.reset_index()

        fig_profit, ax_profit = plt.subplots(figsize=(14, 8))
        sns.barplot(x='Segment', y='Total Profit', hue='Category', data=plot_data_pivot, palette='viridis', ax=ax_profit)
        ax_profit.set_title('Total Profit from Product Categories by Customer Segment')
        ax_profit.set_xlabel('Customer Segment')
        ax_profit.set_ylabel('Total Profit ($)')
        ax_profit.axhline(0, color='white', linewidth=0.8, linestyle='--')
        ax_profit.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_profit)

        fig_sales, ax_sales = plt.subplots(figsize=(14, 8))
        sns.barplot(x='Segment', y='Total Sales', hue='Category', data=plot_data_pivot, palette='magma', ax=ax_sales)
        ax_sales.set_title('Total Sales from Product Categories by Customer Segment')
        ax_sales.set_xlabel('Customer Segment')
        ax_sales.set_ylabel('Total Sales ($)')
        ax_sales.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_sales)

    with tab4:
        st.header("4. Shipping and Regional Analysis")

        st.subheader("4.1 Distribution of Shipping Urgency")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Shipping Urgency', data=data, palette='cool', order=data['Shipping Urgency'].value_counts().index, ax=ax)
        ax.set_title('Distribution of Orders by Shipping Urgency')
        ax.set_xlabel('Shipping Urgency')
        ax.set_ylabel('Number of Orders')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

        st.subheader("4.2 Days to Ship vs. Profit (Violin Plot)")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.violinplot(x='Shipping Urgency', y='Total Profit', data=data, palette='cool', order=['Immediate', 'Urgent', 'Standard'], ax=ax)
        ax.set_title('Profit Distribution by Shipping Urgency')
        ax.set_xlabel('Shipping Urgency')
        ax.set_ylabel('Total Profit')
        ax.axhline(0, color='white', linewidth=0.8, linestyle='--')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

        st.subheader("4.3 Shipping Mode and Profitability")
        ship_mode_profit = data.dropna(subset=['Ship Mode']).groupby('Ship Mode')['Total Profit'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(x=ship_mode_profit.index, y=ship_mode_profit.values, palette='plasma', ax=ax)
        ax.set_title('Total Profit by Ship Mode')
        ax.set_xlabel('Ship Mode')
        ax.set_ylabel('Total Profit ($)')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

        st.subheader("4.4 Regional Shipping Mode Performance Pivot Table")
        regional_shipping_pivot = pd.pivot_table(
            data.dropna(subset=['Ship Mode']),
            index=['Region', 'Ship Mode'],
            values=['Order ID', 'Total Sales', 'Total Profit'],
            aggfunc={
                'Order ID': 'nunique',
                'Total Sales': 'sum',
                'Total Profit': 'sum'
            }
        )
        regional_shipping_pivot['ProfitMargin'] = regional_shipping_pivot['Total Profit'] / regional_shipping_pivot['Total Sales']
        regional_shipping_pivot['ProfitPerOrder'] = regional_shipping_pivot['Total Profit'] / regional_shipping_pivot['Order ID']
        regional_shipping_pivot.rename(columns={'Order ID': 'Number of Unique Orders'}, inplace=True)
        regional_shipping_pivot.sort_index(inplace=True)
        st.dataframe(regional_shipping_pivot)

        st.subheader("4.5 Regional Sales and Profitability")
        region_performance = data.groupby('Region')[['Total Sales', 'Total Profit']].sum().sort_values(by='Total Profit', ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('Regional Performance Overview', fontsize=18)
        sns.barplot(x=region_performance.index, y=region_performance['Total Sales'], ax=axes[0], palette='Blues_d')
        axes[0].set_title('Total Sales by Region')
        axes[0].set_ylabel('Total Sales')
        axes[0].ticklabel_format(style='plain', axis='y')
        sns.barplot(x=region_performance.index, y=region_performance['Total Profit'], ax=axes[1], palette='Greens_d')
        axes[1].set_title('Total Profit by Region')
        axes[1].set_ylabel('Total Profit')
        axes[1].ticklabel_format(style='plain', axis='y')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        st.pyplot(fig)

        st.subheader("4.6 State-wise Profitability Pivot Table")
        state_pivot = pd.pivot_table(
            data,
            values=['Total Sales', 'Total Profit'],
            index='State',
            aggfunc='sum'
        )
        state_pivot['Profit Margin'] = (state_pivot['Total Profit'] / state_pivot['Total Sales'])
        sorted_state_pivot = state_pivot.sort_values(by='Total Profit', ascending=False)

        col_state1, col_state2 = st.columns(2)
        with col_state1:
            st.write("--- Top 5 Most Profitable States ---")
            st.dataframe(sorted_state_pivot.head(5))
        with col_state2:
            st.write("--- Top 5 Least Profitable States ---")
            st.dataframe(sorted_state_pivot.tail(5))

        st.subheader("4.7 Correlation between State (Encoded) and Profit")
        corr_data = data.copy()
        le = LabelEncoder()
        corr_data['State_Encoded'] = le.fit_transform(corr_data['State'])
        correlation_matrix = corr_data[['State_Encoded', 'Total Profit']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title('Correlation between State (Encoded) and Total Profit')
        st.pyplot(fig)

    with tab5:
        st.header("5. Discount and Pricing Analysis")

        st.subheader("5.1 Impact of Discounts on Profitability")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.regplot(x='Discount', y='Total Profit', data=data,
                    scatter_kws={'alpha': 0.2},
                    line_kws={'color': 'red'}, ax=ax)
        ax.axhline(0, color='white', linewidth=0.8, linestyle='--')
        ax.set_title('Impact of Discount on Profitability')
        ax.set_xlabel('Discount Level')
        ax.set_ylabel('Total Profit')
        ax.grid(True, linestyle='--', alpha=0.1)
        st.pyplot(fig)

        st.subheader("5.2 Original Price vs. Discounted Price by Sub-Category")
        # FIX APPLIED HERE: Removed the extra set of brackets
        price_comparison = data.groupby('Sub-Category')[['Original Price', 'Sales Price']].mean().sort_values(by='Original Price', ascending=False)
        fig, ax = plt.subplots(figsize=(16, 8))
        price_comparison.plot(kind='line', marker='o', ax=ax)
        ax.set_title('Average Original Price vs. Sales Price by Sub-Category')
        ax.set_xlabel('Sub-Category')
        ax.set_ylabel('Average Price ($)')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(['Average Original Price', 'Average Sales Price'])
        plt.tight_layout()
        st.pyplot(fig)

    with tab6:
        st.header("6. Temporal Analysis")

        st.subheader("6.1 Sales and Profit Trends Over Time (Monthly)")
        active_years_counts = data['Order Date'].dt.year.value_counts()
        real_start_year = active_years_counts[active_years_counts > 10].index.min() if not active_years_counts[active_years_counts > 10].empty else data['Order Date'].dt.year.min()
        real_end_year = active_years_counts[active_years_counts > 10].index.max() if not active_years_counts[active_years_counts > 10].empty else data['Order Date'].dt.year.max()

        filtered_data = data[(data['Order Date'].dt.year >= real_start_year) & (data['Order Date'].dt.year <= real_end_year)].copy()
        time_series_data = filtered_data.set_index('Order Date')
        monthly_trends = time_series_data[['Total Sales', 'Total Profit']].resample('M').sum()
        fig_sales_profit, axes_sales_profit = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
        monthly_trends['Total Sales'].plot(kind='line', marker='o', ax=axes_sales_profit[0])
        axes_sales_profit[0].set_title(f'Monthly Sales Trends ({real_start_year}-{real_end_year})')
        axes_sales_profit[0].set_ylabel('Total Sales')
        axes_sales_profit[0].grid(True, linestyle='--', alpha=0.5)
        monthly_trends['Total Profit'].plot(kind='line', marker='o', ax=axes_sales_profit[1], color='orange')
        axes_sales_profit[1].set_title(f'Monthly Profit Trends ({real_start_year}-{real_end_year})')
        axes_sales_profit[1].set_xlabel('Month')
        axes_sales_profit[1].set_ylabel('Total Profit')
        axes_sales_profit[1].grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_sales_profit)

        st.subheader("6.2 Order Frequency by Month")
        filtered_data['Order Month'] = filtered_data['Order Date'].dt.month
        monthly_order_counts = filtered_data.groupby('Order Month')['Order ID'].nunique().sort_index()
        month_labels = [pd.to_datetime(f'2000-{m}-01').strftime('%b') for m in monthly_order_counts.index]
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.barplot(x=month_labels, y=monthly_order_counts.values, palette='flare', ax=ax)
        ax.set_title('Total Number of Unique Orders per Month (Across All Years)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Unique Orders')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

        st.subheader("6.3 Yearly Growth in Sales and Profit")
        yearly_totals = time_series_data[['Total Sales', 'Total Profit']].resample('Y').sum()
        yearly_growth = yearly_totals.pct_change() * 100
        yearly_growth = yearly_growth.dropna()
        yearly_growth.index = yearly_growth.index.year
        st.write("Year-Over-Year Growth (%):")
        st.dataframe(yearly_growth)

        fig_yoy, ax_yoy = plt.subplots(figsize=(12, 7))
        yearly_growth.plot(kind='bar', width=0.8, ax=ax_yoy)
        ax_yoy.set_title('Year-Over-Year Growth in Sales and Profit')
        ax_yoy.set_xlabel('Year')
        ax_yoy.set_ylabel('Growth Rate (%)')
        plt.xticks(rotation=0)
        ax_yoy.axhline(0, color='white', linewidth=0.8, linestyle='--')
        ax_yoy.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_yoy)

        fig_yoy_line, ax_yoy_line = plt.subplots(figsize=(12, 7))
        yearly_growth.plot(kind='line', marker='o', linestyle='--', ax=ax_yoy_line)
        ax_yoy_line.set_title('Year-Over-Year Growth in Sales and Profit')
        ax_yoy_line.set_xlabel('Year')
        ax_yoy_line.set_ylabel('Growth Rate (%)')
        plt.xticks(yearly_growth.index)
        ax_yoy_line.axhline(0, color='white', linewidth=0.8, linestyle='--')
        ax_yoy_line.grid(axis='y', linestyle='--', alpha=0.5)
        ax_yoy_line.legend(['Total Sales Growth', 'Total Profit Growth'])
        st.pyplot(fig_yoy_line)

else:
    st.error("Failed to load or clean data. Please check the file path and data integrity.")