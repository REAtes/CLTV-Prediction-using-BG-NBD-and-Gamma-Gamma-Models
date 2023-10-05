# CLTV Prediction using BG-NBD and Gamma-Gamma Models

## Business Problem
A retail company wants to create a roadmap for its sales and marketing activities. To plan for the medium to long term, the company needs to predict the potential value that existing customers will bring in the future.

## Dataset Description
The dataset consists of information about customers who made their last purchases as OmniChannel (both online and offline) shoppers between 2020 and 2021. The dataset includes the following columns:

- `master_id`: Unique customer identifier
- `order_channel`: The channel through which the purchase was made (Android, iOS, Desktop, Mobile, Offline)
- `last_order_channel`: The channel of the last purchase
- `first_order_date`: The date of the customer's first purchase
- `last_order_date`: The date of the customer's last purchase
- `last_order_date_online`: The date of the customer's last online purchase
- `last_order_date_offline`: The date of the customer's last offline purchase
- `order_num_total_ever_online`: Total number of purchases made by the customer online
- `order_num_total_ever_offline`: Total number of purchases made by the customer offline
- `customer_value_total_ever_offline`: Total amount spent by the customer offline
- `customer_value_total_ever_online`: Total amount spent by the customer online
- `interested_in_categories_12`: List of categories in which the customer made purchases in the last 12 months

## Tasks

### Data Preparation

1. The dataset is read, and data exploration is conducted to understand its structure.
2. Outliers in certain columns are identified and replaced with threshold values.
3. New variables for the total number of purchases and total customer value for OmniChannel customers are created.
4. Date columns are converted to the "date" data type.

### Creating the CLTV Data Structure

1. The analysis date is set as 2 days after the last purchase in the dataset.
2. A new DataFrame (cltv_df) is created to store customer_id, recency_cltv_weekly, T_weekly, frequency, and monetary_cltv_avg.
3. Recency_cltv_weekly, T_weekly, frequency, and monetary_cltv_avg are calculated for each customer.

### Building BG/NBD Models

1. The BG/NBD (Beta Geometric/Negative Binomial Distribution) model is fitted to predict expected customer sales within 3 months, 6 months, 9 months, and 12 months.

### Building Gamma-Gamma Models

1. The Gamma-Gamma model is fitted to predict the expected average customer value.
2. The 6-month CLTV is calculated, and the CLTV values are standardized.
3. The top 20 customers with the highest CLTV are identified.

## Usage
To use the provided code, follow these steps:
1. Ensure you have the required libraries installed, including pandas, datetime, lifetimes, and sklearn.
2. Load the dataset "data.csv" into the appropriate directory.
3. Run the code, which includes data preparation, CLTV modeling, segmentation, and optional functionalization.

### Results

The analysis provides insights into customer segments, predicted sales, and customer value. Specific customer segments and top-performing customers are highlighted. The resulting CLTV segments can help the retail company make data-driven decisions and prioritize marketing efforts.
