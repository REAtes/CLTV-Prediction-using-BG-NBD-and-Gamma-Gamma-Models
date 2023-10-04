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

### Task 1: Data Preparation
1. Read the "data.csv" dataset and create a copy of the DataFrame.
2. Define the `outlier_thresholds` and `replace_with_thresholds` functions to handle outliers in certain columns.
3. Identify and replace outliers in the columns: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", and "customer_value_total_ever_online".
4. Create new variables for the total number of purchases and total customer value for OmniChannel customers.
5. Convert date columns to the "date" data type.

### Task 2: Creating the CLTV Data Structure
1. Set the analysis date as 2 days after the last purchase in the dataset.
2. Create a new DataFrame (`cltv_df`) to store customer_id, recency_cltv_weekly, T_weekly, frequency, and monetary_cltv_avg.
3. Calculate `recency_cltv_weekly`, `T_weekly`, `frequency`, and `monetary_cltv_avg` for each customer.

### Task 3: Building BG/NBD and Gamma-Gamma Models, Calculating 6-Month CLTV
1. Fit the BG/NBD model to predict expected customer sales within 3 months and 6 months.
2. Fit the Gamma-Gamma model to predict the expected average customer value.
3. Calculate the 6-month CLTV and standardize the CLTV values.
4. Identify the top 20 customers with the highest CLTV.

### Task 4: Creating Segments Based on CLTV
1. Divide all customers into 4 segments based on their standardized 6-month CLTV and add segment labels to the dataset (`cltv_segment`).
2. Provide short action recommendations for two selected segments.

### Bonus Task: Functionize the Entire Process
1. Create a function (`create_cltv_df`) to perform the entire CLTV prediction process and return the resulting DataFrame.

## Usage
To use the provided code, follow these steps:
1. Ensure you have the required libraries installed, including pandas, datetime, lifetimes, and sklearn.
2. Load the dataset "data.csv" into the appropriate directory.
3. Run the code, which includes data preparation, CLTV modeling, segmentation, and optional functionization.

## Conclusion
This code demonstrates how to prepare and analyze customer data to predict customer lifetime value (CLTV) using BG-NBD and Gamma-Gamma models. The resulting CLTV segments can help the retail company make data-driven decisions and prioritize marketing efforts.
