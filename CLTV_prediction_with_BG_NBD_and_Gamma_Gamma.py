# ### Data Preparation
# 1. Read the "data.csv" dataset and create a copy of the DataFrame.
# 2. Define the outlier_thresholds and replace_with_thresholds functions to handle outliers in certain columns.
# 3. Identify and replace outliers in the columns: "order_num_total_ever_online", "order_num_total_ever_offline",
# "customer_value_total_ever_offline", and "customer_value_total_ever_online".
# 4. Create new variables for the total number of purchases and total customer value for OmniChannel customers.
# 5. Convert date columns to the "date" data type.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1500)

df = pd.read_csv("data.csv")


def check_df(dataframe, head=5):
    print("#################### Shape ####################")
    print(dataframe.shape)
    print("#################### Types ####################")
    print(dataframe.dtypes)
    print("#################### Num of Unique ####################")
    print(dataframe.nunique())  # "dataframe.nunique(dropna=False)" yazarsak null'larıda veriyor.
    print("#################### Head ####################")
    print(dataframe.head(head))
    print("#################### Tail ####################")
    print(dataframe.tail(head))
    print("#################### NA ####################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ####################")
    print(dataframe.describe([0.01, 0.05, 0.75, 0.90, 0.95, 0.99]).T)


check_df(df)


def grab_col_names(dataframe, cat_th=16, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken
        sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_but_car = [col for col in cat_but_car if col != "master_id"]


def cat_summary(dataframe, col_name, bar=False, pie=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if dataframe[col_name].dtype == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)
    if bar:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
    if pie:
        col_name = dataframe[col_name].value_counts()
        data = col_name.values
        keys = col_name.keys().values
        plt.pie(data, labels=keys, autopct='%.0f%%')
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, pie=True)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)


for col in num_cols:
    replace_with_thresholds(df, col)



df["total_transaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)


# ### Creating the CLTV Data Structure
# 1. Set the analysis date as 2 days after the last purchase in the dataset.
# 2. Create a new DataFrame (cltv_df) to store customer_id, recency_cltv_weekly, T_weekly, frequency, and
# monetary_cltv_avg.
# 3. Calculate recency_cltv_weekly, T_weekly, frequency, and monetary_cltv_avg for each customer.

last_order_date = df["last_order_date"].max()
analysing_date = dt.datetime(2021, 6, 1)

rfm_df = df.groupby("master_id")\
    .agg({"last_order_date": lambda last_order_date: (analysing_date - last_order_date.max()).days,
          "total_transaction": lambda total_transaction: total_transaction.sum(),
          "total_value": lambda total_value: total_value.sum()})

rfm_df.columns = ["receny", "frequency", "monetary"]
rfm_df.reset_index()

rfm_df["recency_score"] = pd.qcut(rfm_df["receny"], 5, [5, 4, 3, 2, 1])
rfm_df["frequency_score"] = pd.qcut(rfm_df["frequency"].rank(method="first"), 5, [1, 2, 3, 4, 5])
rfm_df["monetary_score"] = pd.qcut(rfm_df["monetary"], 5, [1, 2, 3, 4, 5])
rfm_df["RF_score"] = rfm_df["recency_score"].astype(str) +\
                     rfm_df["frequency_score"].astype(str)
rfm_df["RFM_score"] = rfm_df['recency_score'].astype(str) +\
                      rfm_df['frequency_score'].astype(str) +\
                      rfm_df['monetary_score'].astype(str)

rfm_df = rfm_df.drop(["recency_score", "frequency_score", "monetary_score"], axis=1)

cltv_df = pd.DataFrame()
cltv_df["master_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7
cltv_df["T_weekly"] = (analysis_date - df["first_order_date"]).dt.days / 7
cltv_df["frequency"] = df["total_transaction"]
cltv_df["monetary_cltv_avg"] = df["total_value"] / df["total_transaction"]

cltv_df = pd.merge(rfm_df, cltv_df, on=["master_id", "frequency"])

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at risk",
    r"[1-2]5": "can't lose them",
    r"3[1-2]": "about to sleep",
    r"33": "need attention",
    r"[3-4][4-5]": "loyal customers",
    r"41": "promising",
    r"[4-5][2-3]": "potential loyal lists",
    r"51": "new customers",
    r"5[4-5]": "champions"
}

cltv_df["segment"] = cltv_df["RF_score"].replace(seg_map, regex=True)


# ### Building BG/NBD Models
# 1. Fit the BG/NBD model to predict expected customer sales within 3 months and 6 months.

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])


cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])


cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

cltv_df["exp_sales_9_month"] = bgf.predict(4*9,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

cltv_df["exp_sales_12_month"] = bgf.predict(4*12,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

top_10 = cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

# ### Building Gamma-Gamma Models
# 1. Fit the Gamma-Gamma model to predict the expected average customer value.
# 2. Calculate the 6-month CLTV and standardize the CLTV values.
# 3. Identify the top 20 customers with the highest CLTV.

ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=12,
                                   freq="M",
                                   discount_rate=0.01)

top_20 = cltv_df.sort_values("cltv", ascending=False)[:20]

cltv_df["cltv_score"] = pd.qcut(cltv_df["cltv"], 5, ["E", "D", "C", "B", "A"])

most_valuable_customer = cltv_df[(cltv_df["segment"] == "champions") & (cltv_df["cltv_score"] == "A")]
