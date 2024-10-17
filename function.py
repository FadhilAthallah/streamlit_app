import pandas as pd
import numpy as np
import math
from scipy.stats import vonmises
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import MinCovDet
from scipy.stats import chi2

def convert_date(df, rttime):
    """
    Converts a datetime column into separate date and time components.

    Parameters:
    rttime : Column name with datetime data.
    """
    df[rttime] = pd.to_datetime(df[rttime], errors='coerce')

    df['month'] = df[rttime].dt.month
    df['date'] = df[rttime].dt.date
    df['day'] = df[rttime].dt.day
    df['hours_minutes'] = df[rttime].dt.strftime('%H:%M')
    df['hours'] = df[rttime].dt.hour

    return df


def calculate_frequencies(df, account_number, transaction_key, channel):
    """
    Calculates transaction frequencies and ratios by account, date, and channel.

    Parameters:
    account_number : Column name for account number.
    rttime : Column name with datetime data.
    transaction_key : Column name for the transaction identifier.
    channel : Column name for the transaction channel.
    """
    
    df['frequency'] = df.groupby([account_number, 'date'])[transaction_key].transform('count')
    df['frequency_by_channel'] = df.groupby([account_number, 'date', channel])[transaction_key].transform('count')
    df['frequency_ratio'] = df['frequency_by_channel'] / df['frequency']

    return df

# ---------- RECENCY ----------
# ---------- RECENCY ----------
# ---------- RECENCY ----------
def previous_transaction(df, rttime, prev): 
    df[rttime] = pd.to_datetime(df[rttime], errors='coerce')
    df[prev] = pd.to_datetime(df[prev], errors='coerce') 
    df['time_diff'] = df[rttime] - df[prev]
    df['time_diff_days'] = df['time_diff'].dt.total_seconds() / 86400

    return df

def previous_transaction2(df, rttime, account_number,channel): 
    df[rttime] = pd.to_datetime(df[rttime])
    df.sort_values(by=[account_number,channel,rttime], inplace=True)
    df['prev_trx_date'] = df.groupby([account_number, channel])[rttime].shift(1)
    df.sort_index(inplace=True)

    df['time_diff'] = df[rttime] - df['prev_trx_date']
    df['time_diff_days'] = df['time_diff'].dt.total_seconds() / 86400

    return df

def recency_feat(data):
    gamma = math.log(0.01)/ (-180)
    result = math.exp( - (gamma)  * data)
    return result

def recency_score(df, time_diff_days):
    df['recency_score'] = df[time_diff_days].apply(lambda x: recency_feat(x) if pd.notnull(x) else 0)
    return df

def hour_stats(df):
    df['hour_mean'] = df.groupby(['trb102'])['hours'].transform('mean')
    df['hour_std'] = df.groupby(['trb102'])['hours'].transform('std')
    df['hour_mean'] = df['hour_mean'].fillna(0)
    df['hour_std'] = df['hour_std'].replace(np.nan, 0)
    return df

def von_mises(mu, std_dev):
    if std_dev <= 0:
        return mu, mu
    else:
        kappa = 1 / std_dev
        lower, upper = vonmises.interval(0.9, kappa, mu)
        
        lower = lower % 24
        upper = upper % 24
        
        if upper < lower:
            return upper, lower
        return lower, upper

def calculate_hour_bound(df, hour_mean, hour_std):
    account_bounds = df.groupby('trb102').agg({
    hour_mean: 'first',
    hour_std: 'first'
        }).reset_index()
    
    account_bounds[['lower_hour_bound', 'upper_hour_bound']] = account_bounds.apply(
    lambda row: pd.Series(von_mises(row[hour_mean], row[hour_std])),
    axis=1
    )

    df = df.merge(account_bounds[['trb102', 'lower_hour_bound', 'upper_hour_bound']], 
              on='trb102', 
              how='left')

    return df

def is_within_bounds(hour, lower, upper):
    if lower <= upper:
        return lower <= hour <= upper
    else:  # interval crosses midnight
        return hour >= lower or hour <= upper
    

def calculate_hour_flag(df, hours, lower_hour_bound, upper_hour_bound):

    df['hour_flag'] = df.apply(
    lambda row: is_within_bounds(row[hours], row[lower_hour_bound], row[upper_hour_bound]),
    axis=1
    )
    return df

# ---------- MONETARY ----------
# ---------- MONETARY ----------
# ---------- MONETARY ----------
def calculate_total_transaction_amount(df, account_number, transaction_amount):
    """
    Calculates the total transaction amount for each account.

    Parameters:
    account_number : Column name for account number.
    transaction_amount : Column name for the transaction amount.
    """

    df['total_transaction_amount'] = df.groupby([account_number,'date'])[transaction_amount].transform('sum')
    return df

def calculate_mad(df, account_number, transaction_amount, k=1.4826):
    """
    Computes the Median Absolute Deviation (MAD) for transaction amounts within each account.

    Parameters:
    account_number : Column name for account number.
    transaction_amount : Column name for transaction amount.
    k : Scaling factor for MAD, default is 1.4826.
    """
    # Ensure the transaction_amount column is numeric, coercing any errors
    df[transaction_amount] = pd.to_numeric(df[transaction_amount], errors='coerce')

    # Handle non-positive values by replacing them with NaN (or drop them if desired)
    df[transaction_amount] = df[transaction_amount].where(df[transaction_amount] > 0)

    # Check for NaN values after filtering
    if df[transaction_amount].isna().all():
        st.warning("All transaction amounts are non-positive or invalid.")
        return df

    # Calculate log of transaction amount
    df['amount_log'] = np.log(df[transaction_amount])

    # Compute median of log amounts grouped by account_number
    median = df.groupby(account_number)['amount_log'].transform('median')

    # Calculate absolute deviation from the median
    abs_deviation = abs(df['amount_log'] - median)

    # Compute the outer median for the absolute deviations
    outer_median = abs_deviation.groupby(df[account_number]).transform('median')

    # Calculate MAD
    mad = k * outer_median

    # Store results back into the DataFrame
    df['median'] = median
    df['absolute_deviation'] = abs_deviation
    df['outer_median'] = outer_median
    df['mad'] = mad

    return df

def calculate_z_score(df, account_number):
    """
    Computes Z-scores for transaction amounts within each account.

    Parameters:
    account_number : Column name for account number.
    transaction_amount : Column name for transaction amount.
    """

    # mean = df.groupby(account_number)['amount_log'].transform('mean')
    std_dev = df.groupby(account_number)['amount_log'].transform('std')
    
    df['standard_deviation'] = round(std_dev, 2)
    df['z_score'] = (df['amount_log'] - df['median']) / df['mad']
    df['amount_log_flag'] = ((df['z_score'] < -3) | (df['z_score'] > 3)).astype(int)
    df['z_score'] = df['z_score'].fillna(0)
    df['z_score'] = df['z_score'].replace([np.inf, -np.inf], 0)

    return df

def calculate_outlier_flags(df, source_number, transaction_amount):
    """
    Flags outliers in transaction amounts using the IQR method.

    Parameters:
    source_number : Column name for the grouping variable.
    transaction_amount : Column name for transaction amount.
    """

    Q3 = df.groupby(source_number)[transaction_amount].transform(lambda x: x.quantile(0.75))
    Q1 = df.groupby(source_number)[transaction_amount].transform(lambda x: x.quantile(0.25))
    
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df['lower_amount_bound'] = lower_bound
    df['upper_amount_bound'] = upper_bound
    df['transaction_amount_flag'] = ((df[transaction_amount] < lower_bound) | (df[transaction_amount] > upper_bound)).astype(int)
    
    return df

def calculate_total_transaction_amount_with_channel_days(df, account_number, channel, transaction_amount, new_column_name):
    """
    Computes the total transaction amount for each account, channel, and day.

    Parameters:
    rttime (str): Column name with datetime data.
    account_number (str): Column name for account number.
    channel (str): Column name for transaction channel.
    transaction_amount (str): Column name for transaction amount.
    new_column_name (str): Name for the new column with the computed totals.
    """
    
    df[new_column_name] = df.groupby([account_number, channel, 'date'])[transaction_amount].transform('sum')
    return df

class RobustMahalanobisDistanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, regularization=1e-6):
        self.regularization = regularization
        self.mcd = MinCovDet(random_state=42)
        
    def fit(self, X, y=None):
        X_subset = X
        self.mcd.fit(X_subset)
        return self
    
    def transform(self, X):
        X_subset = X

        mahalanobis_distances = self.mcd.mahalanobis(X)

        degrees_of_freedom = X_subset.shape[1] 
        threshold = np.sqrt(chi2.ppf(0.975, degrees_of_freedom))

        is_anomaly = mahalanobis_distances > threshold
        return np.column_stack((X, is_anomaly.astype(int)))