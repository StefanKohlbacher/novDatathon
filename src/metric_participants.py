#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import numpy as np

alpha = 0.2


# %%
def _confidence_error(df):
    """
    Returns dataframe with the confidence error calculated for each month, region, brand combination
    df: input dataframe with actuals, predictions, lower and upper bounds
    """
    
    width = abs(df['upper'] - df['lower'])
    below = (df['lower'] - df['actuals'])*(df['lower'] > df['actuals'])
    above = (df['actuals'] - df['upper'])*(df['upper'] < df['actuals'])
    outside = (2/alpha)*(below + above)
    return width + outside


def base_metric(df, market_size):
    
    #country level data: contains y_{c,t}^b (actuals) and \hat{y}_{c,t}^b (predictions)
    df_country = df.groupby(['month','brand'])[['predictions','actuals']].sum().reset_index()
        
    #calculates MAE_r^b
    df['abs_error'] = abs(df['predictions'] - df['actuals'])
    metric_region = df.groupby(['brand','region'])['abs_error'].mean()
    
    #calculates MAE_c^b
    df_country['abs_error'] = abs(df_country['predictions'] - df_country['actuals'])
    metric_country = df_country.groupby(['brand'])['abs_error'].mean()
    
    #calculates the aggregated MAE: term MAE^b
    MAE = (metric_country / market_size.sum()) + (metric_region / market_size).groupby(['brand']).mean()
    
    return 10000*MAE.mean()

    
def interval_metric(df, market_size):
    
    #calculates \Delta_{r,t}^b
    df['confidence_error'] = df.apply(_confidence_error, axis=1)
    
    #calculates \Delta_r^b
    metric = df.groupby(['brand','region'])['confidence_error'].mean()
    
    #calculates \Delta^b
    metric = (metric / market_size).groupby('brand').mean()
    
    return 10000*metric.mean()

    
def ComputeMetrics(solution: pd.DataFrame,
                   sales_train: pd.DataFrame,
                   ground_truth: pd.DataFrame):
    
    """
    solution: pd.DataFrame with columns month, region, brand, sales, lower, upper
    sales_train: pd.DataFrame with loaded training data
    ground_truth: pd.DataFrame with columns month, region, brand, sales 
    """
    
    #merge true sales and solutions to single dataframe
    solution = solution.rename(columns = {'sales': 'predictions'})
    ground_truth = ground_truth.rename(columns = {'sales': 'actuals'})
    df = ground_truth.merge(solution, on=['brand','region','month'], how='left')
    
    #Here checking if all brand - region - month combinations have been submitted in the solution
    if df['predictions'].isna().sum() > 0:
        raise Exception('Submitted solution is missing some of the required brand - region - month combinations')
        
    df['month'] = df['month'].apply(lambda x: pd.Period(x, 'M'))
    
    #the average brand_12_market - terms <m_c>, <m_r> - are calculated here
    #only regions in the ground_truth set are selected
    market_size = sales_train.query('brand == "brand_12_market"').groupby('region').mean().squeeze()
    market_size = market_size[ground_truth['region'].unique()]
    
    return base_metric(df, market_size), interval_metric(df, market_size)

