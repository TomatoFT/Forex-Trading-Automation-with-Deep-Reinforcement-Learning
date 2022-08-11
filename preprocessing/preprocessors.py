import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config import config

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    #_data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data

def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'prcod', 'prchd', 'prcld', 'cshtrd']]

    data['adjcp'] = data['prccd'] 
    data['open'] = data['prcod'] 
    data['high'] = data['prchd'] 
    data['low'] = data['prcld'] 
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    ema = pd.DataFrame()
    boll = pd.DataFrame()
    StRsi = pd.DataFrame()
    smma = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## ema
        temp_ema = stock[stock.tic == unique_ticker[i]]['close_10_ema']
        temp_ema = pd.DataFrame(temp_ema)
        ema = ema.append(temp_ema, ignore_index=True)
        ## adx
        temp_boll = stock[stock.tic == unique_ticker[i]]['boll']
        temp_boll = pd.DataFrame(temp_boll)
        boll = boll.append(temp_boll, ignore_index=True)

        ## Stochastic RSI
        temp_StRsi = stock[stock.tic == unique_ticker[i]]['stochrsi']
        temp_StRsi = pd.DataFrame(temp_StRsi)
        StRsi = StRsi.append(temp_StRsi, ignore_index=True)

        ## SMMA - Smoothed Moving Average
        temp_smma = stock[stock.tic == unique_ticker[i]]['close_7_smma']
        temp_smma = pd.DataFrame(temp_smma)
        smma = smma.append(temp_smma, ignore_index=True)
 #ád

    df['macd'] = macd
    df['rsi'] = rsi
    df['ema'] = ema
    df['boll'] = boll
    df['StRsi'] = StRsi 
    df['smma'] = smma

    return df



def preprocess_data():
    """data preprocessing pipeline"""

    df = load_dataset(file_name=config.TRAINING_DATA_FILE)
    # get data after 2009
    df = df[df.datadate>=20180102]
    # calcualte adjusted price
    df_preprocess = calcualte_price(df)
    # add technical indicators using stockstats
    df_final=add_technical_indicator(df_preprocess)
    # fill the missing values at the beginning
    df_final.fillna(method='bfill',inplace=True)
    return df_final

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df



def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    
    df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0]*start
    #turbulence_index = [0]
    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                #avoid large outlier because of the calculation just begins
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)
    
    
    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index










