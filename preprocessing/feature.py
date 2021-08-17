import os

import pandas as pd
import ta
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel, DonchianChannel, UlcerIndex


def custom_add_volatility_ta(df: pd.DataFrame, high: str, low: str, close: str):
    colprefix = 'Volatility'
    fillna = True
    # Average True Range
    df[f"{colprefix}volatility_atr"] = AverageTrueRange(
        close=df[close], high=df[high], low=df[low], window=10, fillna=fillna
    ).average_true_range()

    # Bollinger Bands
    indicator_bb = BollingerBands(
        close=df[close], window=20, window_dev=2, fillna=fillna
    )
    df[f"{colprefix}volatility_bbm"] = indicator_bb.bollinger_mavg()
    df[f"{colprefix}volatility_bbh"] = indicator_bb.bollinger_hband()
    df[f"{colprefix}volatility_bbl"] = indicator_bb.bollinger_lband()
    df[f"{colprefix}volatility_bbw"] = indicator_bb.bollinger_wband()
    df[f"{colprefix}volatility_bbp"] = indicator_bb.bollinger_pband()
    df[f"{colprefix}volatility_bbhi"] = indicator_bb.bollinger_hband_indicator()
    df[f"{colprefix}volatility_bbli"] = indicator_bb.bollinger_lband_indicator()

    # Keltner Channel
    indicator_kc = KeltnerChannel(
        close=df[close], high=df[high], low=df[low], window=10, fillna=fillna
    )
    df[f"{colprefix}volatility_kcc"] = indicator_kc.keltner_channel_mband()
    df[f"{colprefix}volatility_kch"] = indicator_kc.keltner_channel_hband()
    df[f"{colprefix}volatility_kcl"] = indicator_kc.keltner_channel_lband()
    df[f"{colprefix}volatility_kcw"] = indicator_kc.keltner_channel_wband()
    # df[f"{colprefix}volatility_kcp"] = indicator_kc.keltner_channel_pband()
    df[f"{colprefix}volatility_kchi"] = indicator_kc.keltner_channel_hband_indicator()
    df[f"{colprefix}volatility_kcli"] = indicator_kc.keltner_channel_lband_indicator()

    # Donchian Channel
    indicator_dc = DonchianChannel(
        high=df[high], low=df[low], close=df[close], window=20, offset=0, fillna=fillna
    )
    df[f"{colprefix}volatility_dcl"] = indicator_dc.donchian_channel_lband()
    df[f"{colprefix}volatility_dch"] = indicator_dc.donchian_channel_hband()
    df[f"{colprefix}volatility_dcm"] = indicator_dc.donchian_channel_mband()
    df[f"{colprefix}volatility_dcw"] = indicator_dc.donchian_channel_wband()
    df[f"{colprefix}volatility_dcp"] = indicator_dc.donchian_channel_pband()

    # Ulcer Index
    df[f"{colprefix}volatility_ui"] = UlcerIndex(
        close=df[close], window=14, fillna=fillna
    ).ulcer_index()
    return df


def read_csv(file):
    df = pd.read_csv(file).drop('Date', axis=1)
    return add_feature(df)


def add_feature(df):
    df = df[df['Open'] != 0]
    df = ta.add_volume_ta(df, 'High', 'Low', 'Close', 'Volume', fillna=True)
    df = ta.add_momentum_ta(df, 'High', 'Low', 'Close', 'Volume', fillna=True)
    df = ta.add_volatility_ta(df, 'High', 'Low', 'Close', fillna=True, colprefix='volatility')
    df = ta.add_trend_ta(df, 'High', 'Low', 'Close', fillna=True)
    df = ta.add_others_ta(df, 'Close', fillna=True)
    return df


def get_banque_data():
    for file_name in os.listdir('../data/banque'):
        yield read_csv(f'../data/banque/{file_name}')
