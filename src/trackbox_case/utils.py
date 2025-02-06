def filling_missing_data(df):

    # Assuming df is your DataFrame
    df = df.apply(lambda x: x.fillna(0) if '_x' in x.name else x.fillna(-3400), axis=0)

    return df