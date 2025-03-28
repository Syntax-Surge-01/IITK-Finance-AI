def adjust_for_inflation(df, net_income, market_cap, inflation):
    df['Adjusted_net_income'] = df[net_income] / (1 + df[inflation] / 100)
    df['Adjusted_market_cap'] = df[market_cap] / (1 + df[inflation] / 100)
    return df
