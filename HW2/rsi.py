def myStrategy(pastPriceVec, currentPrice):
    import numpy as np

    windowSize = 25
    ob = 69.1
    os = 32.7
    action = 0
    dataLen = len(pastPriceVec)  # Length of the pastPriceVec (all previous prices up to the current day)
    if dataLen == 0:
        return action

    if dataLen < windowSize:
        return action

    windowedData = pastPriceVec[-windowSize:]
    price_diff = np.diff(windowedData)
    avg_gain = np.mean(price_diff[price_diff > 0]) if np.any(price_diff >= 0) else 0  # 平均上漲幅度
    avg_loss = np.mean(-price_diff[price_diff < 0]) if np.any(price_diff <= 0) else 0  # 平均下跌幅度

    if avg_loss == 0 and avg_gain == 0:
        rsi = 50
    elif avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))


    if rsi < os:  #buy
        action = 1
    elif rsi > ob:  #sell
        action = -1

    return action