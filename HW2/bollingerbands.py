def myStrategy(pastPriceVec, currentPrice):
    import numpy as np
    ma_peroid = 14
    band_w = 2

    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    dataLen = len(pastPriceVec)  # Length of the data vector
    if dataLen == 0:
        return action

    # Compute MA & Sigma
    if dataLen < ma_peroid:
        ma = np.mean(pastPriceVec)  # If given price vector is small than windowSize, compute MA by taking the average
        sigma = np.std(pastPriceVec)
    else:
        windowedData = pastPriceVec[-ma_peroid:]  # Compute the normal MA using windowSize
        ma = np.mean(windowedData)
        sigma = np.std(windowedData)

    # Determine action
    if currentPrice <= ma - (band_w * sigma):  # If price-ma > alpha ==> buy
        action = -1
    elif currentPrice >= ma + (band_w * sigma):  # If price-ma < -beta ==> sell
        action = 1

    return action