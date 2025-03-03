import sys
import numpy as np
import pandas as pd


# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy(pastPriceVec, currentPrice, windowSize, ob, os):
    import numpy as np
    action = 0
    dataLen = len(pastPriceVec)  # Length of the pastPriceVec (all previous prices up to the current day)
    if dataLen == 0:
        return action

    if dataLen < windowSize:
        rsi = None
    else:
        windowedData = pastPriceVec[-windowSize:]
        price_diff = np.diff(windowedData)
        avg_gain = np.mean(price_diff[price_diff > 0]) if np.any(price_diff > 0) else 0  # 平均上漲幅度
        avg_loss = np.mean(-price_diff[price_diff < 0]) if np.any(price_diff < 0) else 0  # 平均下跌幅度
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

    # Determine action
    if rsi is not None:
        if rsi < os:  # buy
            action = 1
        elif rsi > ob:  # sell
            action = -1

    return action


# Compute return rate over a given price vector, with 3 modifiable parameters
def computeReturnRate(priceVec, windowSize, ob, os):
    capital = 1000  # Initial available capital
    capitalOrig = capital  # original capital
    dataCount = len(priceVec)  # day size
    suggestedAction = np.zeros((dataCount, 1))  # Vec of suggested actions
    stockHolding = np.zeros((dataCount, 1))  # Vec of stock holdings
    total = np.zeros((dataCount, 1))  # Vec of total asset
    realAction = np.zeros((dataCount,
                           1))  # Real action, which might be different from suggested action. For instance, when the suggested action is 1 (buy) but you don't have any capital, then the real action is 0 (hold, or do nothing).
    # Run through each day
    for ic in range(dataCount):
        currentPrice = priceVec[ic]  # current price
        suggestedAction[ic] = myStrategy(priceVec[0:ic], currentPrice, windowSize, ob, os)  # Obtain the suggested action
        # get real action by suggested action
        if ic > 0:
            stockHolding[ic] = stockHolding[ic - 1]  # The stock holding from the previous day
        if suggestedAction[ic] == 1:  # Suggested action is "buy"
            if stockHolding[ic] == 0:  # "buy" only if you don't have stock holding
                stockHolding[ic] = capital / currentPrice  # Buy stock using cash
                capital = 0  # Cash
                realAction[ic] = 1
        elif suggestedAction[ic] == -1:  # Suggested action is "sell"
            if stockHolding[ic] > 0:  # "sell" only if you have stock holding
                capital = stockHolding[ic] * currentPrice  # Sell stock to have cash
                stockHolding[ic] = 0  # Stocking holding
                realAction[ic] = -1
        elif suggestedAction[ic] == 0:  # No action
            realAction[ic] = 0
        else:
            assert False
        total[ic] = capital + stockHolding[ic] * currentPrice  # Total asset, including stock holding and cash
    returnRate = (total[-1].item() - capitalOrig) / capitalOrig  # Return rate of this run
    return returnRate


if __name__ == '__main__':
    returnRateBest = -1.00  # Initial best return rate
    df = pd.read_csv(sys.argv[1])  # read stock file
    adjClose = df["Adj Close"].values  # get adj close as the price vector
    adjClose = adjClose[-2500:]
    windowSizeMin = 13;
    windowSizeMax =13;  # Range of windowSize to explore
    obMin = 74;
    obMax = 75;  # Range of alpha to explore
    osMin = 35;
    osMax = 36  # Range of beta to explore
    step = 0.1

    # Start exhaustive search
    for windowSize in range(windowSizeMin, windowSizeMax + 1):  # For-loop for windowSize
        print("windowSize=%d" % (windowSize))
        for ob in np.arange(obMin, obMax + 1, step):  # For-loop for alpha
            print("\tob=%f" % (ob))
            for os in np.arange(osMin, osMax + 1, step):  # For-loop for beta
                print("\t\tos=%f" % (os), end="")  # No newline
                returnRate = computeReturnRate(adjClose, windowSize, ob, os)  # Start the whole run with the given parameters
                print(" ==> returnRate=%f " % (returnRate))
                if returnRate > returnRateBest:  # Keep the best parameters
                    windowSizeBest = windowSize
                    alphaBest = ob
                    betaBest = os
                    returnRateBest = returnRate
    print("Best settings: windowSize=%d, ob=%f, os=%f ==> returnRate=%f" % (
    windowSizeBest, alphaBest, betaBest, returnRateBest))  # Print the best result
