import sys
import numpy as np
import pandas as pd

short_ema_prev = None
long_ema_prev = None
signal_line_prev = None
macd_prev = None

def myStrategy(pastPriceVec, currentPrice, short_n, long_n, signal_n, ma_peroid, band_w):
    import numpy as np
    global short_ema_prev, long_ema_prev, signal_line_prev, macd_prev  # Use global variables to store previous values

    action = 0
    dataLen = len(pastPriceVec)  # Length of the pastPriceVec (all previous prices up to the current day)
    if dataLen == 0:
        short_ema_prev = None
        long_ema_prev = None
        signal_line_prev = None
        macd_prev = None
        return action
    # Initialize EMAs for the first time or continue updating from previous values
    if short_ema_prev is None or dataLen == short_n:
        short_ema_prev = np.mean(pastPriceVec)  # Initialize with SMA for short EMA
    if long_ema_prev is None or dataLen == long_n:
        long_ema_prev = np.mean(pastPriceVec)  # Initialize with SMA for long EMA
    if macd_prev is None or signal_line_prev is None:
        macd_prev = 0
        signal_line_prev = 0

    # Smoothing factors
    alpha_short = 2 / (short_n + 1)
    alpha_long = 2 / (long_n + 1)
    alpha_signal = 2 / (signal_n + 1)

    # Update short EMA with new price
    short_ema = (currentPrice * alpha_short) + (short_ema_prev * (1 - alpha_short))
    # Update long EMA with new price
    long_ema = (currentPrice * alpha_long) + (long_ema_prev * (1 - alpha_long))

    # Calculate MACD
    macd = short_ema - long_ema

    # Update signal line (EMA of MACD)
    signal_line = (macd * alpha_signal) + (signal_line_prev * (1 - alpha_signal))

    if dataLen < ma_peroid:
        ma = np.mean(pastPriceVec)  # If given price vector is small than windowSize, compute MA by taking the average
        sigma = np.std(pastPriceVec)
    else:
        windowedData = pastPriceVec[-ma_peroid:]  # Compute the normal MA using windowSize
        ma = np.mean(windowedData)
        sigma = np.std(windowedData)

    # Determine action based on MACD and Signal Line

    if macd > signal_line and currentPrice >= ma + (band_w * sigma):
        action = 1  # Buy signal
    elif macd < signal_line and currentPrice <= ma - (band_w * sigma):
        action = -1  # Sell signal

    # Store the current EMAs and MACD for the next call
    short_ema_prev = short_ema
    long_ema_prev = long_ema
    macd_prev = macd
    signal_line_prev = signal_line

    return action

def computeReturnRate(priceVec, short_n, long_n, signal_n, ma_peroid, band_w):
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
        suggestedAction[ic] = myStrategy(priceVec[0:ic], currentPrice, short_n, long_n, signal_n, ma_peroid, band_w)  # Obtain the suggested action
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

if __name__=='__main__':
    returnRateBest=-1.00	 # Initial best return rate
    df=pd.read_csv(sys.argv[1])	# read stock file
    adjClose=df["Adj Close"].values		# get adj close as the price vector
    short_n_Min=8; short_n_Max=20;	# Range of windowSize to explore
    long_n_Min=20; long_n_Max=60;			# Range of alpha to explore
    signal_n_Min=5; signal_n_Max=15				# Range of beta to explore
    ma_p_Min = 5; ma_p_Max = 35;  # Range of alpha to explore
    band_Min = 1; band_Max = 3
    # Start exhaustive search
    for ma_p in range(ma_p_Min, ma_p_Max + 1):
        print("ma_p=%d" % (ma_p))
        for band in range(band_Min, band_Max + 1):
            print("\tband=%d" % (band))
            for windowSize in range(short_n_Min, short_n_Max+1):		# For-loop for windowSize
                print("\tshort_n=%d" %(windowSize))
                for alpha in range(long_n_Min, long_n_Max+1):	    	# For-loop for alpha
                    print("\tlong_n=%d" %(alpha))
                    for beta in range(signal_n_Min, signal_n_Max+1):		# For-loop for beta
                        print("\t\tsignal_n=%d" %(beta), end="")	# No newline
                        returnRate=computeReturnRate(adjClose, windowSize, alpha, beta, ma_p, band)		# Start the whole run with the given parameters
                        print(" ==> returnRate=%f " %(returnRate))
                        if returnRate > returnRateBest:		# Keep the best parameters
                            windowSizeBest=windowSize
                            alphaBest=alpha
                            betaBest=beta
                            returnRateBest=returnRate
                        short_ema_prev = None
                        long_ema_prev = None
                        signal_line_prev = None
                        macd_prev = None
    print("Best settings: windowSize=%d, alpha=%d, beta=%d ==> returnRate=%f" %(windowSizeBest,alphaBest,betaBest,returnRateBest))