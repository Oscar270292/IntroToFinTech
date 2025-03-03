# Declare variables to hold previous EMA and MACD values
short_ema_prev = None
long_ema_prev = None
signal_line_prev = None
macd_prev = None

def myStrategy(pastPriceVec, currentPrice):
    import numpy as np
    global short_ema_prev, long_ema_prev, signal_line_prev, macd_prev  # Use global variables to store previous values

    ma_peroid = 14
    band_w = 2
    short_n = 8  # Short-term period for EMA
    long_n = 47  # Long-term period for EMA
    signal_n = 7  # Period for the signal line (9-day EMA of MACD)
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
