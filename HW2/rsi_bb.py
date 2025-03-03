# Declare variables to hold previous EMA and MACD values
short_ema_prev = None
long_ema_prev = None
signal_line_prev = None
macd_prev = None


def myStrategy(pastPriceVec, currentPrice):
    import numpy as np
    global short_ema_prev, long_ema_prev, signal_line_prev, macd_prev  # Use global variables to store previous values

    short_n = 8  # Short-term period for EMA
    long_n = 47  # Long-term period for EMA
    signal_n = 7  # Period for the signal line (9-day EMA of MACD)
    windowSize = 26
    ob = 68
    os = 32
    action = 0
    dataLen = len(pastPriceVec)  # Length of the pastPriceVec (all previous prices up to the current day)
    if dataLen == 0:
        short_ema_prev = None
        long_ema_prev = None
        signal_line_prev = None
        macd_prev = None
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
            rsi = 100 - (100/(1+rs))

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

    # Determine action
    if rsi is not None:
        if rsi <= os and macd < signal_line:  #buy
            action = 1
        elif rsi >= ob and macd > signal_line:  #sell
            action = -1

    return action