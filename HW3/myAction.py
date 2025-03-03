import numpy as np
import pandas as pd
# A simple greedy approach
def myActionSimple(priceMat, transFeeRate):
    # Explanation of my approach:
	# 1. Technical indicator used: Watch next day price
	# 2. if next day price > today price + transFee ==> buy
    #       * buy the best stock
	#    if next day price < today price + transFee ==> sell
    #       * sell if you are holding stock
    # 3. You should sell before buy to get cash each day
    # default
    cash = 1000
    hold = 0
    # user definition
    nextDay = 1
    dataLen, stockCount = priceMat.shape  # day size & stock count   
    stockHolding = np.zeros((dataLen,stockCount))  # Mat of stock holdings
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    
    for day in range( 0, dataLen-nextDay ) :
        dayPrices = priceMat[day]  # Today price of each stock
        nextDayPrices = priceMat[ day + nextDay ]  # Next day price of each stock
        
        if day > 0:
            stockHolding[day] = stockHolding[day-1]  # The stock holding from the previous action day
        
        buyStock = -1  # which stock should buy. No action when is -1
        buyPrice = 0  # use how much cash to buy
        sellStock = []  # which stock should sell. No action when is null
        sellPrice = []  # get how much cash from sell
        bestPriceDiff = 0  # difference in today price & next day price of "buy" stock
        stockCurrentPrice = 0  # The current price of "buy" stock
        
        # Check next day price to "sell"
        for stock in range(stockCount) :
            todayPrice = dayPrices[stock]  # Today price
            nextDayPrice = nextDayPrices[stock]  # Next day price
            holding = stockHolding[day][stock]  # how much stock you are holding
            
            if holding > 0 :  # "sell" only when you have stock holding
                if nextDayPrice < todayPrice*(1+transFeeRate) :  # next day price < today price, should "sell"
                    sellStock.append(stock)
                    # "Sell"
                    sellPrice.append(holding * todayPrice)
                    cash = holding * todayPrice*(1-transFeeRate) # Sell stock to have cash
                    stockHolding[day][sellStock] = 0
        
        # Check next day price to "buy"
        if cash > 0 :  # "buy" only when you have cash
            for stock in range(stockCount) :
                todayPrice = dayPrices[stock]  # Today price
                nextDayPrice = nextDayPrices[stock]  # Next day price
                
                if nextDayPrice > todayPrice*(1+transFeeRate) :  # next day price > today price, should "buy"
                    diff = nextDayPrice - todayPrice*(1+transFeeRate)
                    if diff > bestPriceDiff :  # this stock is better
                        bestPriceDiff = diff
                        buyStock = stock
                        stockCurrentPrice = todayPrice
            # "Buy" the best stock
            if buyStock >= 0 :
                buyPrice = cash
                stockHolding[day][buyStock] = cash*(1-transFeeRate) / stockCurrentPrice # Buy stock using cash
                cash = 0
                
        # Save your action this day
        if buyStock >= 0 or len(sellStock) > 0 :
            action = []
            if len(sellStock) > 0 :
                for i in range( len(sellStock) ) :
                    action = [day, sellStock[i], -1, sellPrice[i]]
                    actionMat.append( action )
            if buyStock >= 0 :
                action = [day, -1, buyStock, buyPrice]
                actionMat.append( action )
    return actionMat

# A DP-based approach to obtain the optimal return
def myAction01(priceMat, transFeeRate):
    total_len = len(priceMat)
    df_flow = [[0] * 8]
    df_route = [[0] * 8]
    for i in range(8):
        if i % 2 != 0:  # 竒
            df_flow[0][i] = 1000
        else:
            df_flow[0][i] = 1000 * (1 - transFeeRate) / priceMat[0][i // 2]

    def construct_max(flow, type, i, j):
        price = priceMat[i][j // 2]

        if type == 's':
            array_compare = np.zeros(8)
            for index in range(4):
                if (index == j // 2):
                    array_compare[index] = flow[i - 1][index * 2] * priceMat[i][index] / price
                else:
                    array_compare[index] = flow[i - 1][index * 2] * priceMat[i][index] * (1 - transFeeRate) ** 2 / price
            array_compare[4] = flow[i - 1][1] * (1 - transFeeRate) / price
            array_compare[5] = flow[i - 1][3] * (1 - transFeeRate) / price
            array_compare[6] = flow[i - 1][5] * (1 - transFeeRate) / price
            array_compare[7] = flow[i - 1][7] * (1 - transFeeRate) / price
            max_value = np.max(array_compare)
            max_index = np.argmax(array_compare)
            if max_index < 4:
                max_index *= 2
            else:
                max_index = (max_index % 4) * 2 + 1
        else:
            array_compare = np.zeros(2)
            array_compare[0] = flow[i - 1][j - 1] * price * (1 - transFeeRate)  # 注意
            array_compare[1] = flow[i - 1][j]
            max_value = np.max(array_compare)
            if np.argmax(array_compare) == 0:
                max_index = j - 1
            else:
                max_index = j
        return max_value, max_index

    def construct_mat(flow, route):
        column_991 = flow[total_len - 1]
        column_991 = np.array(column_991)
        for i in range(0, 8, 2):
            column_991[i] *= priceMat[total_len - 1][i // 2] * (1 - transFeeRate)

        max_index = np.argmax(column_991)
        row = max_index
        day = total_len - 1
        point = route[day][row]
        actionMat = []
        mat = [0.0] * 4  # here

        if day == total_len - 1 and point % 2 == 0:
            mat[0] = day
            mat[1] = point // 2
            mat[2] = -1
            mat[3] = flow[day][point] * priceMat[total_len - 1][point // 2]
            actionMat.insert(0, mat)
        c_holding = 0
        while point != -1 and day - 1 >= 0:
            mat = [0.0] * 4  # here

            while day - 1 >= 0 and route[day - 1][point] == point:
                point = route[day - 1][point]
                day -= 1

            row = point
            point = route[day - 1][point]
            if day - 1 > 0:
                mat[0] = day - 1
                if point % 2 == 0 and row % 2 == 0:  # s to s
                    mat[1] = point // 2
                    mat[2] = row // 2
                    mat[3] = flow[day - 1][point] * priceMat[day - 1][point // 2]
                    row = point
                    point = route[day - 1][row]
                    day -= 1
                elif point % 2 == 0 and row % 2 != 0:  # s to c
                    mat[1] = point // 2
                    mat[2] = -1
                    mat[3] = flow[day - 1][point] * priceMat[day - 1][point // 2]
                    row = point
                    point = route[day - 1][row]
                    if (len(actionMat) != 0):
                        c_holding += (actionMat[0][0]) - (day - 1)
                    day -= 1
                elif point % 2 != 0 and row % 2 == 0:  # c to s
                    mat[1] = -1
                    mat[2] = row // 2
                    mat[3] = flow[day - 1][point]
                    row = point
                    point = route[day - 1][row]
                    day -= 1

                actionMat.insert(0, mat)
        return actionMat, c_holding

    for i in range(1, total_len):
        new_day = [0] * 8
        source = [-2] * 8
        for j in range(8):
            if j % 2 != 0:  # 竒
                new_day[j], source[j] = construct_max(df_flow, 'c', i, j)
            else:
                new_day[j], source[j] = construct_max(df_flow, 's', i, j)
        # print(new_day)
        df_flow.append(new_day)
        df_route.append(source)
    # print(df_flow)
    actionMat, c_holding = construct_mat(df_flow, df_route)
    return actionMat

# An approach that allow non-consecutive K days to hold all cash without any stocks

def myAction02_find_chold(priceMat, transFeeRate):
    total_len = len(priceMat)
    df_flow = [[0] * 8]
    df_route = [[0] * 8]
    for i in range(8):
        if i % 2 != 0:  # 竒
            df_flow[0][i] = 1000
        else:
            df_flow[0][i] = 1000 * (1 - transFeeRate) / priceMat[0][i // 2]

    def construct_max(flow, type, i, j):
        price = priceMat[i][j // 2]

        if type == 's':
            array_compare = np.zeros(8)
            for index in range(4):
                if (index == j // 2):
                    array_compare[index] = flow[i - 1][index * 2] * priceMat[i][index] / price
                else:
                    array_compare[index] = flow[i - 1][index * 2] * priceMat[i][index] * (1 - transFeeRate) ** 2 / price
            array_compare[4] = flow[i - 1][1] * (1 - transFeeRate) / price
            array_compare[5] = flow[i - 1][3] * (1 - transFeeRate) / price
            array_compare[6] = flow[i - 1][5] * (1 - transFeeRate) / price
            array_compare[7] = flow[i - 1][7] * (1 - transFeeRate) / price
            max_value = np.max(array_compare)
            max_index = np.argmax(array_compare)
            if max_index < 4:
                max_index *= 2
            else:
                max_index = (max_index % 4) * 2 + 1
        else:
            array_compare = np.zeros(2)
            array_compare[0] = flow[i - 1][j - 1] * price * (1 - transFeeRate)  # 注意
            array_compare[1] = flow[i - 1][j]
            max_value = np.max(array_compare)
            if np.argmax(array_compare) == 0:
                max_index = j - 1
            else:
                max_index = j
        return max_value, max_index

    def construct_mat(flow, route):
        column_991 = flow[total_len - 1]
        column_991 = np.array(column_991)
        for i in range(0, 8, 2):
            column_991[i] *= priceMat[total_len - 1][i // 2] * (1 - transFeeRate)

        max_index = np.argmax(column_991)
        row = max_index
        day = total_len - 1
        point = route[day][row]
        actionMat = []
        mat = [0.0] * 4  # here

        if day == total_len - 1 and point % 2 == 0:
            mat[0] = day
            mat[1] = point // 2
            mat[2] = -1
            mat[3] = flow[day][point] * priceMat[total_len - 1][point // 2]
            actionMat.insert(0, mat)
        c_holding = 0
        while point != -1 and day - 1 >= 0:
            mat = [0.0] * 4  # here

            while day - 1 >= 0 and route[day - 1][point] == point:
                point = route[day - 1][point]
                day -= 1

            row = point
            point = route[day - 1][point]
            if day - 1 > 0:
                mat[0] = day - 1
                if point % 2 == 0 and row % 2 == 0:  # s to s
                    mat[1] = point // 2
                    mat[2] = row // 2
                    mat[3] = flow[day - 1][point] * priceMat[day - 1][point // 2]
                    row = point
                    point = route[day - 1][row]
                    day -= 1
                elif point % 2 == 0 and row % 2 != 0:  # s to c
                    mat[1] = point // 2
                    mat[2] = -1
                    mat[3] = flow[day - 1][point] * priceMat[day - 1][point // 2]
                    row = point
                    point = route[day - 1][row]
                    if (len(actionMat) != 0):
                        c_holding += (actionMat[0][0]) - (day - 1)
                    day -= 1
                elif point % 2 != 0 and row % 2 == 0:  # c to s
                    mat[1] = -1
                    mat[2] = row // 2
                    mat[3] = flow[day - 1][point]
                    row = point
                    point = route[day - 1][row]
                    day -= 1

                actionMat.insert(0, mat)
        return actionMat, c_holding

    for i in range(1, total_len):
        new_day = [0] * 8
        source = [-2] * 8
        for j in range(8):
            if j % 2 != 0:  # 竒
                new_day[j], source[j] = construct_max(df_flow, 'c', i, j)
            else:
                new_day[j], source[j] = construct_max(df_flow, 's', i, j)
        # print(new_day)
        df_flow.append(new_day)
        df_route.append(source)
    # print(df_flow)
    actionMat, c_holding = construct_mat(df_flow, df_route)
    return actionMat, c_holding

def myAction02(priceMat, transFeeRate, K):
    cut_pm = priceMat[:len(priceMat)-K]
    actionMat1, c_holding = myAction02_find_chold(cut_pm, transFeeRate)
    day_to_hold = K - c_holding
    actionMat = myAction03(priceMat, transFeeRate, day_to_hold)

    return actionMat

# An approach that allow consecutive K days to hold all cash without any stocks    
def myAction03(priceMat, transFeeRate, K):
    total_len = len(priceMat)
    # df_flow = [[0] * 8]
    # df_route = [[0] * 8]
    # for i in range(8):
    #     if i % 2 != 0:  # 竒
    #         df_flow[0][i] = 1000
    #     else:
    #         df_flow[0][i] = 1000 * (1 - transFeeRate) / priceMat[0][i // 2]


    def construct_max(flow, type, i, j):
        price = priceMat[i][j // 2]

        if type == 's':
            array_compare = np.zeros(8)
            for index in range(4):
                if (index == j // 2):
                    array_compare[index] = flow[i - 1][index * 2] * priceMat[i][index] / price
                else:
                    array_compare[index] = flow[i - 1][index * 2] * priceMat[i][index] * (1 - transFeeRate) ** 2 / price
            array_compare[4] = flow[i - 1][1] * (1 - transFeeRate) / price
            array_compare[5] = flow[i - 1][3] * (1 - transFeeRate) / price
            array_compare[6] = flow[i - 1][5] * (1 - transFeeRate) / price
            array_compare[7] = flow[i - 1][7] * (1 - transFeeRate) / price
            max_value = np.max(array_compare)
            max_index = np.argmax(array_compare)
            if max_index < 4:
                max_index *= 2
            else:
                max_index = (max_index % 4) * 2 + 1
        else:
            array_compare = np.zeros(2)
            array_compare[0] = flow[i - 1][j - 1] * price * (1 - transFeeRate)  # 注意
            array_compare[1] = flow[i - 1][j]
            max_value = np.max(array_compare)
            if np.argmax(array_compare) == 0:
                max_index = j - 1
            else:
                max_index = j
        return max_value, max_index

    def construct_mat(flow, route, tot_len):
        column_991 = flow[tot_len - 1]
        column_991 = np.array(column_991)
        for i in range(0, 8, 2):
            column_991[i] *= priceMat[tot_len - 1][i // 2] * (1 - transFeeRate)

        max_index = np.argmax(column_991)
        row = max_index
        day = tot_len - 1
        point = route[day][row]
        actionMat = []
        mat = [0.0] * 4  # here

        if day == total_len - 1 and point % 2 == 0:
            mat[0] = day
            mat[1] = point // 2
            mat[2] = -1
            mat[3] = flow[day][point] * priceMat[tot_len - 1][point // 2]
            actionMat.insert(0, mat)
        while point != -1 and day - 1 >= 0:
            mat = [0.0] * 4  # here

            while day - 1 >= 0 and route[day - 1][point] == point:
                point = route[day - 1][point]
                day -= 1

            row = point
            point = route[day - 1][point]
            if day - 1 > 0:
                mat[0] = day - 1
                if point % 2 == 0 and row % 2 == 0:  # s to s
                    mat[1] = point // 2
                    mat[2] = row // 2
                    mat[3] = flow[day - 1][point] * priceMat[day - 1][point // 2]
                    row = point
                    point = route[day - 1][row]
                    day -= 1
                elif point % 2 == 0 and row % 2 != 0:  # s to c
                    mat[1] = point // 2
                    mat[2] = -1
                    mat[3] = flow[day - 1][point] * priceMat[day - 1][point // 2]
                    row = point
                    point = route[day - 1][row]
                    day -= 1
                elif point % 2 != 0 and row % 2 == 0:  # c to s
                    mat[1] = -1
                    mat[2] = row // 2
                    mat[3] = flow[day - 1][point]
                    row = point
                    point = route[day - 1][row]
                    day -= 1
                if mat[3] != 0.0:
                    actionMat.insert(0, mat)
        return actionMat

    #max_value, start_day = find_start_day(priceMat, K)
    # print(start_day)
    # print(max_value)
    start_days = [0] * 200
    for test_day in range(total_len - K- 200+1, total_len - K +1):
        start_days[test_day - (total_len - K - 200+1)] = test_day

    max_r = 0
    for start_day in start_days:

        df_flow = [[0] * 8]
        df_route = [[0] * 8]
        for i in range(8):
            if i % 2 != 0:  # 竒
                df_flow[0][i] = 1000
            else:
                df_flow[0][i] = 1000 * (1 - transFeeRate) / priceMat[0][i // 2]

        for i in range(1, start_day + 1):
            new_day = [0] * 8
            source = [-2] * 8
            for j in range(8):
                if j % 2 != 0:  # 竒
                    new_day[j], source[j] = construct_max(df_flow, 'c', i, j)
                else:
                    new_day[j], source[j] = construct_max(df_flow, 's', i, j)
            # print(new_day)
            df_flow.append(new_day)
            df_route.append(source)
        # print(start_day)
        actionMat = construct_mat(df_flow, df_route, start_day + 1)
        column_991 = df_flow[start_day]
        column_991 = np.array(column_991)
        for i in range(0, 8, 2):
            column_991[i] *= priceMat[start_day][i // 2] * (1 - transFeeRate)
        capital_holding = np.max(column_991)
        # actionMat = actionMat[:-1]
        # print(actionMat)
        if actionMat[-1][2] != -1:  # corner case
            max_index = np.argmax(column_991)
            point = df_route[start_day][max_index]
            mat = [0.0] * 4
            mat[0] = start_day
            mat[1] = actionMat[-1][2]
            mat[2] = -1
            mat[3] = df_flow[start_day][point] * priceMat[start_day][point // 2]
            actionMat.append(mat)

        if start_day == total_len - K:
            if actionMat[-1][3] >= max_r:

                return actionMat
            break

        for i in range(start_day + 1, start_day + K + 1):
            df_flow.append([0.0] * 8)
            df_route.append([0.0] * 8)

        df_route[start_day + K] = [-1] * 8
        for i in range(8):
            if i % 2 != 0:  # 竒
                df_flow[start_day + K ][i] = capital_holding
            else:
                df_flow[start_day + K ][i] = capital_holding * (1 - transFeeRate) / priceMat[start_day + K][i // 2]

        for i in range(start_day + K + 1, total_len):
            new_day = [0] * 8
            source = [-2] * 8
            for j in range(8):
                if j % 2 != 0:  # 竒
                    new_day[j], source[j] = construct_max(df_flow, 'c', i, j)
                else:
                    new_day[j], source[j] = construct_max(df_flow, 's', i, j)
            # print(new_day)
            df_flow.append(new_day)
            df_route.append(source)

        actionMat += construct_mat(df_flow, df_route, total_len)

        if actionMat[-1][3] >= max_r:
            true_actionMat = actionMat
            max_r = actionMat[-1][3]
    actionMat = true_actionMat

    return actionMat