import numpy as np
import pandas as pd
transFeeRate = 0.01

file_path = r'C:\Users\GL66\PycharmProjects\IntroToFinTech\HW3\priceMat0992.txt'
df = pd.read_csv(file_path, delimiter=' ', header=None)
total_len = len(df[0])
df_flow =[[0]*8]
df_route = [[0]*8]
for i in range(8):
    if i % 2 != 0:  # 竒
        df_flow[0][i] = 1000
    else:
        df_flow[0][i] = 1000 * (1 - transFeeRate) / df.iloc[0][i / 2]

def construct_max(flow, type, i, j):
    price = df.at[i, j//2]

    if type == 's':
        array_compare = np.zeros(8)
        for index in range(4):
            if(index == j//2):
                array_compare[index] = flow[i-1][index*2] * df.at[i, index] / price
            else:
                array_compare[index] = flow[i-1][index*2] * df.at[i, index]* (1 - transFeeRate)**2 / price
        array_compare[4] = flow[i-1][1] * (1 - transFeeRate)/ price
        array_compare[5] = flow[i-1][3] * (1 - transFeeRate) / price
        array_compare[6] = flow[i-1][5] * (1 - transFeeRate) / price
        array_compare[7] = flow[i-1][7] * (1 - transFeeRate) / price
        max_value = np.max(array_compare)
        max_index = np.argmax(array_compare)
        if max_index < 4:
            max_index *= 2
        else:
            max_index = (max_index % 4)*2 + 1
    else:
        array_compare = np.zeros(2)
        array_compare[0] = flow[i-1][j-1] * price * (1 - transFeeRate) #注意
        array_compare[1] = flow[i-1][j]
        max_value = np.max(array_compare)
        if np.argmax(array_compare) == 0:
            max_index = j-1
        else:
            max_index = j
    return max_value, max_index

def construct_mat(flow, route):
    column_991 = flow[total_len-1]
    column_991 = np.array(column_991)
    for i in range(0, 8, 2):
        column_991[i] *= df.at[total_len-1, i//2] * (1 - transFeeRate)
    print(column_991)
    max_index = np.argmax(column_991)
    row = max_index
    day = total_len-1
    point = route[day][row]
    actionMat = []
    mat = [0.0] * 4  # here

    if day == total_len-1 and point % 2 == 0:
        mat[0] = day
        mat[1] = point // 2
        mat[2] = -1
        mat[3] = flow[day][point]* df.at[total_len-1, point//2]
        actionMat.insert(0, mat)
    c_holding = 0
    while point != -1 and day-1 >= 0:
        mat = [0.0] * 4 #here

        while day-1 >= 0 and route[day-1][point] == point:
            point = route[day-1][point]
            day -= 1

        row = point
        point = route[day-1][point]
        if day -1 > 0:
            mat[0] = day - 1
            if point % 2 == 0 and row % 2 == 0: #s to s
                mat[1] = point // 2
                mat[2] = row // 2
                mat[3] = flow[day-1][point] * df.at[day-1, point//2]
                row = point
                point = route[day-1][row]
                day -= 1
            elif point % 2 == 0 and row % 2 != 0: #s to c
                mat[1] = point // 2
                mat[2] = -1
                mat[3] = flow[day-1][point] * df.at[day-1, point//2]
                row = point
                point = route[day-1][row]
                if(len(actionMat) != 0):
                    c_holding += (actionMat[0][0]) - (day - 1)
                day -= 1

            elif point % 2 != 0 and row % 2 == 0: #c to s
                mat[1] = -1
                mat[2] = row // 2
                mat[3] = flow[day-1][point]
                row = point
                point = route[day-1][row]
                day -= 1

            actionMat.insert(0, mat)
    return actionMat, c_holding


for i in range(1, total_len):
    new_day = [0]*8
    source = [-2]*8
    for j in range(8):
        if j % 2 != 0:  # 竒
            new_day[j], source[j] = construct_max(df_flow, 'c', i, j)
        else:
            new_day[j], source[j] = construct_max(df_flow, 's', i, j)
    #print(new_day)
    df_flow.append(new_day)
    df_route.append(source)
#print(df_flow)
actionMat, c_holding = construct_mat(df_flow, df_route)
print(actionMat)
c_holding += (actionMat[0][0]+1)
print(c_holding)

