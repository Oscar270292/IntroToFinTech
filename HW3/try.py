import numpy as np
import pandas as pd
transFeeRate = 0.01
K = 200

file_path = r'C:\Users\GL66\PycharmProjects\IntroToFinTech\HW3\priceMat0992.txt'
df = pd.read_csv(file_path, delimiter=' ', header=None)
priceMat = df.values
total_len = len(priceMat)
# df_flow =[[0]*8]
# df_route = [[0]*8]
# for i in range(8):
#     if i % 2 != 0:  # 竒
#         df_flow[0][i] = 1000
#     else:
#         df_flow[0][i] = 1000 * (1 - transFeeRate) / priceMat[0][i // 2]

def find_start_day(priceMat, K):
    drop = [0] * (total_len - K-1)
    for i in range(total_len - K-1):
        for j in range(4):
            drop[i] += ((priceMat[i+K+1][j] - priceMat[i][j]) / priceMat[i][j])
    drop = np.array(drop)
    max_value = np.min(drop)
    start_day = np.argmin(drop)

    return max_value, start_day

def construct_max(flow, type, i, j):
    price = priceMat[i][j // 2]

    if type == 's':
        array_compare = np.zeros(8)
        for index in range(4):
            if(index == j//2):
                array_compare[index] = flow[i-1][index*2] * priceMat[i][index] / price
            else:
                array_compare[index] = flow[i-1][index*2] * priceMat[i][index]* (1 - transFeeRate)**2 / price
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
start_days = [0] * 50
for test_day in range(total_len-K-50, total_len-K):
    start_days[test_day - (total_len-K-50)] = test_day

max_r = 0
# for start_day in start_days:

df_flow = [[0] * 8]
df_route = [[0] * 8]
for i in range(8):
    if i % 2 != 0:  # 竒
        df_flow[0][i] = 1000
    else:
        df_flow[0][i] = 1000 * (1 - transFeeRate) / priceMat[0][i // 2]

start_day = 791
for i in range(1, start_day+1):
    new_day = [0] * 8
    source = [-2] * 8
    for j in range(8):
        if j % 2 != 0:  # 竒
            new_day[j], source[j] = construct_max(df_flow, 'c', i, j)
        else:
            new_day[j], source[j] = construct_max(df_flow, 's', i, j)
    #print(new_day)
    df_flow.append(new_day)
    df_route.append(source)
#print(start_day)
actionMat = construct_mat(df_flow, df_route, start_day+1)
column_991 = df_flow[start_day]
column_991 = np.array(column_991)
for i in range(0, 8, 2):
    column_991[i] *= priceMat[start_day][i // 2] * (1 - transFeeRate)
capital_holding = np.max(column_991)
#actionMat = actionMat[:-1]
#print(actionMat)
if actionMat[-1][2] != -1: #corner case
    max_index = np.argmax(column_991)
    point = df_route[start_day][max_index]
    mat = [0.0] * 4
    mat[0] = start_day
    mat[1] = actionMat[-1][2]
    mat[2] = -1
    mat[3] = df_flow[start_day][point]* priceMat[start_day][point // 2]
    actionMat.append(mat)

if start_day == total_len-K-1:
    if actionMat[-1][3] >= max_r:
        true_actionMat = actionMat
        max_r = actionMat[-1][3]
        print("biggest")
        print(true_actionMat)


for i in range(start_day+1, start_day+K+2): #為了多一行來初始化
    df_flow.append([0.0]*8)
    df_route.append([0.0]*8)

df_route[start_day+K+1] = [-1]*8
for i in range(8):
    if i % 2 != 0:  # 竒
        df_flow[start_day+K+1][i] = capital_holding
    else:
        df_flow[start_day+K+1][i] = capital_holding * (1 - transFeeRate) / priceMat[start_day+K+1][i // 2]


for i in range(start_day+K+2, total_len):
    new_day = [0] * 8
    source = [-2] * 8
    for j in range(8):
        if j % 2 != 0:  # 竒
            new_day[j], source[j] = construct_max(df_flow, 'c', i, j)
        else:
            new_day[j], source[j] = construct_max(df_flow, 's', i, j)
    #print(new_day)
    df_flow.append(new_day)
    df_route.append(source)

actionMat += construct_mat(df_flow, df_route, total_len)
#print(true_actionMat)