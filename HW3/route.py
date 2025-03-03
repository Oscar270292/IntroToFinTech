import numpy as np
import pandas as pd
transFeeRate = 0.01

file_path = r'C:\Users\GL66\PycharmProjects\IntroToFinTech\HW3\priceMat0992.txt'
df = pd.read_csv(file_path, delimiter=' ', header=None)
total_len = 692
df_flow = pd.DataFrame(0.0, index=range(8), columns=range(1))
df_route = pd.DataFrame(-1, index=range(8), columns=[0])
for i in range(8):
    if i % 2 != 0: #竒
        df_flow.at[i, 0] = 1000
    else:
        df_flow.at[i, 0] = 1000* (1 - transFeeRate)/df.iloc[0][i/2]


def construct_max(flow, type, i, j):
    price = df.at[i, j//2]

    if type == 's':
        array_compare = np.zeros(8)
        for index in range(4):
            if(index == j//2):
                array_compare[index] = flow.at[index * 2, i - 1] * df.at[i, index] / price
            else:
                array_compare[index] = flow.at[index*2, i-1] * df.at[i, index]* (1 - transFeeRate)**2 / price
        array_compare[4] = flow.at[1, i - 1] * (1 - transFeeRate)/ price
        array_compare[5] = flow.at[3, i - 1] * (1 - transFeeRate) / price
        array_compare[6] = flow.at[5, i - 1] * (1 - transFeeRate) / price
        array_compare[7] = flow.at[7, i - 1] * (1 - transFeeRate) / price
        max_value = np.max(array_compare)
        max_index = np.argmax(array_compare)
        if max_index < 4:
            max_index *= 2
        else:
            max_index = (max_index % 4)*2 + 1
    else:
        array_compare = np.zeros(2)
        array_compare[0] = flow.at[j-1, i-1] * price * (1 - transFeeRate) #注意
        array_compare[1] = flow.at[j, i-1]
        max_value = np.max(array_compare)
        if np.argmax(array_compare) == 0:
            max_index = j-1
        else:
            max_index = j
    return max_value, max_index

def construct_mat(flow, route):
    column_991 = flow.iloc[:, total_len-1]
    for i in range(0, 8, 2):
        column_991[i] *= df.at[total_len-1, i//2] * (1 - transFeeRate)
    print(column_991)

    max_index = np.argmax(column_991)
    row = max_index
    day = total_len-1
    point = route.at[row, day]
    actionMat = []
    max_route = []
    mat = [0.0] * 4  # here

    if day == total_len-1 and point % 2 == 0:
        mat[0] = day
        mat[1] = point // 2
        mat[2] = -1
        mat[3] = flow.at[point, day] / (1 - transFeeRate)
    if mat[3] != 0.0:
        actionMat.insert(0, mat)
    max_route.insert(0, row)
    while point != -1 and day-1 >= 0:
        mat = [0.0] * 4 #here

        while day-1 >= 0 and route.at[point, day-1] == point:
            row = point
            point = route.at[point, day - 1]
            max_route.insert(0, row)
            day -= 1

        row = point
        point = route.at[point, day - 1]
        if day -1 > 0:
            mat[0] = day - 1
            if point % 2 == 0 and row % 2 == 0: #s to s
                mat[1] = point // 2
                mat[2] = row // 2
                mat[3] = flow.at[point, day-1] * df.at[day-1, point//2]
            elif point % 2 == 0 and row % 2 != 0: #s to c
                mat[1] = point // 2
                mat[2] = -1
                mat[3] = flow.at[point, day-1] * df.at[day-1, point//2]
            elif point % 2 != 0 and row % 2 == 0: #c to s
                mat[1] = -1
                mat[2] = row // 2
                mat[3] = flow.at[point, day-1]
            row = point
            point = route.at[row, day - 1]
            max_route.insert(0, row)
            day -= 1
            if mat[3] != 0.0:
                actionMat.insert(0, mat)
    return actionMat, max_route


for i in range(1, total_len):
    new_day = pd.DataFrame(0.0, index=range(8), columns=[i])
    source = pd.DataFrame(-2, index=range(8), columns=[i])
    for j in range(8):
        if j % 2 != 0:  # 竒
            new_day.iloc[j, 0], source.iloc[j, 0] = construct_max(df_flow, 'c', i, j)
        else:
            new_day.iloc[j, 0], source.iloc[j, 0] = construct_max(df_flow, 's', i, j)
    #print(new_day)
    df_flow = pd.concat([df_flow, new_day], axis=1)
    df_route = pd.concat([df_route, source], axis=1)
#df_route.to_csv('df_route.csv', index=False)  # index=False 不輸出索引
actionMat, max_route= construct_mat(df_flow, df_route)
max_route.append(6)
print(actionMat)

print(max_route)
print(len(max_route))
