import numpy as np
#
data = [[1, 2, 3],
        [2, 4, 6],
        [1, 1, 1]]
X = data
# def show_feature_corr(X):
for i in range(0, 3):
    for j in range(0, 3):
        z = X[i][j]
        print("%d %d %-2d" % (i, j, z),)
            # x = X.rows[i]
            # y = X.columns[j]
            # z = X[i][j]
            #return(z)
# if __name__ == '__main__':
#     print(show_feature_corr(X=Xy))