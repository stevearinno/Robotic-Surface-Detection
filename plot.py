import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from load_data import load_data
from plot_by_class import plot_by_class
from calculator import running_average

foldername = 'data'
X, X_test_kaggle, y, groups, lenc = load_data(foldername)

y = lenc.inverse_transform(y)
classes = np.unique(y)


# HOW MANY EXAMPLES
j = 3

# CHOOSE SENSOR
i = 6
# 0 = Orientation.X
# 1 = Orientation.Y, 
# 2 = Orientation.Z, 
# 3 = Orientation.W, 
# 4 = AngularVelocity.X, 
# 5 = AngularVelocity.Y, 
# 6 = AngularVelocity.Z, 
# 7 = LinearAcceleration.X, 
# 8 = LinearAcceleration.Y, 
# 9 = LinearAcceleration.Z

# ALL CLASSES
#['carpet','concrete','fine_concrete','hard_tiles','hard_tiles_large_space','soft_pvc','soft_tiles','tiled','wood']
classes = ['carpet','concrete', 'wood']

plot_by_class(X, y, list(classes), i, j, False)

# N = MATERIAL 
n = 6
testattava = X[n,i]
#print(np.size(X,1))
alkuperainen = X[n,i]

# M = running average amount
m = 5
start = m // 2
end = 128 - (m // 2)
print("tesdasdsad")
print(start)
print(end)

testattava = running_average(testattava, m)
alkuperainen = alkuperainen[start:end]

test0 = np.ravel(testattava)
test1 = np.ravel(alkuperainen)


plt.figure(figsize=(12,5))
plt.title(y[n])
plt.plot(test0, label='Running average')
plt.plot(test1, label='Orginal')

tulos1 = np.abs(testattava - alkuperainen)
maxmin = np.abs(np.max(alkuperainen)-np.min(alkuperainen))


print("total difference to running avg: ", sum(tulos1))
print("average difference to running avg: ", np.average((tulos1)))
print("difference between max & min: ", maxmin)
print("biggest difference to running avg: ", np.max(tulos1))
'''
#CARPET
test0 = np.ravel(X[4,i])
#CONCRETE
test1 = np.ravel(X[12,i])
#FINE CONCRETE
test2 = np.ravel(X[5,i])
#HARD TILES
test3 = np.ravel(X[1,i])
#HARD TILES LARGE SPACE
test4 = np.ravel(X[39,i])
#SOFT PVC
test5 = np.ravel(X[6,i])
#SOFT TILES
test6 = np.ravel(X[0,i])
#TILED
test7 = np.ravel(X[8,i])
#WOOD
test8 = np.ravel(X[27,i])

plt.title("Angular Velocity Z")
plt.plot(test0, label='Carpet')
plt.plot(test1, label='Concrete')
plt.plot(test2, label='Fine Concrete')
plt.plot(test3, label='Hard tiles')
plt.plot(test4, label='Hard tiles large space')
plt.plot(test5, label='Soft pvc')
plt.plot(test6, label='Soft tiles')
plt.plot(test7, label='Tiled')
plt.plot(test8, label='Wood')
'''
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

