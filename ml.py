import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd

df = pd.read_csv("weatherHistory.csv")
df1 = df[["Humidity","Wind Speed (km/h)","Wind Bearing (degrees)","Visibility (km)","Loud Cover","Pressure (millibars)"]]
df1=df1.dropna(axis=0)
df2 = df[["Temperature (C)"]]
df2=df2.dropna(axis=0)
x_train = df1.to_numpy()
y_train = df2.to_numpy()

b_init = 0
w_init = np.array([0 for i in range(x_train.shape[1])])

def compute_cost(x,y,w,b):
    cost = 0
    m = x.shape[0]
    for i in range(m):
        f_wb_i = np.dot(x[i],w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost/(2 * m)
    return cost

def compute_gradient(x,y,w,b):
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        err = (np.dot(x[i],w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err*x[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw, dj_db

def gradient_descent(x,y,w_in,b_in,alpha,num_iters,compute_cost,compute_gradient):
    J_hist = []
    w_hist = []
    w = w_in
    b = b_in
    for i in range(num_iters):
        dj_dw , dj_db = compute_gradient(x,y,w,b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        J_hist.append(compute_cost(x,y,w,b))
        w_hist.append(w[0])
        if(i<=40 or i>=460):
          print("Number of iterations: {}, Value of w: {},  Value of b: {}".format(i,w,b))
    return w , b , J_hist, w_hist

initial_w = np.zeros_like(w_init)
initial_b = 0
iterations = 500
alpha = 5.0e-08
J_hist = []
w_final, b_final, J_hist, w_hist = gradient_descent(x_train,y_train,initial_w,initial_b,alpha,iterations,compute_cost,compute_gradient)
print("Optimal values of w and b are: {} and {}".format(w_final,b_final))

num_iters = [i for i in range(iterations)]
plt.plot(num_iters[-28:],J_hist[-28:])
plt.show()
print(num_iters)

plt.plot(w_hist,J_hist)
plt.show()

a = []
print("Enter Humidity: ", end=" ")
a.append(float(input(" ")))
print("Enter Wind Speed (in kmph): ", end=" ")
a.append(float(input(" ")))
print("Enter Wind Bearing (in degrees): ", end=" ")
a.append(float(input(" ")))
print("Enter Visibility (in km): ", end=" ")
a.append(float(input(" ")))
print("Enter Pressure (in millibars) ", end=" ")
a.append(float(input(" ")))
x = np.array(a)
price = np.dot(x,w_final) + b_final
print("The apparent temperature should be {}".format(price[0]))
