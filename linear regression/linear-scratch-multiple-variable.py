import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np 
import math , copy
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_csv("linear regression/archive/Real estate.csv")

columns = ['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']

print(data.head())

X = data[columns].values


y = data['Y house price of unit area'].values.reshape(-1, 1)

print(X)

print("\n","\n",X.shape[0])

b_init = 0.0
w_init = np.array([ 0.0, 0.0, 0.0, ])

def basic_plot():
    n = len(columns)
    #the graphs for the initial details and all of the stuff for it that there is .
    fig, axes = plt.subplots(
        nrows=len(columns),
        ncols=1,
        figsize=(8, 3.5 * n),
        sharex=False
    )

    for i, col_name in enumerate(columns): 
        axes[i].scatter(data[col_name], data['Y house price of unit area'])
        axes[i].set_xlabel(col_name)
        axes[i].set_ylabel("Price")

    plt.subplots_adjust(hspace=4) 
    plt.tight_layout()
    plt.show()


# #changing this for the mulitple regeression

def cost_function(X,y,w,b):
    cost=0.0
    m=X.shape[0]

    for i in range(m):
        tempc=np.dot(X[i],w) +b
        cost+= ((y[i]-tempc)**2).item()

    total_cost=cost/(2*m)

    return total_cost



    # for i in range(m):

    


# def cost_function(X,y,w,b):


#     m = X.shape[0]
#     cost=0.0
#     for i in range(m):
#         f_wb_i=np.dot(X[i],w)
#         cost=cost+(f_wb_i-y[i])**2
#     cost=cost/(2*m)
#     return cost    
    

def compute_gradient(X,y,w,b):

    m,n = X.shape           
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] =( dj_dw[j] + err * X[i, j] ).item()   
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw


        

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):   
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history



initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X, y, initial_w, initial_b,
                                                    cost_function, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final.item():0.2f},{w_final}")

m,_ = X.shape
# ----- PREDICTIONS -----

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the actual data
ax.scatter(X[:, 0], X[:, 1], y, c='blue', marker='o', alpha=0.6)

# Create grid for the regression plane
x_surf, y_surf = np.meshgrid(
    np.linspace(X[:, 0].min(), X[:, 0].max(), 20),
    np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
)

# Compute predicted z values on the grid
z_surf = (w_final[0] * x_surf) + (w_final[1] * y_surf) + b_final

# Draw regression plane in red
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5)

ax.set_xlabel("X2 House Age")
ax.set_ylabel("X3 Distance to MRT")
ax.set_zlabel("House Price")
ax.set_title("3D Regression Plane (Red) and Data Points (Blue)")

plt.show()




# #chagning this too for the multiple variable

# def gradient_descent(X,y,w,b):
#     m,n=X.shape()

#     dj_dw=np.zeros((n,))
#     dj_db=0

#     for i in range(m):



#     # n = len(points)

#     # for i in range(len(points)):
#     #     x = points.iloc[i]['X2 house age']
#     #     y = points.iloc[i]['Y house price of unit area']

#     #     m_gradient += -(2/n)*x*(y - (m_now*x + b_now))
#     #     b_gradient += -(2/n)*(y - (m_now*x + b_now))

#     # m = m_now - m_gradient * l
#     # b = b_now - b_gradient * l

#     # return m, b

# m = 0
# b = 0
# l = 0.000001
# epochs = 5000



# def gradient_descent_for_multiple_features():
#     pass







# print("Training started...")

# for i in range(epochs):
#     m, b = gradient_descent(m, b, l, data)

    
#     if i % 500 == 0:
#         print(f"Processing... Epoch {i}/{epochs}")

# print("Training done!")
# print("Final values -> m:", m, "b:", b)

# plt.scatter(data['X2 house age'], data['Y house price of unit area'], color='black')
# plt.plot(list(range(0, 120)), [m*x + b for x in range(0, 120)], color='red')
# plt.show()
# input("Press Enter to close...")