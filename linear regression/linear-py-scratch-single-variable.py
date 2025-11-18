import pandas as pd  
import matplotlib.pyplot as plt  

data = pd.read_csv("/Users/harshit/Code/machine-ll/imachine/linear regression/archive/Real estate.csv")

columns = ['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']

print(data.head())

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

def cost_function(m,b,points):
    total_cost = 0
    for i in range(len(points)):
        x = points.iloc[i]['X2 house age']
        y = points.iloc[i]['Y house price of unit area']
        total_cost += (y - (m*x + b))**2
    return total_cost / float(len(points))
        
def gradient_descent(m_now, b_now, l, points):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(len(points)):
        x = points.iloc[i]['X2 house age']
        y = points.iloc[i]['Y house price of unit area']

        m_gradient += -(2/n)*x*(y - (m_now*x + b_now))
        b_gradient += -(2/n)*(y - (m_now*x + b_now))

    m = m_now - m_gradient * l
    b = b_now - b_gradient * l

    return m, b

m = 0
b = 0
l = 0.000001
epochs = 5000

def test_ll():
    pass




print("Training started...")

for i in range(epochs):
    m, b = gradient_descent(m, b, l, data)

    
    if i % 500 == 0:
        print(f"Processing... Epoch {i}/{epochs}")

print("Training done!")
print("Final values -> m:", m, "b:", b)

plt.scatter(data['X2 house age'], data['Y house price of unit area'], color='black')
plt.plot(list(range(0, 120)), [m*x + b for x in range(0, 120)], color='red')
plt.show()
input("Press Enter to close...")