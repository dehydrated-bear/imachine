import pandas as pd  
import matplotlib.pyplot as plt 


data=pd.read_csv("linear regression/archive/Real estate.csv")

columns=['X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores']

print(data.head())

n = len(columns)



fig, axes = plt.subplots(
    nrows=len(columns),    # one row per feature
    ncols=1,                       # only one column
    figsize=(8, 3.5 * n),   # scale height with number of plots
    sharex= False                  # share the same X axis if you want
)

for i, col_name in enumerate(columns): 
    # Assuming you want to plot each column against a common 'x_axis_column'
     axes[i].scatter(data['Y house price of unit area'], data[col_name]) 
     axes[i].set_title(f' {col_name}') 
     axes[i].set_xlabel(col_name) 
     axes[i].set_ylabel("price")

plt.subplots_adjust(hspace=4) 
plt.tight_layout()

plt.show()