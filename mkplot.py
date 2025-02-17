import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file into a Pandas DataFrame
try:
    df = pd.read_csv('results.csv')
except FileNotFoundError:
    print("Error: results.csv not found. Make sure the file is in the same directory as the script.")
    exit()

# Check if the DataFrame is empty
if df.empty:
    print("Error: results.csv is empty.")
    exit()

# Plot CPU time
plt.plot(df['N'], df['CPU Time (us)'], label='CPU Time (us)')

# Plot GPU time
plt.plot(df['N'], df['GPU Time (ns)'], label='GPU Time (ns)')

# Set the axis labels and title
plt.xlabel('N')
plt.ylabel('Time (us and ns)')
plt.title('CPU and GPU Execution Times')

# Add a legend
plt.legend()

# Set logarithmic scale for x and y axis
plt.xscale('log', base=2)
plt.yscale('log')

# Add grid
plt.grid(True)

# Show the plot
plt.show()
