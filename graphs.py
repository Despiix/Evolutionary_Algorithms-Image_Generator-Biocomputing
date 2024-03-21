import pandas as pd
import matplotlib.pyplot as plt

# Adjust the path to where your .txt log file is stored
file_path = 'Generation_Log(3c).txt'

# Read the data from the .txt file, specifying the separator as comma
# Assuming the structure you provided, with relevant data in specific columns
log_data = pd.read_csv(file_path, header=None, sep=',', usecols=[1, 3], names=['Generation', 'MedianFitness'])

# Convert Generation to numeric values (if necessary)
log_data['Generation'] = pd.to_numeric(log_data['Generation'], errors='coerce')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(log_data['Generation'], log_data['MedianFitness'], marker='o', linestyle='-', color='blue')
plt.title('Fitness Progression Over Generations')
plt.xlabel('Generation')
plt.ylabel('Median Fitness')
plt.grid(True)
plt.show()