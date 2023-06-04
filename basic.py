import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Read the CSV file
data = pd.read_csv("simple_6_flash_led_data.log",delimiter=',')  #Timestamp, Amps

# # Check if 'Amps' column exists
# if ' Amps' not in data.columns:
#     raise ValueError("The 'Amps' column does not exist in the CSV file.")

# # Create the figure and axis
# fig, ax = plt.subplots()

# # Initialize the line plot
# line, = ax.plot([], [], lw=2)

# # Set the axis labels
# ax.set_xlabel('Timestamp')
# ax.set_ylabel(' Amps')

# # Initialization function to clear the plot
# def init():
#     line.set_data([], [])
#     return line,

# # Function to update the plot with each animation frame
# def update(frame):
#     # Get the data for the current frame
#     x = data['Timestamp'][:frame]
#     y = data[' Amps'][:frame]
    
#     # Update the line plot
#     line.set_data(x, y)
    
#     # Set the x-axis limits to accommodate all data points
#     ax.set_xlim(data['Timestamp'].min(), data['Timestamp'].max())
    
#     return line,

# # Create the animation with 100ms delay between frames
# animation = FuncAnimation(fig, update, frames=100, interval=100, blit=True)

# # Show the plot
# plt.show()
# Create the figure and axis
fig, ax = plt.subplots()

# Plot the line
ax.plot(data['Timestamp'], data[' Amps'], lw=2)

# Set the axis labels
ax.set_xlabel('Timestamp')
ax.set_ylabel('Amps')

# Show the plot
plt.show()