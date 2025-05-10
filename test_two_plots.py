import matplotlib.pyplot as plt
import numpy as np

# Create sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x) * 100  # Scaled differently for demonstration

fig, ax1 = plt.subplots()

# Plot the first line with the left y-axis
ax1.plot(x, y1, 'g-', label='Sin(x)')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Sin(x)', color='g')

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
ax2.plot(x, y2, 'b-', label='Cos(x) * 100')
ax2.set_ylabel('Cos(x) * 100', color='b')

# Add legends for both axes
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title("Dual Y-Axis Plot")
plt.show()