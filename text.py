import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,20,100)
print(x)
y=np.sin(x)
print(y)
plt.plot(x,y)
plt.ylabel('y')
plt.xlabel('x')
plt.show()

