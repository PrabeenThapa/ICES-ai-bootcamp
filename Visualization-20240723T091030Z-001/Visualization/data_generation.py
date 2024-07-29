import numpy as np
import pandas as pd

np.random.seed(42)

# 4 features

size = np.random.randint(800, 4000, 100)
bedroom = np.random.randint(1, 6, 100)
age = np.random.randint(0,40, 100)
price = np.random.randint(20000, 70000, 100) + age * -2 + size * 0.5 + bedroom * 50
price /= 1000 # price in k

house_data = pd.DataFrame({'Size': size,
                           'Bedrooms': bedroom,
                           'Age': age,
                           'Price': price})

house_data.to_csv('./house_price.csv', index = False)
print(house_data.head())