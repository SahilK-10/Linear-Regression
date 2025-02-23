import numpy as np
import matplotlib.pyplot as plt  # For visualization

# Step 1: Define Sample Data (House Size vs. Price)
house_size = np.array([600, 800, 1000, 1200, 1500]
                      )  # House size in square feet
price = np.array([150000, 200000, 250000, 300000, 375000]
                 )  # Price in thousands

# Step 2: Normalize Data (Centering around mean)
hs_mean, p_mean = np.mean(house_size), np.mean(price)
hs_normalized, p_normalized = house_size - hs_mean, price - p_mean

# Step 3: Calculate Slope (m) and Intercept (c) using Least Squares Method
m = np.sum(hs_normalized * p_normalized) / np.sum(hs_normalized**2)
c = p_mean - m * hs_mean

# Step 4: Make Predictions
p_pred = m * house_size + c

# Step 5: Visualize the Regression Line
plt.scatter(house_size, price, color='blue',
            label="Actual Data")  # Plot original data
plt.plot(house_size, p_pred, color='red',
         label="Predicted Line")  # Regression line
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price($)")
plt.legend()
plt.title("Simple Linear Regression (House Price Prediction)")
plt.show()

# Step 6: Print Model Parameters
print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")
print("Equation of Line: y =", round(m, 2), "* x +", round(c, 2))

# Step 7: User Input for Predictions
choice = input(
    "Do you want to \n(1) Predict price for a plot size or \n(2) Suggest plot size for a budget? \nEnter 1 or 2: ")

if choice == "1":
    user_size = float(input("Enter the house size (in sq ft): "))
    predicted_price = m * user_size + c
    print(f"Estimated Price: ${predicted_price:.2f}")

elif choice == "2":
    user_budget = float(input("Enter your budget (in $): "))
    suggested_size = (user_budget - c) / m
    print(f"Suggested House Size: {suggested_size:.2f} sq ft")

else:
    print("Invalid choice. Please enter 1 or 2.")
