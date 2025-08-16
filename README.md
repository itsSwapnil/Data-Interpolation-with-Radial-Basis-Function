# Battery-Development-project

This project is a PySpark-based solution for processing battery data, performing forward and backward interpolation for missing values, and spatial interpolation using Radial Basis Functions (RBF). The final output is a CSV file containing the interpolated data.

# Features

**Data Preprocessing:** Handles data cleaning and renaming for consistent column names.

**Forward and Backward Interpolation:** Fills missing values in battery sensor data by leveraging forward and backward windows.

**Radial Basis Function (RBF) Interpolation:** Applies RBF interpolation to estimate spatially distributed sensor values on a battery grid.

**Output to CSV:** Saves the processed and interpolated data into a CSV file for further analysis.

# Prerequisites

Ensure the following tools and libraries are installed before running the project:

# Tools

Python 3.8+

Apache Spark 3.0+

Python Libraries

**Install the required Python libraries using pip:**

pip install pyspark pandas numpy scipy

# Installation

**Clone the repository:**

git clone https://github.com/your-repo/battery-data-interpolation.git
cd battery-data-interpolation

Place your input CSV file (sample_battery_data.csv) in the project directory.

Update the data_path_csv variable in the script if the file path differs.

# Project Structure

├── battery_interpolation.py    # Main PySpark script

├── sample_battery_data.csv     # Input data file

├── requirements.txt            # required libraries to pip install

├── README.md                   # Project documentation

# How to Run

Start the Spark environment.

spark-submit battery_interpolation.py

The processed output will be saved as battery_interpolation.csv in the specified directory.

# Key Components

Forward and Backward Interpolation

Radial Basis Function (RBF) Interpolation

Sample Output Data

The output CSV contains interpolated values for all sensor data columns, including spatially interpolated values for a 104-point grid.

**Contribution**

Feel free to fork this repository, submit pull requests, or report issues for any suggestions or improvements.

---
🙋 Author

LinkedIn: http://www.linkedin.com/in/SwapnilTaware

GitHub: https://github.com/itsSwapnil

Email: tawareswapnil23@gmail.com

---

# License

This project is licensed under the MIT License. See the LICENSE file for details.
