# Energy-efficiency-in-Deep-Learning-Consumption
This analysis explores the inefficiencies of energy usage in deep learning models. Comprised of 30k+ nodes to find: What are the key factors that contribute to energy consumption in AI models, and how can businesses reduce their energy usage without compromising performance?

To open this file directly in spyder you can copy the following code from line 6:

import os
import urllib.request
import subprocess

# URL of the Python file
url = "https://raw.githubusercontent.com/angelina7x/Energy-efficiency-in-Deep-Learning-Consumption/main/Energy%20Efficiency%20in%20Deep%20Learning%20Models.py"

# Local file path
local_file = "Energy_Efficiency_in_Deep_Learning_Models.py"

# Download the file
urllib.request.urlretrieve(url, local_file)

# Open the file in Spyder
subprocess.run(["spyder", local_file])
