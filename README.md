Supply Chain Optimizer 🩺🚚
Overview
This project is an interactive web application designed to solve complex vehicle routing problems (VRP) for urban logistics. Using real-world taxi trip data from NYC as a proxy for customer locations, this tool simulates a medical supply distribution network and uses Google's OR-Tools to find the most efficient delivery routes.

The final product is a Dash web application that allows a user to set parameters like fleet size and vehicle capacity, run the optimization, and visualize the optimized routes on an interactive map of New York City. The application also includes a "non-optimized" greedy algorithm to provide a clear benchmark and demonstrate the significant efficiency gains from the OR-Tools solver.

Key Features
Realistic Data Simulation: Leverages millions of real NYC taxi trip records to create a geographically accurate distribution of customer demand points.

Powerful Optimization Engine: Implements Google's OR-Tools to solve the Capacitated Vehicle Routing Problem (CVRP), minimizing total distance while respecting vehicle capacity constraints.

Comparative Analysis: Includes a "non-optimized" greedy solver to provide a direct, visual comparison against the sophisticated OR-Tools solution, quantifying the value of optimization.

Interactive Web Interface: A user-friendly dashboard built with Dash and Plotly allows for easy control of simulation parameters and clear visualization of results.

Robust and Efficient: The application is structured to run efficiently by using a consistent, pre-sampled set of orders for fair comparisons.

Tech Stack
Backend & Core Logic: Python

Optimization Solver: Google OR-Tools

Data Manipulation: Pandas, NumPy

Web Dashboard: Dash

Interactive Visualizations: Plotly

Production Web Server: Gunicorn

Project Structure
SupplyChainOptimizer/
│
├── .venv/                      # Virtual environment files
├── app.py                      # The main Dash web application
├── optimizer_prototype.ipynb   # Jupyter Notebook for development and testing
├── requirements.txt            # List of Python packages for deployment
├── Procfile                    # Command for the deployment server (Render)
├── yellow_tripdata_2024-01.parquet # Raw NYC taxi data (required for the app)
└── taxi+_zone_lookup.csv           # Lookup table for taxi zones (required for the app)

Local Setup and Usage
To run this application on your local machine, follow these steps:

Clone the Repository:

git clone https://github.com/your-username/supply-chain-optimizer.git
cd supply-chain-optimizer

Create and Activate a Virtual Environment:

# Create the environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source .venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

Run the Application:

python app.py

Open your web browser and navigate to http://127.0.0.1:8050/ to use the tool.

Future Improvements
This project serves as a powerful foundation. Future enhancements could include:

Real-World Routing: Integrating a routing engine like OSRM or the Google Maps API to use actual road network distances and visualize turn-by-turn routes instead of straight lines.

Advanced Constraints: Adding complexity to the model, such as delivery time windows, driver shifts, or multi-depot starting points.

Live Deployment: Finalizing the deployment on a service like Render by pre-processing the data to overcome memory limitations.
