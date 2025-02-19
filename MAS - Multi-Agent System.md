# Multi-Agent System (MAS) for Automated ML Workflow

## Expected Solution
- Implement a **multi-agent system (MAS)** using (1) **AutoGen**, (2) **CrewAI**, and (3) **OpenAI Swarm**.
- Agents handle different parts of an **ML workflow**:
  - **Agent 0:** Generate simulated data (e.g., linear regression with specified true parameters).
  - **Agent 1:** Load and clean data.
  - **Agent 2:** Perform modeling (e.g., linear regression).
  - **Agent 3:** Analyze results (e.g., regression diagnostics).
  - **Agent 4:** Visualize results (e.g., scatterplots, feature importance).
  - **Agent 5:** Generate an executive report.

## Test Tasks & Metrics
### **Task 1: Identifying Significant Variables in Regression**
- Use simulated data to determine the most influential variables.
- **Methods:** Linear regression-based feature selection.
- **Metrics:** F-score, recall, or precision in selecting correct variables.

### **Task 2: Predicting Y Given X (Supervised Learning)**
- Train models for regression/classification.
- **Metrics:** 
  - Regression: **L1/L2 loss**.
  - Classification: **Accuracy**.

### **Task 3: Hypothesis Testing**
- Test the significance of a coefficient in regression.
- **Methods:** t-tests, Mann-Whitney U tests.
- **Metrics:** Type I/II errors, p-values.

## Expected Tools
- **Multi-Agent Frameworks:** AutoGen, CrewAI.
- **Execution:** Python subprocess (`subprocess.run()` for Python, `rpy2` for R).
- **Visualization:** Matplotlib, Seaborn.

## Deliverable
- **A Python class** that includes:
  - **Predefined agent workflows** and hyperparameters (e.g., agent roles, data generation parameters).
  - **Configurable ML pipeline** that allows users to modify models and analysis settings.
  - **Final output:** 
    - A structured **report** summarizing findings.
    - **Plots and statistical insights** as saved JSON or image files.
  - **README file** with: setup guide and example use cases.
