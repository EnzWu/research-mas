import os
import re
import ast
import json
import markdown
import sys
from io import StringIO
from mistralai import Mistral
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

api_key = "Your_mistral_api_key"

client = Mistral(api_key=api_key)
agents = {
    "agent0" :  "ag:16138242:20250223:agent-0:414d3677",
    "agent1" :  "ag:16138242:20250223:agent-1:c12d9bbf",
    "agent2" :  "ag:16138242:20250223:agent-2:4c8fd40e",
    "agent3" :  "ag:16138242:20250223:agent-3:a7f289b0",
    "agent4" :  "ag:16138242:20250223:agent-4:fa0adac7",
    "agent5" :  "ag:16138242:20250223:agent-5:83ffd2f8",
    "agent6" :  "ag:16138242:20250223:agent-6:0f2ee8da"
}

def extract_python_code(text):
    code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
    return [block.strip() for block in code_blocks]

def extract_and_organize_code(text):
    code_blocks = extract_python_code(text)
    return code_blocks

def run_queryintake_agent(query):
    print("### Run query intake agent")
    print(f"User query: {query}")
    try:
        response = client.agents.complete(
            agent_id=agents.get("agent6"),
            messages=[
                {
                    "role": "user",
                    "content": "You are an AI assistant that formats responses as JSON objects. Always respond with a valid JSON object."
                },
                {
                    "role": "user",
                    "content": f"Please process the following query and return a JSON object with the appropriate parameters: {query}"
                },
            ]
        )
        result = response.choices[0].message.content
        try:
            print(result)
            result_json = json.loads(result)
            print(f"### Query taken and the general input would be:###")
            return result_json
        except json.JSONDecodeError:
            print("Error: Response is not a valid JSON. Retrying...")
            return run_queryintake_agent(query)
    except Exception as e:
        print(f"Request failed: {e}. Please check your request.")
        return None

def write_markdown_file(filename, content):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, 'w') as f:
        f.write(content)

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

class PythonAgentWorkflow:
    def __init__(self):
        self.dataset = None
        self.images_dir = self.create_images_folder()
        global output0
    # create image folder to store visualizations
    def create_images_folder(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(script_dir, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        return images_dir
    # general agent call
    def run_agent(self, agent_name, input):
        agent_id = agents.get(agent_name)
        if not agent_id:
            print(f"Error: No agent found for {agent_name}")
            return None, None
        print(f"### Running {agent_name} ###")
        print(f"Task: {input}")
        try:
            # record each console result
            original_stdout = sys.stdout
            output_buffer = StringIO()
            sys.stdout = Tee(sys.stdout, output_buffer)

            response = client.agents.complete(
                agent_id=agent_id,
                messages=[
                    {
                        "role": "user",
                        "content": input
                    }
                ]
            )

            result = response.choices[0].message.content
            print(f"### {agent_name} Output ###\n{result}\n")
            codes = extract_python_code(result)
            self.execute_code(codes, dataset=None)

            sys.stdout = original_stdout
            captured_output = output_buffer.getvalue()
            # print(result)
            return result, captured_output

        except Exception as e:
            print(f"Request failed for {agent_name}: {e}")
            return None, None

    def execute_code(self, codes, dataset=None):
        for i, code in enumerate(codes):
            print(f"Executing code block {i+1}:")
            # execute codes generated in namespace
            try:
                namespace = {
                    'pd': pd,
                    'plt': plt,
                    'sns': sns,
                    'LinearRegression': LinearRegression,
                    'train_test_split': train_test_split,
                    'mean_squared_error': mean_squared_error,
                    'r2_score': r2_score,
                    'StringIO': StringIO,
                    'os': os,
                    'data': dataset, # Inject dataset
                    'images_dir': self.images_dir
                }
                exec(code, namespace)
                print("Execution successful.\n")
            except Exception as e:
                print(f"Error executing code: {e}\n")

    def task1(self, query, data):
        result, output1 = self.run_agent("agent2", f"Use the cleaned dataset{data} to perform linear regression with {query['model_specification']} model setting.")
        result, output2 = self.run_agent("agent3", f"Use the cleaned dataset{data} , determine the most influential variables with modeling result {output1} using linear regression-based feature selection with {query['model_specification']}.")
        print(f"### Task 1 Output ###\n{result}\n")
        return result, output1 + output2

    def task2(self, query, data):
        result, output1 = self.task1(query, data)
        result, output2 = self.run_agent("agent2", f"Use the cleaned dataset {data}, train {query['model_specification']} model with hyperparameters in: {query['model_specification']} and feature selection results {output1}")
        result, output3 = self.run_agent("agent3", f"Use the cleaned dataset {data} , analyze prediction performance based on the training result {output1 + output2} and model setting {query}.")
        return result, output1 + output2 + output3

    def task3(self, query, data):
        result, output1 = self.task1(query, data)
        result, output2 = self.run_agent("agent3", f"Use the cleaned dataset {data} , perform hypothesis testing on the significance of coefficients in the regression discussed in {output1} with method from setting {query}.")
        return result, output1 + output2

    def workflow(self, query):
        if query['use_uploaded_dataset'] == False:
            result, output0 = self.run_agent("agent0", f"Generate synthetic data with parameters in: {query}.")
        else:
            self.dataset = pd.read_csv(query['data_file_path'])
            output0 = self.dataset.to_string()
        
        _ , output1 = self.run_agent("agent1", f"Clean the dataset using parameters: {query['cleaning_parameters']}. The dataset could be obtained by running the code in previous output: {output0} or directly imported from data file path {self.dataset}. Parameters and data file path should be explicitly stated in the code.")

        output2 = ""
        if query['task'][0] == '1':
            result, output2 = self.task1(query,output0)
        elif query['task'][0] == '2':
            result, output2 = self.task2(query, output0)
        elif query['task'][0] == '3':
            result, output2 = self.task3(query,output0)
        else:
            result, output2 = "No task performed", ""
        # generate visual results
        visual_result, output3 = self.run_agent("agent4", f"Write python codes in the given data setting for generating visualizations based on analysis results {output0}, {output1}, and {output2}.")
        # generate report/Readme file content
        _, report_content= self.run_agent("agent5", f"Compile final report using analysis results {output0}, {output1}, {output2}, and {output3}, visualizations generated, and report template {query['report_template']}. Format in Markdown and include images using relative paths.")
        _, readme_content= self.run_agent("agent5", f"Generate a README file in Markdown format using the template {query['readme_template']}.")

        write_markdown_file("report.md", report_content)
        write_markdown_file("README.md", readme_content)

        print("### Exit WORKFLOW")
        return None

# demonstration
query = """
Load this data: https://raw.githubusercontent.com/fivethirtyeight/data/master/bad-drivers/bad-drivers.csv

The dataset consists of 51 datapoints and has eight columns:
- State
- Number of drivers involved in fatal collisions per billion miles
- Percentage Of Drivers Involved In Fatal Collisions Who Were Speeding
- Percentage Of Drivers Involved In Fatal Collisions Who Were Alcohol-Impaired
- Percentage Of Drivers Involved In Fatal Collisions Who Were Not Distracted
- Percentage Of Drivers Involved In Fatal Collisions Who Had Not Been Involved In Any Previous Accidents
- Car Insurance Premiums ()
"""

result = run_queryintake_agent(query)
print(result)
test = PythonAgentWorkflow()
test.workflow(result)
