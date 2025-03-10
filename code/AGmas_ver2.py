import os
import re
import json
import asyncio
from typing import List
from dataclasses import dataclass
from autogen_core import MessageContext, RoutedAgent, SingleThreadedAgentRuntime, TopicId, message_handler, type_subscription, FunctionCall
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage, LLMMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool, Tool
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()
my_api = os.getenv("OPENAI_API_KEY")

if not my_api:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=my_api
    )

# Create the tool.
code_executor = DockerCommandLineCodeExecutor()
code_execution_tool = PythonCodeExecutionTool(code_executor)

def extract_code_from_function_call(function_call: FunctionCall) -> str:
    try:
        arguments = json.loads(function_call.arguments)
        return arguments
    except (json.JSONDecodeError, AttributeError):
        return ""

def extract_python_code(text):
    code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
    return [block.strip() for block in code_blocks]

def write_markdown_file(filename, content):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, 'w') as f:
        f.write(content)

@dataclass
class Message:
    content: str

config_extractor_topic_type = "ConfigExtractorAgent"
data_gen_topic_type = "DataGenerationAgent"
data_clean_topic_type = "DataCleanerAgent"
modeling_topic_type = "ModelerAgent"
analysis_topic_type = "AnalysisAgent"
visualization_topic_type = "VisualizingAgent"
summary_topic_type = "ReportAgent"

@type_subscription(topic_type=config_extractor_topic_type)
class ConfigExtractorAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A analysis configuration extractor agent.")
        self._system_message = SystemMessage(
            content=(
                '''You are a specialized user query processor that could turn the user query input into a json output for a later systematic automated machine learning workflow with format: 
                "simulation_parameters": {Parameters for the simulation, such as sample size or duration}, 
                "use_uploaded_dataset": TRUE/FALSE boolean type usable by python, 
                "data_file_path": "The file path of the data to be used", "cleaning_parameters": {Parameters for data cleaning, such as handling missing values}, 
                "model_specification": {Specifications for the machine learning model}, 
                "analysis_settings": {Settings for data analysis}, "visualization_settings": {Settings for data visualization}, "report_template": "Template for generating reports", "readme_template": "Template for the README file", "agent_workflows": [Workflows for the AI agents],
                "task":"{task number} the summary" ".
                Even if the user only provides a dataset file path in the query, we will consider it using the uploaded dataset. Otherwise, it should be false. Besides, use the user input to determine which task among below three the user is up for: 
                Task 1: Identifying significant Variables in Regression
                    - Use simulated data to determine the most influential variables.
                    - Methods: Linear regression-based feature selection.
                    - Metrics: F-score, recall, or precision in selecting correct variables.

                Task 2: Predicting Y Given X (Supervised Learning)
                    - Train models for regression/classification.
                    - Metrics:
                        - Regression: **L1/L2 loss**.
                        - Classification: **Accuracy**.

                Task 3: Hypothesis Testing
                    - Test the significance of a coefficient in regression.
                    - Methods: t-tests, Mann-Whitney U tests.
                    - Metrics: Type I/II errors, p-values.'''
            )
        )
        self._model_client = model_client

    @message_handler
    async def handle_user_query(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Please process the following query and return a JSON object with the appropriate parameters: {message.content}"
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        # record = record + response
        assert isinstance(response, str)
        print(f"{'-'*80}\n{self.id.type}:\n{response}")

        await self.publish_message(Message(response), topic_id=TopicId(data_clean_topic_type, source=self.id.key))

@type_subscription(topic_type=data_clean_topic_type)
class DataCleanerAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool: List[Tool]) -> None:
        super().__init__("A data cleaning agent.")
        self._system_messages: List[LLMMessage] = [SystemMessage(
            content=(
                '''You are a data cleaning expert experienced in preprocessing and cleaning data, 
                ensuring quality input for the ML pipeline with guidelines set by user input cleaning parameters. 
                You load and clean the dataset from the data file path in the input if available, 
                using cleaning rules defined in cleaning parameters in the input as well. 
                The code in the arguments of functioncall should have key named 'code'.
                Be sure to break down the code chunks in steps. Output the cleaned dataset. 
                Make sure not to use a place holder but to write out the python codes with specific given file path and dataset parameters from previous input.
                By executing the code, the cleaned data should be save to local file with file name 'cleaned_dataset.csv'. 
                Be sure to define every parameter in the currunt context that suits the dataset, and the produced code entirely should be able to run by themselves.
                Make sure using correct indentation that guarantees the code is directly executable.'''
            )
        )]
        self._model_client = model_client
        self._tools = tool

    @asynccontextmanager
    async def local_execution_context(self, code: str):
        """
        Async context manager to execute Python code in a local namespace.
        """
        local_namespace = {}
        try:
            # Execute the code dynamically in the provided namespace
            exec(code, {}, local_namespace)
            yield local_namespace  # Provide access to the namespace after execution
        except Exception as e:
            yield {"error": f"Error executing code: {str(e)}"}
    async def execute_python_code(self, code: str) -> str:
        """
        Executes Python code locally within a controlled namespace.
        """
        async with self.local_execution_context(code) as result:
            if "error" in result:
                return result["error"]
            return f"Code executed successfully. Namespace: {result}"

    @message_handler
    async def handle_dataset_cleaning(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Clean the dataset using cleaning parameters and dataset obtained from the previous agent output: {message.content}. Parameters and data file path should be explicitly stated in the code."
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=prompt, source="user")]
        llm_result = await self._model_client.create(
            messages=session,
            tools=self._tools,
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        if isinstance(response, str):
            print(f"{'-'*80}\n{self.id.type}:\n{response}")
            await self.publish_message(Message(response), topic_id=TopicId(modeling_topic_type, source=self.id.key))
            return  #  Exit early after publishing
        print(f"{'-'*80}\n{self.id.type}:\n{response}")
        if isinstance(response, list) and all(isinstance(call, FunctionCall) for call in response):
            code_to_execute = extract_code_from_function_call(response[0])
            print(code_to_execute)

            execution_result = await self.execute_python_code(code_to_execute['code'])
            response.append(execution_result)
            print(f"{'-'*80}\n{self.id.type}:\nCode executed. Result:\n{execution_result}")
            await self.publish_message(Message(response), topic_id=TopicId(modeling_topic_type, source=self.id.key))

        else:
            print("Unexpected response format from model client.")

@type_subscription(topic_type=modeling_topic_type)
class ModelerAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool: List[Tool]) -> None:
        super().__init__("A data modeling agent.")
        self._system_message = SystemMessage(
            content=(
                '''You are a modeling specialist skilled in building and tuning models, 
                leveraging the configurable ML pipeline and hyperparameter settings specified in model specification and analysis settings given in the user's input. 
                You fit models with the specified dataset to perform feature selection; 
                Train regression/classification models on the cleaned dataset based on model specification from the input while allowing users to modify hyperparameters and pipeline configurations. 
                Be sure to break down the code chunks in steps. 
                Make sure not to use a place holder but to write out the python codes with specific given file path and dataset parameters from previous input.
                Every file generated during the analysis process should be stored to the local file by certain python code for later reference in summary report.
                Be sure to define every parameter in the currunt context that suits the dataset, and the produced code entirely should be able to run by themselves.
                Make sure using correct indentation that guarantees the code is directly executable.'''
            )
        )
        self._model_client = model_client
        self._tools = tool

    @asynccontextmanager
    async def local_execution_context(self, code: str):
        """
        Async context manager to execute Python code in a local namespace.
        """
        local_namespace = {}
        try:
            # Loop through each code block and execute it
            for idx, code_block in enumerate(code):
                print(f"Executing code block {idx + 1}:")
                try:
                    exec(code_block, {}, local_namespace)
                    print("Execution successful.\n")
                except Exception as e:
                    local_namespace["error"] = f"Error executing code block {idx + 1}: {str(e)}"
                    print(f"Execution failed: {str(e)}\n")
                # break  # Stop execution if an error occurs
            yield local_namespace  # Provide access to the namespace after execution
        except Exception as e:
            local_namespace["error"] = f"Context manager error: {str(e)}"
            yield local_namespace  # Ensure a yield happens even in outer exception
        finally:
            pass
    async def execute_python_code(self, code: str) -> str:
        """
        Executes Python code locally within a controlled namespace.
        """
        async with self.local_execution_context(code) as result:
            if "error" in result:
                return result["error"]
            return f"Code executed successfully. Namespace: {result}"

    @message_handler
    async def handle_data_modeling(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Use the cleaned dataset to perform linear regression with specified model setting and feature selection or train model with hyperparameters based on information obtained from preprocessed results {message.content}."
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        if isinstance(response, str):
            print(f"{'-'*80}\n{self.id.type}:\n{response}")
            code = extract_python_code(response)
            print(code)
            if code:
                execution_result = await self.execute_python_code(code)
                output = response + execution_result
                print(f"{'-'*80}\n{self.id.type}:\nCode executed. Result:\n{execution_result}")
                await self.publish_message(Message(output), topic_id=TopicId(analysis_topic_type, source=self.id.key))
                return
            else:
                await self.publish_message(Message(response), topic_id=TopicId(analysis_topic_type, source=self.id.key))
                return  #  Exit early after publishing
        else:
            print("Unexpected response format from model client.")

@type_subscription(topic_type=analysis_topic_type)
class AnalysisAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool: List[Tool]) -> None:
        super().__init__("A modeling analysis agent.")
        self._system_message = SystemMessage(
            content=(
                '''You are a data diagnostics expert that possesses a deep understanding of statistical diagnostics, 
                capable of computing F-score, recall, precision, L1/L2 losses for regression, accuracy for classification, 
                and executing hypothesis tests such as t-tests and Mann-Whitney U tests. 
                You perform advanced analyses including feature selection, prediction evaluation, 
                and hypothesis testing using methods and metrics defined in analysis settings from the user's input. 
                Be sure to break down the code chunks in steps. 
                Make sure not to use a place holder but to write out the python codes with specific given file path and dataset parameters from previous input.
                Every file generated during the analysis process should be stored to the local file by certain python code for later reference in summary report.
                Be sure to define every parameter in the currunt context that suits the dataset, and the produced code entirely should be able to run by themselves.
                Make sure using correct indentation that guarantees the code is directly executable.'''
            )
        )
        self._model_client = model_client
        self._tools = tool

    @asynccontextmanager
    async def local_execution_context(self, code: str):
        """
        Async context manager to execute Python code in a local namespace.
        """
        local_namespace = {}
        try:
            # Loop through each code block and execute it
            for idx, code_block in enumerate(code):
                print(f"Executing code block {idx + 1}:")
                try:
                    exec(code_block, {}, local_namespace)
                    print("Execution successful.\n")
                except Exception as e:
                    local_namespace["error"] = f"Error executing code block {idx + 1}: {str(e)}"
                    print(f"Execution failed: {str(e)}\n")
                # break  # Stop execution if an error occurs
            yield local_namespace  # Provide access to the namespace after execution
        except Exception as e:
            local_namespace["error"] = f"Context manager error: {str(e)}"
            yield local_namespace  # Ensure a yield happens even in outer exception
        finally:
            pass
    async def execute_python_code(self, code: str) -> str:
        """
        Executes Python code locally within a controlled namespace.
        """
        async with self.local_execution_context(code) as result:
            if "error" in result:
                return result["error"]
            return f"Code executed successfully. Namespace: {result}"

    @message_handler
    async def handle_modeling_analysis(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Use the cleaned dataset to determine the most influential variables with modeling result using linear regression-based feature selection or analyze prediction performance or perform hypothesis testing on the significance of coefficients in the regression discussed with method based on the training result and model setting obtained from preprocessed results {message.content}"
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        if isinstance(response, str):
            print(f"{'-'*80}\n{self.id.type}:\n{response}")
            code = extract_python_code(response)
            print(code)
            if code:
                execution_result = await self.execute_python_code(code)
                output = response + execution_result
                print(f"{'-'*80}\n{self.id.type}:\nCode executed. Result:\n{execution_result}")
                await self.publish_message(Message(output), topic_id=TopicId(visualization_topic_type, source=self.id.key))
                return
            else:
                await self.publish_message(Message(response), topic_id=TopicId(visualization_topic_type, source=self.id.key))
                return  #  Exit early after publishing
        else:
            print("Unexpected response format from model client.")

@type_subscription(topic_type=visualization_topic_type)
class VisualizingAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool: List[Tool]) -> None:
        super().__init__("A analysis visualizing agent.")
        self._system_message = SystemMessage(
            content=(
                '''You are a visualization specialist that experts in visual data analysis, 
                transforming model results and diagnostics into clear, insightful graphical representations. 
                You create executable visualizations Python code using Matplotlib and Seaborn based on visualization settings given from the input or visual aid with best fit, 
                including scatterplots, feature importance charts, and diagnostic plots. 
                The code should also include steps that create a local image file named "images" to store all the visualizations in it and return a response with a brief explanation of each plot for later report generation. 
                Be sure to break down the whole python code into code chunks by different steps.
                Make sure not to use a place holder but to write out the python codes with specific given file path and dataset parameters from previous input.
                By executing the code, the visualization generated should be save to local file named 'images' for later reference in report markdown. 
                Every file generated during the analysis process should be stored to the local file by certain python code for later reference in summary report.
                Be sure to define every parameter in the currunt context that suits the dataset, and the produced code entirely should be able to run by themselves.
                Make sure using correct indentation that guarantees the code is directly executable.'''
            )
        )
        self._model_client = model_client
        self._tools = tool

    @asynccontextmanager
    async def local_execution_context(self, code: str):
        """
        Async context manager to execute Python code in a local namespace.
        """
        local_namespace = {}
        try:
            # Loop through each code block and execute it
            for idx, code_block in enumerate(code):
                print(f"Executing code block {idx + 1}:")
                try:
                    exec(code_block, {}, local_namespace)
                    print("Execution successful.\n")
                except Exception as e:
                    local_namespace["error"] = f"Error executing code block {idx + 1}: {str(e)}"
                    print(f"Execution failed: {str(e)}\n")
                # break  # Stop execution if an error occurs
            yield local_namespace  # Provide access to the namespace after execution
        except Exception as e:
            local_namespace["error"] = f"Context manager error: {str(e)}"
            yield local_namespace  # Ensure a yield happens even in outer exception
        finally:
            pass
    async def execute_python_code(self, code: str) -> str:
        """
        Executes Python code locally within a controlled namespace.
        """
        async with self.local_execution_context(code) as result:
            if "error" in result:
                return result["error"]
            # Check if a PNG filepath is available
            png_filepath = result.get('png_filepath', None)
            if png_filepath:
                return f"Visualization saved to: {png_filepath}"
            return f"Code executed successfully. Namespace: {result}"

    @message_handler
    async def handle_analysis_visualization(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Write python codes in the given data setting for generating visualizations based on analysis results obtained {message.content}"
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        if isinstance(response, str):
            print(f"{'-'*80}\n{self.id.type}:\n{response}")
            code = extract_python_code(response)
            print(code)
            if code:
                execution_result = await self.execute_python_code(code)
                output = response + execution_result
                print(f"{'-'*80}\n{self.id.type}:\nCode executed. Result:\n{execution_result}")
                await self.publish_message(Message(output), topic_id=TopicId(summary_topic_type, source=self.id.key))
                return
            else:
                await self.publish_message(Message(response), topic_id=TopicId(summary_topic_type, source=self.id.key))
                return  #  Exit early after publishing
        else:
            print("Unexpected response format from model client.")

@type_subscription(topic_type=summary_topic_type)
class ReportAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool: List[Tool]) -> None:
        super().__init__("A analysis report agent.")
        self._system_message = SystemMessage(
            content=(
                '''You are a report generation specialist that specializes in crafting detailed, 
                structured reports that integrate JSON statistical insights and visual outputs, 
                as well as comprehensive documentation for deployment and configuration. 
                You summarize findings and assemble the final report based on previous analysis from input and generate a README Markdown file containing setup guides, use cases, and instructions for modifying hyperparameters and agent workflows from agent workflows given in the input. 
                You should output report content and README content seperately as str type stored in one json form output.
                Be explicit about the regression summary output.'''
            )
        )
        self._model_client = model_client
        self._tools = tool

    @message_handler
    async def handle_analysis_report(self, message: Message, ctx: MessageContext) -> None:
        prompt = f"Compile final report using analysis results obtained from previous response {message.content} and visualizations generated. Format in Markdown and include images using relative paths. Return a JSON object with two keys:- `report`: The markdown content for the final report.- `readme`: The markdown content for the README file.Ensure the response is a valid JSON object."
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
        )
        response = llm_result.content
        if isinstance(response, str):
            print(f"{'-'*80}\n{self.id.type}:\n{response}")
        else:
            print("Unexpected response format from model client.")
        try:
            response = re.findall(r'```json\n(.*?)\n```', response, re.DOTALL)[0]
            response_data = json.loads(response)
            report = response_data.get("report", "")
            readme = response_data.get("readme", "")

            write_markdown_file("Report.md", report)
            write_markdown_file("README.md", readme)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            print("LLM Response:", response)

runtime = SingleThreadedAgentRuntime()

async def register_agents():
    data_cleaner_tool_schema = [
        FunctionTool(
            name="CodeExecutor",
            description="Executes python code in a sandbox and returns the output",
            func=code_execution_tool.run_json,
        ),
    ]

    await ConfigExtractorAgent.register(
        runtime, type=config_extractor_topic_type, factory=lambda: ConfigExtractorAgent(model_client=model_client))

    await DataCleanerAgent.register(
        runtime, type=data_clean_topic_type, factory=lambda: DataCleanerAgent(model_client=model_client,tool=data_cleaner_tool_schema))

    await ModelerAgent.register(
        runtime, type=modeling_topic_type, factory=lambda: ModelerAgent(model_client=model_client,tool=data_cleaner_tool_schema))

    await AnalysisAgent.register(
        runtime, type=analysis_topic_type, factory=lambda: AnalysisAgent(model_client=model_client,tool=data_cleaner_tool_schema))

    await VisualizingAgent.register(
        runtime, type=visualization_topic_type, factory=lambda: VisualizingAgent(model_client=model_client,tool=data_cleaner_tool_schema))

    await ReportAgent.register(
        runtime, type=summary_topic_type, factory=lambda: ReportAgent(model_client=model_client,tool=data_cleaner_tool_schema))

async def main():
    
    await register_agents()

    runtime.start()

    query = """
    Load this data: https://raw.githubusercontent.com/fivethirtyeight/data/master/bad-drivers/bad-drivers.csv
    Identify the most important parameter in the dataset and use it as target variable in later analysis.
    The dataset consists of 51 datapoints and has eight columns:
    - State
    - Number of drivers involved in fatal collisions per billion miles
    - Percentage Of Drivers Involved In Fatal Collisions Who Were Speeding
    - Percentage Of Drivers Involved In Fatal Collisions Who Were Alcohol-Impaired
    - Percentage Of Drivers Involved In Fatal Collisions Who Had Not Been Involved In Any Previous Accidents
    - Percentage Of Drivers Involved In Fatal Collisions Who Had Not Been Involved In Any Previous Accidents
    - Car Insurance Premiums ()
    """

    await runtime.publish_message(
        Message(content=query),
        topic_id=TopicId(config_extractor_topic_type, source="default"),
    )

    await runtime.stop_when_idle()

if __name__ == "__main__":
    asyncio.run(main())
