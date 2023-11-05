import sys
import os

from typing import Callable
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/dspy")
import dspy
import dsp

gpt4 = dspy.OpenAI(model='gpt-4')
dspy.settings.configure(lm=gpt4)

class ActionInfo:
    name: str
    description: str
    usage: dict
    return_value: str
    function: Callable
    is_primitive: bool = False

noop = lambda  **kwargs : None

ACTIONS = [
    ActionInfo(
        name="List Files",
        description="Use this to navigate the file system.",
        usage={   
            "dir_path": "a valid relative path to a directory, such as \".\" or \"folder1/folder2\""
        },
        return_value="The observation will be a list of files and folders in dir_path or current directory is dir_path is empty, or an error message if dir_path is invalid.",
        function=noop,
        is_primitive=True
    ),
    ActionInfo(
        name="Read File",
        description="Use this to read an existing file.",
        usage={
            "file_name": "a valid file name with relative path to current directory if needed"
        },
        return_value="The observation will be the contents of the file read.",
        function=noop,
        is_primitive=True
    ),
    ActionInfo(
        name="Write File",
        description="Use this to write a file. If the file already exists, it will be overwritten.",
        usage={
            "file_name": "a valid file name with relative path to current directory if needed",
            "content": "the content to be written to the file"
        },
        return_value="A success message if the file is written successfully, or an error message if the file cannot be written.",
        function=noop,
        is_primitive=True
    ),
    ActionInfo(
        name="Append File",
        description="Use this to append a file to a new location with a new name.",
        usage={
            "file_name": "a valid file name with relative path to current directory if needed",
            "content": "the content to be appended to the file"
        },
        return_value="A success message if the file is appended successfully, or an error message if the file cannot be appended.",
        function=noop,
        is_primitive=True
    ),
    ActionInfo(
        name="Copy File",
        description="Use this to copy a file to a new location with a new name.",
        usage={
            "source": "a valid file name with relative path to current directory if needed",
            "destination": "a valid file name with relative path to current directory if needed"
        },
        return_value="A success message if the file is copied successfully, or an error message if the file cannot be copied.",
        function=noop,
        is_primitive=True
    ),
    ActionInfo(
        name="Undo Edit Script",
        description="Use this to undo the last edit of the python script.",
        usage={
            "script_name": "a valid python script name with relative path to current directory if needed"
        },
        return_value="The observation will be the content of the script before the last edit. If the script does not exist, the observation will be an error message.",
        function=noop,
        is_primitive=True
    ),
    ActionInfo(
        name="Execute Script",
        description="Use this to execute the python script. The script must already exist.",
        usage={
            "script_name": "a valid python script name with relative path to current directory if needed"
        },
        return_value="The observation will be output of the script or errors.",
        function=noop,
        is_primitive=True
    ),
    ActionInfo(
        name="Python REPL",
        description="A python REPL. Use this to execute single line python commands.",
        usage={
            "command": "a valid python command"
        },
        return_value="The observation will be output of the command or errors.",
        function=noop,
        is_primitive=True 
    ),
    ActionInfo(
        name="Request Help",
        description="Use this to request help from human. Use this only when the provided tools and files are not enough for accomplishing necessary steps, such as requesting API reference or installing a library. So you should check through the provided tools and files first.",
        usage={
            "request": "a detailed description on what to do"
        },
        return_value="The observation will be the response from human.",
        function=noop,
        is_primitive=True
    ),
    ActionInfo(
        name="Final Answer",
        description="Use this to provide the final answer to the current task.",
        usage={
            "final_answer": "a detailed description on the final answer"
        },
        return_value="The observation will be empty.",
        function=(lambda **kwargs: ""),
        is_primitive=True
    ),ActionInfo(
        name="Understand File",
        description="Use this to read the whole file and understand certain aspects. You should provide detailed description on what to look for and what should be returned. To get a better understanding of the file, you can use Inspect Script Lines action to inspect specific part of the file.",
        usage={
            "file_name": "a valid file name with relative path to current directory if needed",
            "things_to_look_for": "a detailed description on what to look for and what should returned"
        },
        return_value="The observation will be a description of relevant content and lines in the file. If the file does not exist, the observation will be an error message.",
        function=noop
    ),
    ActionInfo(
        name="Append Summary to Research Log",
        description="Append to the summary of previous step to research log",
        usage={
            "content": "a string within 500 character limit"
        },
        return_value="The observation will be a success message if the content is appended to the research log. Otherwise, the observation will be an error message.",
        function=noop
    ),
    ActionInfo(
        name="Inspect Script Lines",
        description="Use this to inspect specific part of a python script precisely, or the full content of a short script. The number of lines to display is limited to 100 lines. This is especially helpful when debugging.",
        usage={
            "script_name": "a valid python script name with relative path to current directory if needed",
            "start_line_number": "a valid line number",
            "end_line_number": "a valid line number"
        },
        return_value="The observation will be the content of the script between start_line_number and end_line_number . If the script does not exist, the observation will be an error message.",
        function=noop
    ),
    ActionInfo(
        name="Edit Script (AI)",
        description="Use this to do a relatively large but cohesive edit over a python script. Instead of editing the script directly, you should describe the edit instruction so that another AI can help you do this.",
        usage={
            "script_name": "a valid python script name with relative path to current directory if needed. An empty sctipt will be created if it does not exist.",
            "edit_instruction": "a detailed step by step description on how to edit it.",
            "save_name": "a valid file name with relative path to current directory if needed"
        },
        return_value="The observation will be the edited content of the script. If the script does not exist, the observation will be an error message. You should always double check whether the edit is correct. If it is far from correct, you can use the Undo Edit Script action to undo the edit.",
        function=noop
    ),
    ActionInfo(
        name="Edit Script Segment (AI)",
        description="Use this to do a relatively large but cohesive edit over a python script over a segment. Instead of editing the script directly, you should describe the edit instruction so that another AI can help you do this.",
        usage={
            "script_name": "a valid python script name with relative path to current directory if needed. An empty sctipt will be created if it does not exist.",
            "start_line_number": "a valid line number",
            "end_line_number": "a valid line number",
            "edit_instruction": "a detailed step by step description on how to edit it.",
            "save_name": "a valid file name with relative path to current directory if needed"
        },
        return_value="The observation will be the edited content of the script. If the script does not exist, the observation will be an error message. You should always double check whether the edit is correct. If it is far from correct, you can use the Undo Edit Script action to undo the edit.",
        function=noop
    ),
    ActionInfo(
        name="Reflection",
        description="Use this to look over all the past steps and reflect. You should provide detailed description on what to reflect on and what should be returned.",
        usage={
            "things_to_reflect_on": "a detailed description on what to reflect on and what should be returned"
        },
        return_value="The observation will be a the reflection.",
        function=noop
    ),
    ActionInfo(
        name="Retrieval from Research Log",
        description="Use this to retrieve relevant information from the research log. You should provide detailed description on what to look for and what should be returned.",
        usage={
            "current_plan": "a detailed description of the current research plan and status",
        },
        return_value="The observation will be a description of relevant content and lines in the research log.",
        function=noop
    )
]

format_prompt_dict = {
    "Reflection": "What does the observation mean? If there is an error, what caused the error and how to debug?",
    "Research Plan and Status": "The full high level research plan, with current status and confirmed results of each step briefly annotated. It must only include progress that has been made by previous steps. If there is any update, enclose the new update text in double asterisks **like this**. If there is no update, just copy the previous step Research Plan and Status. The high level plan from the previous step should be fully retained, unless it is intentionally revised.",
    "Fact Check": "List all objective statements in the updates to Research Plan and Status one by one and point out whether it is guessed versus directly confirmed by the previous observation directly above. Performance numbers can only be confirmed by running the code and observing the output.",
    "Thought": "What you are currently doing, what actions to perform and why",
    "Action": "the action to take, should be one of the names of the tools",
    "Action Input": "the input to the action as a valid JSON string",
}


class ResearchReAct(dspy.Module):
    def __init__(self, signature, max_iters=5, tools=None):
        super().__init__()
        self.signature = signature = dspy.Predict(signature).signature
        self.max_iters = max_iters

        self.tools = tools
        self.tools = {tool.name: tool for tool in self.tools} #if isinstance(self.tools, list) else self.tools

        
        self.input_fields = {k: v for k, v in self.signature.kwargs.items() if isinstance(v, dspy.InputField)}

        inputs_ = ', '.join(self.input_fields.keys())
        instr = [signature.instructions]
        instr.append(f"You will be given {inputs_} and to accomplish it, You have access to the following tools: \n")
        for idx, tool in enumerate(self.tools):
            tool = self.tools[tool]
            usage = ",\n            ".join([f"\"{k}\": [{v}]" for k, v in tool.usage.items()])
            tools_prompt = f"""- {tool.name}:
            {tool.description}
            Usage:
            ```
            Action: {tool.name}
            Action Input: {{
                {usage}
            }}
            Observation: [{tool.return_value}]
            ```
                """.strip() + "\n"
            instr.append(tools_prompt)
        instr = '\n'.join(instr)
        self.react = [dspy.Predict(dsp.Template(instr, **self._generate_signature(i))) for i in range(1, max_iters + 1)]

    def _generate_signature(self, iters):
        signature_dict = {}
        for key, val in self.input_fields.items():
            signature_dict[key] = val
        for output, output_desc in format_prompt_dict.items():
            signature_dict[output] =  dspy.OutputField(prefix=f"{output}:", desc=output_desc)
        return signature_dict
    
    def act(self, output):
        try:
            action = output[f"Action"]
            action_input = output[f"Action Input"]

            if action == 'Final Answer': return "Done"

            output[f"Observation"] = self.tools[action](action_input)

        except Exception as e:
            output[f"Observation"] = "Failed to parse action. Bad formatting or incorrect action name."
        

    def forward(self, **kwargs):
        args = {key: kwargs[key] for key in self.input_fields.keys() if key in kwargs}

        for hop in range(self.max_iters):
            output = self.react[hop](**args)
            
            action_val = self.act(output)
            if action_val == "Done":
                break
            args.update(output)

        return dspy.Prediction() 
# [Reasoning]: Summarize the reasoning behind the action
# [Action]: Summarize all relevant details of the action objectively
# [Observation]: Summarize all relevant details in the observation objectively
#         

class SummerizationTask(dspy.Signature):
    '''
    Summarize your actions and observations. Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
    '''
    reflection = dspy.InputField(desc = format_prompt_dict['Reflection'])
    research_plan_and_status = dspy.InputField(desc = format_prompt_dict["Research Plan and Status"])
    fact_check = dspy.InputField(desc=format_prompt_dict["Fact Check"])
    thought = dspy.InputField(desc=format_prompt_dict["Thought"])
    action = dspy.InputField(desc=format_prompt_dict["Action"])
    action_input = dspy.InputField(desc=format_prompt_dict["Action Input"])
    action_ouput = dspy.InputField(desc= "The output of the action input")
    reasoning_summary = dspy.OutputField(desc="Summarize all relevant details of the action objectively")
    action_summary = dspy.OutputField(desc= "Summarize all relevant details of the action objectively")
    observation_summary = dspy.OutputField(desc="Summarize all relevant details in the action ouput objectively")

def summary_format(summaries):
    return f"\n{''.join([f'summary{i}: {summaries[i]}\n\n' for i in range(len(summaries))])}" 


class SummarizeResearchLog(dspy.Signature):
    '''
    Concisely summarize and list all relevant information from the research log that will be helpful for future step in this format:
    '''
    summaries = dspy.InputField(desc = "list of summaries", format=summary_format)
    output = dspy.OutputField(desc="output")

generate_answer = dspy.Predict(SummarizeResearchLog)
generate_answer(summaries = ["code1", "code2", "code3"])

class ResearchTask(dspy.Signature):
    """Perform research based on the given Research Problem. You do not know anything about this problem so far. 
Follow these instructions and do not forget them:
- First, come up with a high level plan based on your understanding of the problem and available tools and record it in the Research Plan and Status. You can revise the plan later.
- Research Plan and Status should well organized and succinctly keep track of 1) high level plan (can be revised), 2) what steps have been done and what steps are in progress, 3) short results and conclusions of each step after it has been performed. 
- Research Plan and Status must only include progress that has been made by previous steps. It should not include results not directly confirmed by the previous observation. 
- Performance numbers and estimates can only be confirmed and included in the status by running the code and observing the output.
- You should come up with a good experiment design that addresses the problem, and whenever applicable, define and measure the baseline performance of the relevant system or model before attempting any improvements.
- Follow the plan and try to achieve the goal as straightforwardly as possible.
- Highlight the supporting experiment results and reasoning before drawing any conclusions. 
- Do not try installing any new packages or libraries.
- If you believe you have solved the problem, you can use the Final Answer action to submit your answer. You can only submit once, so double check that you have achieved the goal before submitting.  """
    research_problem= dspy.InputField(desc="Contains the research problem and constraints")
# agent = ResearchReAct(ResearchTask, tools= ACTIONS)
# agent(research_problem = "Go through the data_description.txt file to understand the data and all the features. You can summarize it in your research logs to keep track of what all you have to do. Then fill in the provided train.py script to train a model and iterate over different models or feature selections to get a better performance. Never try to read any csv files directly. Do not forget to execute the changes you made to check for performance. ")