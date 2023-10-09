# Large Language Model System Benchmark
A benchmark Paper of  data science code generation LLM System



## ICASSP Data Science Code Dataset

**Dataset Structure Example:**

- **Approaches** differ from each other due to different ways to split the code/ divide the task.
- **Parts** differ from each other due to different part after split.

```
├── Paper1

		├── Approach 1

		├── Approach 2

		├──Approach 3

      			├── Part 1

      			├──Part 2

           			├── ans

           			├── input

           			├── `.cfg`

           			├── `split.txt`

       				├── `details.txt`

           			├── `prompt.txt`

           			├── `code_context.txt`

           			├── `reference_code.txt`

           			├── `test_code.py`         (compare the output)

           			├──`test_generate_pickle.py` (run the generated code)

├── Paper2

├── Paper3

├── Paper4
```

 

High Level Approach with minor split information

- **`.cfg`** contains the meta info of the paper, including the background, high-level idea of the whole paper.

- **`split`** Requires the system to split and finish the high-level task by itself without providing additional tips.

- **`details`** Blank in this approach case.

- **`prompt.txt`** is the official prompt we recommend using for querying large language models. Prompt.txt will include information from **`.cfg`** and **`split`** from, and also some comments/ key points extracted from the code.
-   **ans and input** contain the pickles file caching the input and solution objects.

- **`code_context.txt`** is the executable code context for evaluation.

- **`reference_code.txt`** is the ground truth solution code.

 

- **`test_code.py`** contains the testing code

- **`test_generate_pickle.py`** is the script we use the generate the input pickle files in input

If the dataset is ever corrupted, you can run python `cache_input_ans.py` to regenerate input and ans in each problem directory.

…

Several approaches between them

…

LowLevel Approach with minor split

- **`.cfg`** contains the meta info of the paper, including the background, high-level idea of the whole paper
- **`split`** contains the information of all parts of the code and . (Read Data, Data Preprocessing, Reconstruct Dataset Class, Data Cleaning, Models, Training, Loss Function, Evaluation, Metrics )
- **`details`** contains the comments/ key points extracted from the ground truth solution code.
- **`prompt.txt`** is the official prompt we recommend using for querying large language models. `Prompt.txt` will include information from **`.cfg`, `details` ,**and **`split`**
- **ans and input** contain the pickles file caching the input and solution objects.
- **`code_context.txt`** is the executable code context for evaluation.
- **`reference_code.txt`** is the ground truth solution code.
- **`test_code.py`** contains the testing code
- **`test_generate_pickle.py`** is the script we use the generate the input pickle files in input

 

 

 

 

 

 

 

­­­­­
