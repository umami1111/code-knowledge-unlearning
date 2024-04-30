# code-knowledge-unlearning

- Developing a knowledge unlearning algorithm for code generation models
- Built on an evaluation framework [CodeIPPrompt](https://sites.google.com/view/codeipprompt/) ([Github](https://github.com/zh1yu4nyu/CodeIPPrompt))

# Installation

The Python packages required to run the programs are numpy (tested on 1.21.2) and pyarrow (tested on 10.0.1). The binary of JPlag can be downloaded from the official [JPlag release](https://github.com/jplag/jplag/releases). For the Java component, please install [Dolos library](https://www.npmjs.com/package/@dodona/dolos-lib) following official instructions. 

Since it requires the JavaScript runtime Node.js with version 14 or higher, please first check your version with the command
```sh
$ node --version
```
If it reports an error or a version older than 14, please install Node following the official [instructions](https://dolos.ugent.be/guide/installation.html#install-node-js). The platform has been tested with openjdk version 17.0.6, nodejs version v16.19.1, and npm version 8.19.3. 

Then please install dolos with
```sh
$ npm install -g @dodona/dolos
```
Up to the date of release, this will install the latest version of dolos at v2.1.0. You can check the version by 
```sh
$ dolos --version
```
And a sample output looks like this: 
```sh
Dolos v2.1.0
Node v16.19.1
Tree-sitter v0.20.1
```

Please note that the default dolos does not come with cpp parser, so please add the parser following the official [instructions](https://dolos.ugent.be/guide/languages.html#adding-a-new-language). In our setup, you can add it by:
```sh
$ npm install -g tree-sitter-cpp@0.20
```

At last, the environment settings for code generation can vary for different models (e.g., CodeGen, CodeParrot), which are therefore not detailed here. 

# Usage

CodeIPPrompt is an evaluation platform consisting of two major components, prompt construction and code generation model evaluation.

## Prompt Construction

The prompt construction procedure is designed to be generalized to any given (licensed) source code database with programming language of Python, C, C++, C#, and Java. The script prompt_github.py constructs prompts from a given folder containing GitHub repositories. To use it, you need to specify the license and programming language of the target prompts, such as:
```sh
$ python3 prompt_github.py agpl3 python
```

Several sets of prompts derived from real-world licensed code from GitHub can be downloaded at [https://zenodo.org/record/7987148](https://zenodo.org/record/7987148).

## Model Evaluation

To evaluate a given code generation model, please run the model on the constructed prompts. An example of generating programs using CodeParrot is provided in Example_Parrot.py, please replace the paths in your setup. An example is:
```sh
$ python3 Example_Parrot.py codeparrot/codeparrot prompts_agpl3_python_2023-03-27-21-21-29
```

To generate programs faster, give to the third argument the number of prompts to be used for generation at a time.
```
$ python3 Example_Parrot.py codeparrot/codeparrot-small prompts_agpl3_python_2023-03-27-21-21-29 --num_prompts_per_gen 32
```

As far as I can see, with 32GB GPU,

- 32 prompts works for CodeParrot Small
- only 1 prompt works for CodeParrot (meaning it cannot be batched)


Once programs have been generated and saved in the *Programs* directory, run model_eval.py to obtain the results and save them in CSV files. For example:
```sh
$ python3 model_eval.py codegen2Bmulti_agpl3_python_2023-03-27-20-32-30
```

At last, please use results.py to get the final results in terms of *Expected Maximum (EM)* and *Empirical Probability (EP)*. For example:
```sh
$ python3 results.py codegen2Bmulti 
```

## Unlearning
Prepare a dataset and fine-tune CodeParrot (Small) in a similar manner to the [training of CodeParrot](https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/codeparrot_training.py).

```
$ sh prepare_unlearning_datasets.sh
$ python run_unlearning.py --output_dir output/unlearning --per_device_train_batch_size 2 --gradient_accumulation_steps=32 --learning_rate 5e-04 --weight_decay 0.1 --warmup_steps 10 --max_steps 500
```
