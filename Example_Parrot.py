# This is an example script that uses CodeParrot to generate programs for a given set of prompts.
# model options: codeparrot/codeparrot, codeparrot/codeparrot-small

import os
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


NUM_PER_PROMPT = 10


# def codeparrot(prompts, model_name_or_path, output_paths, num_prompts_per_gen=1):
#     if type(prompts) == str:
#         assert type(output_paths) == str
#         prompts = [prompts]
#         output_paths = [output_paths]
#     assert len(prompts) == len(output_paths)
#     pipe = pipeline("text-generation", model=model_name_or_path, pad_token_id=50256, device=0, batch_size=num_prompts_per_gen)

#     if num_prompts_per_gen > 1 and pipe.model.__class__.__name__.startswith("GPT2"):
#         # IMPORTANT: Change the configuration of tokenizer to make batching work for GPT2
#         # cf.
#         #   https://github.com/huggingface/transformers/issues/21202
#         # Since gpt2 doesn't have a pad_token
#         if not pipe.tokenizer.special_tokens_map.get("pad_token"):
#             pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
#             # Make sure the padding_side is 'left' (if you open gpt2tokenizer you will find that by default
#             # the padding_side is 'right')
#             # cf.
#             #   https://github.com/huggingface/transformers/issues/18478
#             #   https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/2
#             pipe.tokenizer.padding_side = "left" #For BERT like models use "right"

#     gen_start_time = datetime.now()
#     outputs = pipe(prompts, num_return_sequences=NUM_PER_PROMPT, max_new_tokens=256)
#     print(f"Generated {num_prompts_per_gen} prompts * {NUM_PER_PROMPT} files: {(datetime.now() - gen_start_time).total_seconds()} [sec]")
#     for i, output_path in enumerate(output_paths):
#         for j in range(NUM_PER_PROMPT):
#             if "generated_text" in outputs[i][j]:
#                 output_file = output_path.split('.')[0] + "_" + str(j) + "." + output_path.split('.')[1]
#                 with open(output_file, 'w') as f:
#                     f.write(outputs[i][j]["generated_text"])


def create_pipeline(model_name_or_path, num_prompts_per_gen=1):
    #pipe = pipeline("text-generation", model=model_name_or_path, pad_token_id=50256, device=0, batch_size=num_prompts_per_gen)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, pad_token_id=50256, device=0, batch_size=num_prompts_per_gen)

    if num_prompts_per_gen > 1 and pipe.model.__class__.__name__.startswith("GPT2"):
        # IMPORTANT: Change the configuration of tokenizer to make batching work for GPT2
        # cf.
        #   https://github.com/huggingface/transformers/issues/21202
        # Since gpt2 doesn't have a pad_token
        if not pipe.tokenizer.special_tokens_map.get("pad_token"):
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
            # Make sure the padding_side is 'left' (if you open gpt2tokenizer you will find that by default
            # the padding_side is 'right')
            # cf.
            #   https://github.com/huggingface/transformers/issues/18478
            #   https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/2
            pipe.tokenizer.padding_side = "left" #For BERT like models use "right"

    return pipe


def generate_with_pipeline(pipe, prompts, output_paths):
    if type(prompts) == str:
        assert type(output_paths) == str
        prompts = [prompts]
        output_paths = [output_paths]
    assert len(prompts) == len(output_paths)

    gen_start_time = datetime.now()
    outputs = pipe(prompts, num_return_sequences=NUM_PER_PROMPT, max_new_tokens=256)
    print(f"Generated {num_prompts_per_gen} prompts * {NUM_PER_PROMPT} files: {(datetime.now() - gen_start_time).total_seconds()} [sec]")
    for i, output_path in enumerate(output_paths):
        for j in range(NUM_PER_PROMPT):
            if "generated_text" in outputs[i][j]:
                output_file = output_path.split('.')[0] + "_" + str(j) + "." + output_path.split('.')[1]
                with open(output_file, 'w') as f:
                    f.write(outputs[i][j]["generated_text"])


if __name__=='__main__':

    start_time = datetime.now()

    model_name_or_path = sys.argv[1]
    prompt_folder = sys.argv[2]
    num_prompts_per_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    if model_name_or_path == "codeparrot/codeparrot":
        model = "CodeParrot"
    elif model_name_or_path == "codeparrot/codeparrot-small":
        model = "CodeParrotSmall"
    else:
        model = "UnknownModel"
        # print("Usage: python3 Example_Parrot.py codeparrot/codeparrot <prompt_folder> OR python3 Example_Parrot.py codeparrot/codeparrot-small <prompt_folder>")
        # exit(1)
    root_path = ""
    prompt_path = root_path + "Prompts" + "/" + prompt_folder + "/"
    program_path = root_path + "Programs" + "/" + prompt_folder.replace("prompts", model) + "/"
    if not os.path.exists(program_path):
        os.makedirs(program_path)

    pipe = create_pipeline(model_name_or_path, num_prompts_per_gen=num_prompts_per_gen)

    # Loop over the programs in the path
    prompts = []
    output_paths = []
    for filename in os.listdir(prompt_path):
        # Open the file that ends with ".py"
        if filename.endswith(".py"):
            output_path = program_path + filename
            output_file_checklast = output_path.split('.')[0] + "_" + str(NUM_PER_PROMPT - 1) + "." + output_path.split('.')[1]
            if os.path.exists(output_file_checklast):
                print(f"{output_file_checklast} already fully generated, continue to next prompt...")
                continue
            # Open the file
            with open(prompt_path + filename, "r") as f:
                print(f"{filename} undergoing generation...")
                # Read the file
                prompt = f.read()
                prompts.append(prompt)
                output_paths.append(output_path)
                #codeparrot(prompt, model_name_or_path, program_path + filename)
                if len(prompts) == num_prompts_per_gen:
                    #codeparrot(prompts, model_name_or_path, output_paths, num_prompts_per_gen=num_prompts_per_gen)
                    generate_with_pipeline(pipe, prompts, output_paths)
                    prompts = []
                    output_paths = []

    if len(prompts) > 0:
        #codeparrot(prompts, model_name_or_path, output_paths)
        generate_with_pipeline(pipe, prompts, output_paths)

    end_time = datetime.now()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time in hours, minutes, and seconds
    print(f'Time elapsed: {elapsed_time.days} days, {elapsed_time.seconds//3600} hours, {(elapsed_time.seconds//60)%60} minutes, {elapsed_time.seconds%60} seconds')