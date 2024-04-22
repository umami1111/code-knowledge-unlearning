# This is an example script that uses CodeParrot to generate programs for a given set of prompts.
# model options: codeparrot/codeparrot, codeparrot/codeparrot-small

import argparse
import os
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


NUM_PER_PROMPT = 10


def create_pipeline(model_name_or_path, num_prompts_per_gen=1):
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


def generate_with_pipeline(pipe, prompts, output_paths, num_prompts_per_gen):
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

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path", type=str)
    parser.add_argument("prompt_folder", type=str, help="Name of prompt directory in Prompts")
    parser.add_argument("--output_prefix", type=str, default=None, help="Prefix of output directory created under Programs")
    parser.add_argument("--num_prompts_per_gen", type=int, default=8, help="Number of prompts bathed in a single generation")
    args = parser.parse_args()
    print(args)

    if args.model_name_or_path == "codeparrot/codeparrot":
        output_prefix = "CodeParrot"
    elif args.model_name_or_path == "codeparrot/codeparrot-small":
        output_prefix = "CodeParrotSmall"
    else:
        assert args.output_prefix != None
        output_prefix = args.output_prefix

    root_path = ""
    prompt_path = root_path + "Prompts" + "/" + args.prompt_folder + "/"
    program_path = root_path + "Programs" + "/" + args.prompt_folder.replace("prompts", output_prefix) + "/"
    if not os.path.exists(program_path):
        os.makedirs(program_path)

    pipe = create_pipeline(args.model_name_or_path, num_prompts_per_gen=args.num_prompts_per_gen)

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
                if len(prompts) == args.num_prompts_per_gen:
                    generate_with_pipeline(pipe, prompts, output_paths, args.num_prompts_per_gen)
                    prompts = []
                    output_paths = []

    if len(prompts) > 0:
        generate_with_pipeline(pipe, prompts, output_paths, args.num_prompts_per_gen)

    end_time = datetime.now()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time in hours, minutes, and seconds
    print(f'Time elapsed: {elapsed_time.days} days, {elapsed_time.seconds//3600} hours, {(elapsed_time.seconds//60)%60} minutes, {elapsed_time.seconds%60} seconds')