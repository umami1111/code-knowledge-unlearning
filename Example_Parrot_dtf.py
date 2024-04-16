# This is an example script that uses CodeParrot to generate programs using dynamic filtering method for a given set of prompts.
# model options: codeparrot/codeparrot, codeparrot/codeparrot-small

import os
import sys
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess

NUM_PER_PROMPT = 10

def check_file(test, source, language):
    match_results = "match_" + test.split("/")[-1].split(".")[0] + ".txt"
    subprocess.call("node dolos_match.js " + test + " " + source + " > " + match_results, shell=True)

    file_extension = test.split(".")[-1]
    temp_matched = "temp." + file_extension

    dolos_score = 0

    if os.stat(match_results).st_size == 0:
        print("No matched code snippets")

    else:
        with open(match_results, "r") as f:
            for line in f:
                start_line_number = line.split("matches with")[1].split("}")[0].split("{")[1].split(",")[0]
                end_line_number = line.split("matches with")[1].split("}")[0].split("{")[1].split(",")[1].split("->")[1].replace(" ", "")
                start_line_number = str(int(start_line_number) + 1)
                end_line_number = str(int(end_line_number) + 1)
                # Use sed to get the matched code snippets
                subprocess.call(["sed", "-n", start_line_number + "," + end_line_number + "p", source], stdout=open(temp_matched, "a"))

        dolos_result = subprocess.run(["node", "dolos_score.js", test, temp_matched], stdout=subprocess.PIPE)
        try:
            dolos_score = dolos_result.stdout.decode().split("Similarity: ")[1].replace("\n", "")
        except:
            dolos_score = 0

    subprocess.call("rm -rf " + temp_matched, shell=True)
    subprocess.call("rm -rf " + match_results, shell=True)

    return dolos_score

def evaluate_similarity(prompt, generated_text, source, language='python'):
    # For testing purposes, we'll use a random similarity score
    with open("./gen_temp.py", "w") as f:
        f.write(generated_text)
    dolos_score = check_file("./gen_temp.py", source, language)
    return float(dolos_score)

def generate_code_with_filtering(model, tokenizer, prompt, source_paths, max_new_tokens=256, similarity_threshold=0.5, chunk_size=60, device='cuda', num_sequences=1):
    if isinstance(prompt, str):
        input_sequences = [prompt] * num_sequences
    elif isinstance(prompt, list):
        input_sequences = []
        for i in prompt:
            input_sequences.extend([i] * num_sequences)

    finished = [0] * len(input_sequences)

    for _ in range(max_new_tokens // chunk_size):
        # print(f"Input: {input_sequences}", flush=True)
        encoded = tokenizer(input_sequences, return_tensors="pt", padding=True).to(device)
        # print(encoded)
        outputs = model.generate(
            **encoded,
            max_new_tokens=chunk_size,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_texts = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
        similarity_scores = [evaluate_similarity(prompt, gen_text, source_paths[idx//num_sequences]) for idx, gen_text in enumerate(generated_texts)]
        # print(f"Generated: {generated_texts}", flush=True)
        # print(f"Sim score: {similarity_scores}", flush=True)

        for i in range(len(input_sequences)):
            if similarity_scores[i] > similarity_threshold:
                # Roll back one token for this sequence
                current_token = outputs['sequences'][i][-1].item()
                # print(outputs['scores'][-1][i].unique().shape, flush=True)
                while True:
                    # set current token score to -inf
                    outputs['scores'][-1][i][current_token] = -float('inf')

                    # Check if all logits are -inf (no valid tokens left)
                    if torch.all(outputs['scores'][-1][i] == -float('inf')).item():
                        # Add EOS token and terminate generation for this sequence
                        # print(f"\t [{current_token}] -> [{tokenizer.eos_token_id}]", flush=True)
                        outputs['sequences'][i][-1] = tokenizer.eos_token_id
                        input_sequences[i] = tokenizer.decode(outputs['sequences'][i][:-1], skip_special_tokens=True)
                        finished[i] = 1
                        break

                    # sample new token
                    next_token = torch.multinomial(torch.nn.functional.softmax(outputs['scores'][-1][i], dim=-1), num_samples=1)
                    outputs['sequences'][i][-1] = next_token
                    # print(f"\t [{current_token}] -> [{next_token.item()}]", flush=True)
                    current_token = next_token.item()

                    # Check the similarity score for the new token
                    new_generated_text = tokenizer.decode(outputs['sequences'][i], skip_special_tokens=True)
                    new_similarity_score = evaluate_similarity(prompt, new_generated_text, source_paths[i//num_sequences])

                    if new_similarity_score <= similarity_threshold:
                        break
            if finished[i]==0:
                input_sequences[i] = tokenizer.decode(outputs['sequences'][i], skip_special_tokens=True)
    os.remove("./gen_temp.py")
    return input_sequences

def codeparrot(model, tokenizer, prompts, output_paths, source_paths, num_prompts_per_gen=1):
    if type(prompts) == str:
        assert type(output_paths) == str
        assert type(source_paths) == str
        prompts = [prompts]
        output_paths = [output_paths]
    assert len(prompts) == len(output_paths)
    assert len(prompts) == len(source_paths)

    gen_start_time = datetime.now()

    outputs = generate_code_with_filtering(model=model, tokenizer=tokenizer, prompt=prompts, source_paths=source_paths, device=model.device, num_sequences=NUM_PER_PROMPT)
    print(f"Generated {num_prompts_per_gen} prompts * {NUM_PER_PROMPT} files: {(datetime.now() - gen_start_time).total_seconds()} [sec]")
    for i, output_path in enumerate(output_paths):
        for j in range(NUM_PER_PROMPT):
            output_file = output_path.split('.')[0] + "_" + str(j) + "." + output_path.split('.')[1]
            with open(output_file, 'w') as f:
                f.write(outputs[NUM_PER_PROMPT*i + j])


if __name__=='__main__':

    start_time = datetime.now()

    model_name = sys.argv[1]
    prompt_folder = sys.argv[2]
    num_prompts_per_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    if model_name == "codeparrot/codeparrot":
        model = "CodeParrot"
    elif model_name == "codeparrot/codeparrot-small":
        model = "CodeParrotSmall"
    else:
        print("Usage: python3 Example_Parrot.py codeparrot/codeparrot <prompt_folder> OR python3 Example_Parrot.py codeparrot/codeparrot-small <prompt_folder>")
        exit(1)
    root_path = ""
    prompt_path = root_path + "Prompts" + "/" + prompt_folder + "/"
    program_path = root_path + "Programs" + "/" + prompt_folder.replace("prompts", model) + "/"
    source_path = root_path + "Source" + "/" + prompt_folder.replace("prompts", "source") + "/"
    if not os.path.exists(program_path):
        os.makedirs(program_path)
    # Loop over the programs in the path
    prompts = []
    output_paths = []
    source_paths = []

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map='auto')

    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = 'left'


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
                source_paths.append(source_path + filename)
                #codeparrot(prompt, model_name, program_path + filename)
                if len(prompts) == num_prompts_per_gen:
                    codeparrot(model, tokenizer, prompts, output_paths, source_paths, num_prompts_per_gen=num_prompts_per_gen)
                    prompts = []
                    output_paths = []
                    source_paths = []

    if len(prompts) > 0:
        codeparrot(prompts, model_name, output_paths)

    end_time = datetime.now()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time in hours, minutes, and seconds
    print(f'Time elapsed: {elapsed_time.days} days, {elapsed_time.seconds//3600} hours, {(elapsed_time.seconds//60)%60} minutes, {elapsed_time.seconds%60} seconds')
