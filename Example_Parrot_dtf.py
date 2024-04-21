# This is an example script that uses CodeParrot to generate programs using dynamic filtering method for a given set of prompts.
# model options: codeparrot/codeparrot, codeparrot/codeparrot-small

import os
import sys
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess

NUM_PER_PROMPT = 10

from joblib import Parallel, delayed
import tempfile


def check_file(test, source, language):
    dolos_result = subprocess.run([f"node dolos_match_and_score.js {test} {source}"], shell=True, stdout=subprocess.PIPE)
    dolos_score = 0
    try:
        dolos_score = dolos_result.stdout.decode().split("Similarity: ")[1].replace("\n", "")
    except:
        dolos_score = 0
    return dolos_score

def evaluate_similarity(prompt, generated_text, source, language='python'):
    with tempfile.NamedTemporaryFile(mode="w+", suffix='.py') as f:
        f.write(generated_text)
        f.flush()
        dolos_score = check_file(f.name, source, language)
    return float(dolos_score)

def generate_code_with_filtering(model, tokenizer, prompt, source_paths, max_new_tokens=256, similarity_threshold=0.5, chunk_size=50, device='cuda', num_sequences=1):
    if isinstance(prompt, str):
        input_sequences = [prompt] * num_sequences
    elif isinstance(prompt, list):
        input_sequences = []
        for i in prompt:
            input_sequences.extend([i] * num_sequences)

    finished = [0] * len(input_sequences)

    for _ in range(max_new_tokens // chunk_size):
        encoded = tokenizer(input_sequences, return_tensors="pt", padding=True).to(device)
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

        # similarity_scores = [evaluate_similarity(prompt, gen_text, source_paths[idx//num_sequences]) for idx, gen_text in enumerate(generated_texts)]
        similarity_scores = Parallel(n_jobs=-1)(delayed(evaluate_similarity)(
            prompt,
            gen_text,
            source_paths[idx//num_sequences])
            for idx, gen_text in enumerate(generated_texts))

        for i in range(len(input_sequences)):
            if similarity_scores[i] > similarity_threshold:
                # Roll back one token for this sequence
                current_token = outputs['sequences'][i][-1].item()
                while True:
                    # set current token score to -inf
                    outputs['scores'][-1][i][current_token] = -float('inf')

                    # Check if all logits are -inf (no valid tokens left)
                    if torch.all(outputs['scores'][-1][i] == -float('inf')).item():
                        outputs['sequences'][i][-1] = tokenizer.eos_token_id
                        input_sequences[i] = tokenizer.decode(outputs['sequences'][i][:-1], skip_special_tokens=True)
                        finished[i] = 1
                        break

                    # sample new token
                    next_token = torch.multinomial(torch.nn.functional.softmax(outputs['scores'][-1][i], dim=-1), num_samples=1)
                    outputs['sequences'][i][-1] = next_token
                    current_token = next_token.item()

                    # Check the similarity score for the new token
                    new_generated_text = tokenizer.decode(outputs['sequences'][i], skip_special_tokens=True)
                    new_similarity_score = evaluate_similarity(prompt, new_generated_text, source_paths[i//num_sequences])

                    if new_similarity_score <= similarity_threshold:
                        break
            if finished[i]==0:
                input_sequences[i] = tokenizer.decode(outputs['sequences'][i], skip_special_tokens=True)
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
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
