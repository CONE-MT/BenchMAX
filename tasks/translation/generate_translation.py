import argparse
import os
import json

from prompt import PROMPTS, LANG_DICT
from models import build_model
from output_parser import build_output_parser
from utils import (
    load_testset, 
    load_unidirectional_testset, 
    post_process, 
    get_directions, 
    handle_arg_string,
    is_unidirectional,
)

def build_prompts(src_dataset, src_lang, tgt_lang, task):
    prompts = []
    src_lang = LANG_DICT[src_lang]
    tgt_lang = LANG_DICT[tgt_lang]
    prompt_template = PROMPTS[task]
    for sample in src_dataset:
        if task == "nexus":
            arguments = "\n".join(sample["kwargs"]["args"])
            if arguments:
                prompt = prompt_template.format(src_lang=src_lang, tgt_lang=tgt_lang, src=sample["text"], args=arguments)
            else:
                prompt = PROMPTS["general"].format(src_lang=src_lang, tgt_lang=tgt_lang, src=sample["text"])
        else:
            prompt = prompt_template.format(src_lang=src_lang, tgt_lang=tgt_lang, src=sample["text"])
        prompts.append(prompt)
    return prompts

def is_done(fpath):
    try:
        with open(fpath, "r", encoding="utf-8") as reader:
            _ = json.load(reader)
        return True
    except:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-lang", "-s", type=str, default=None)
    parser.add_argument("--tgt-lang", "-t", type=str, default=None)
    parser.add_argument("--languages", "-l", type=str, default=None)
    parser.add_argument("--task-name", type=str, required=True, help="The task name of test set")
    parser.add_argument("--model-name", "-m", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--infer-backend", type=str, required=True)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--num-concurrent", type=int, default=1)
    parser.add_argument("--output-parser", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regeneration even if output files exist")
    parser.add_argument("--disable-thinking", action="store_true", help="Disable thinking mode for models like Qwen3")
    args, unknown_args = parser.parse_known_args()
    unknown_args_dict = {unknown_args[i].lstrip('--'): handle_arg_string(unknown_args[i + 1]) for i in range(0, len(unknown_args), 2)}

    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    langs = args.languages
    if langs is None and (src_lang is None or tgt_lang is None):
        raise ValueError("Either --languages or --src-lang and --tgt-lang should be provided")
    if langs is not None and (src_lang is not None or tgt_lang is not None):
        raise ValueError("Only one of --languages or --src-lang and --tgt-lang should be provided")

    task_name = args.task_name
    directions = get_directions(src_lang, tgt_lang, langs)
    unidirectional = is_unidirectional(task_name)

    model_name = os.path.basename(args.model_name)
    output_dir = os.path.join(args.output_dir, model_name, task_name)
    os.makedirs(output_dir, exist_ok=True)
    filtered_directions = []
    for s_l, t_l in directions:
        output_file = f"result_{s_l}-{t_l}.json"
        done = is_done(os.path.join(output_dir, output_file))
        if done and not args.force_regenerate:
            print(f"Skipping {s_l}-{t_l} as the output file already exists.")
        else:
            if args.force_regenerate and done:
                print(f"Regenerating {s_l}-{t_l} as --force-regenerate is enabled.")
            filtered_directions.append((s_l, t_l))
    directions = filtered_directions
    if not directions:
        print("All output files already exist. Exiting.")
        return

    print("Loading test set and building prompts...")
    prompts, sizes = [], []
    src_datasets, tgt_datasets = {}, {}
    if not unidirectional:
        for s_l, t_l in directions:
            if s_l not in src_datasets:
                src_datasets[s_l] = load_testset(task_name, s_l, is_src=True)
            if t_l not in tgt_datasets:
                tgt_datasets[t_l] = load_testset(task_name, t_l)
            prompts.extend(build_prompts(src_datasets[s_l], s_l, t_l, task_name))
        sizes = [len(src_datasets[directions[0][0]])] * len(directions)
    else:
        for s_l, t_l in directions:
            src_dataset, tgt_dataset = load_unidirectional_testset(task_name, s_l, t_l)
            d = f"{s_l}-{t_l}"
            src_datasets[d] = src_dataset
            tgt_datasets[d] = tgt_dataset
            prompts.extend(build_prompts(src_dataset, s_l, t_l, task_name))
            sizes.append(len(src_dataset))

    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

    print(f"Building model {args.model_name}")
    model = build_model(args.model_name, args.infer_backend, args.tokenizer, args.tensor_parallel_size, args.trust_remote_code, args.base_url, args.num_concurrent, **unknown_args_dict)
    model_output_parser = build_output_parser(args.output_parser)

    print("Generating translations...")
    print(messages[0])
    chat_kwargs = {}
    if args.disable_thinking:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    outputs = model.chat(messages, temperature=0.0, max_tokens=args.max_tokens, **chat_kwargs)
    outputs = model_output_parser.parse(outputs)
    post_process(outputs, task_name)

    offset = 0
    for (src_lang, tgt_lang), size in zip(directions, sizes):
        hyps = outputs[offset: offset + size]
        output_file = f"result_{src_lang}-{tgt_lang}.json"
        with open(os.path.join(output_dir, output_file), "w") as f:
            json.dump({"outputs": hyps}, f, indent=4, ensure_ascii=False)
        offset += size


if __name__ == "__main__":
    main()