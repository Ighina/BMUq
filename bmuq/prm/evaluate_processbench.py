import os
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool
from openai import OpenAI
from datasets import load_dataset

# Global variables for multiprocessing
MODEL = None
CLIENT = None
USE_BERT = True


def init_worker(use_bert):
    """Initialize global variables in each worker process"""
    global MODEL, CLIENT, USE_BERT
    USE_BERT = use_bert

    if use_bert:
        from bmuq.prm.inference import InferenceBertForTokenClassificationWithEmbeddings

        pretrained_model = "bmuq/prm/prm_mpnetv2_2410/final_model"
        featurizer_model = "sentence-transformers/all-mpnet-base-v2"
        MODEL = InferenceBertForTokenClassificationWithEmbeddings(
            pretrained_model=pretrained_model,
            featurizer_model=featurizer_model,
        )
        MODEL.to("mps")
    else:
        CLIENT = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")


def single_process(d):
    global MODEL, CLIENT, USE_BERT
    # print(MODEL)
    steps = d["steps"]
    messages = []
    if USE_BERT:
        input_text = [d["problem"]] + steps
        output = MODEL(inputs_text=input_text)
        probs = output[0]  # List of probabilities for each step

        for sdx, prob in enumerate(probs[1:]):
            if prob[1] + prob[2] < 0.5:
                return sdx
    else:
        for sdx, step in enumerate(steps):
            if sdx == 0:
                messages.append(
                    {"role": "user", "content": d["problem"] + "\n\n" + step}
                )
            else:
                messages.append({"role": "user", "content": step})
            completion = CLIENT.chat.completions.create(
                model="Llama3.1-8B-PRM-Mistral-Data",
                messages=messages,
                n=1,
                temperature=0.0,
                max_tokens=1,
            )
            judgment = (
                completion.choices[0].message.content.strip().lower().startswith("+")
            )
            if not judgment:
                return sdx
            messages.append({"role": "assistant", "content": "+"})
    return -1


def main(use_bert=True):
    global MODEL, CLIENT, USE_BERT
    USE_BERT = use_bert

    if use_bert:
        from bmuq.prm.inference import InferenceBertForTokenClassificationWithEmbeddings

        pretrained_model = "bmuq/prm/prm_mpnetv2_2410/final_model"
        featurizer_model = "sentence-transformers/all-mpnet-base-v2"

        MODEL = InferenceBertForTokenClassificationWithEmbeddings(
            pretrained_model=pretrained_model,
            featurizer_model=featurizer_model,
        )
        MODEL.to("mps")
        os.makedirs("outputs/PRM-BERT", exist_ok=True)
    else:
        CLIENT = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")
        os.makedirs("outputs/Llama3.1-8B-PRM-Mistral-Data", exist_ok=True)

    configs = ["gsm8k", "math", "olympiadbench", "omnimath"]
    for config in configs:
        input_data = load_dataset("Qwen/ProcessBench", split=config)
        with Pool(4, initializer=init_worker, initargs=(use_bert,)) as p:
            predictions = list(
                tqdm(
                    p.imap(single_process, input_data),
                    total=len(input_data),
                    desc=f"Processing {config}",
                    dynamic_ncols=True,
                )
            )

        res_data = []
        for idx, d in enumerate(input_data):
            new_d = d.copy()
            new_d["prediction"] = predictions[idx]
            new_d["match"] = predictions[idx] == d["label"]
            res_data.append(new_d)

        data1 = [e for e in res_data if e["label"] != -1]
        data2 = [e for e in res_data if e["label"] == -1]
        outdir = (
            "outputs/Llama3.1-8B-PRM-Mistral-Data"
            if not use_bert
            else "outputs/PRM-BERT"
        )

        with open(f"{outdir}/{config}_error.jsonl", "w") as f:
            for e in data1:
                f.write(json.dumps(e) + "\n")
        with open(f"{outdir}/{config}_correct.jsonl", "w") as f:
            for e in data2:
                f.write(json.dumps(e) + "\n")

        acc1 = np.mean([e["match"] for e in data1]) * 100
        acc2 = np.mean([e["match"] for e in data2]) * 100
        f1 = 2 * acc1 * acc2 / (acc1 + acc2)
        print(f"{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}")


if __name__ == "__main__":
    main(use_bert=True)
