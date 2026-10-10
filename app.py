from functools import lru_cache
from pathlib import Path

import gradio as gr
import spaces
import joblib
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from features import prompt_features
from build_model import load_model

artifact = load_model()

@lru_cache(maxsize=2)
def tokenizer(model):
    return AutoTokenizer.from_pretrained(artifact[model.lower() + '_model'], use_fast=True)

@spaces.GPU(duration=30)
def predict(prompt, model, max_new_tokens):
    if not prompt or not prompt.strip():
        raise gr.Error('Enter a prompt first.')
    if len(prompt) > 50000:
        raise gr.Error('Please limit the prompt to 50,000 characters.')
    if model not in ('Mistral', 'Qwen', 'Compare both'):
        raise gr.Error('Choose Mistral, Qwen, or Compare both.')
    if not float(max_new_tokens).is_integer() or not 1 <= max_new_tokens <= 4096:
        raise gr.Error('Output token limit must be an integer between 1 and 4096.')
    models = ['Mistral', 'Qwen'] if model == 'Compare both' else [model]
    output = pd.DataFrame({'Resource metric': artifact['targets'], 'Unit': ['ms', 'ms', 'MB', 'MB', 'MB', 'tokens/s']})
    notes = []
    for name in models:
        ids = tokenizer(name)(prompt, add_special_tokens=True)['input_ids']
        features = prompt_features(prompt)
        features.update(model=name, input_tokens_actual=min(len(ids),4096), max_new_tokens=int(max_new_tokens))
        values = artifact['predictor'].predict(pd.DataFrame([features])[artifact['features']])[0]
        if not np.isfinite(values).all():
            raise gr.Error('The predictor returned an invalid estimate.')
        output[name] = values.round(3)
        notes.append(f'{name}: {len(ids)} input tokens' + (' (token count capped at 4096, matching notebook)' if len(ids)>4096 else ''))
    return output, '\n\n'.join(notes)

with gr.Blocks(title='PromptScope') as demo:
    gr.Markdown('# PromptScope\nEstimate the resources needed to process a prompt with Mistral or Qwen.')
    gr.Markdown('These are ML estimates based on the original benchmark hardware and configuration. They are not live measurements or GPU/CPU utilization percentages. No LLM generation is performed.')
    prompt = gr.Textbox(label='Your prompt', lines=7, placeholder='Explain binary search and provide a Java implementation.')
    with gr.Row():
        model = gr.Dropdown(['Mistral', 'Qwen', 'Compare both'], value='Compare both', label='Model')
        tokens = gr.Number(value=300, precision=0, minimum=1, maximum=4096, label='Maximum new tokens')
    button = gr.Button('Estimate resources', variant='primary')
    result = gr.Dataframe(label='Predicted resource usage', interactive=False)
    details = gr.Markdown()
    button.click(predict, [prompt, model, tokens], [result, details], api_name='predict_resources')
    gr.Examples([['Explain binary search and provide an efficient Java implementation.', 'Compare both', 300]], [prompt, model, tokens])

if __name__ == '__main__':
    demo.queue(max_size=20).launch(server_name='0.0.0.0', server_port=7860)
