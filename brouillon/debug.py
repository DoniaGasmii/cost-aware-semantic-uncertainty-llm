from llama_cpp import Llama
import numpy as np

llm = Llama(
    model_path="models/qwen2.5-7b-instruct-q4_k_m.gguf",
    n_ctx=8192,            # or higher if you need it
    logits_all=True        # <-- REQUIRED for logprobs
)

prompt = "The capital of France is"
res = llm.create_completion(
    prompt=prompt,
    max_tokens=1,
    temperature=0,
    logprobs=50,           # top-50 token logprobs
    echo=True              # include prompt tokens in the output
)

choice = res["choices"][0]
top = choice["logprobs"]["top_logprobs"][-1]   # dict: token -> logprob for next token
probs = {tok: float(np.exp(lp)) for tok, lp in top.items()}
print(sorted(probs.items(), key=lambda x: -x[1])[:10])
