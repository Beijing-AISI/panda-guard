1. hf needs acclerate
2. auto download hf
3. vllm needs setuptools
4. for SGLang:
    * uv pip install "sglang[all]>=0.4.6.post3"
    * uv pip install "patch==1.*"
    * need set CUDA_HOME
5. for Ollama:
    * curl -fsSL https://ollama.com/install.sh | sh
    * uv pip install ollama