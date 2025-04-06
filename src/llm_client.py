# modules/llm_client.py
from langchain_ollama import OllamaLLM
from omegaconf import DictConfig

def create_llm_client(cfg: DictConfig):
    """Initialize LLM client based on configuration"""
    if cfg.model.type == "ollama":
        client = OllamaLLM(
            model=cfg.model.name,
            temperature=cfg.model.parameters.temperature,
            max_tokens=cfg.model.parameters.max_tokens,
            top_p=cfg.model.parameters.top_p,
            top_k=cfg.model.parameters.top_k,
            repeat_penalty=cfg.model.parameters.repeat_penalty,
            frequency_penalty=cfg.model.parameters.frequency_penalty,
            presence_penalty=cfg.model.parameters.presence_penalty,
            timeout=30
        )
        return client
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")