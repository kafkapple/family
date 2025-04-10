# modules/llm_client.py
from langchain_ollama import OllamaLLM
from omegaconf import DictConfig
import os
from openai import OpenAI
from anthropic import Anthropic
from google.generativeai import GenerativeModel, configure

def create_llm_client(cfg: DictConfig):
    """Initialize LLM client based on configuration"""
    # 공통 파라미터
    common_params = {
        "temperature": cfg.model.parameters.temperature,
        "max_tokens": cfg.model.parameters.max_tokens,
        "top_p": cfg.model.parameters.top_p,
    }

    if cfg.model.type == "ollama":
        # Ollama 특수 파라미터
        ollama_params = {
            "top_k": cfg.model.parameters.ollama.top_k,
            "repeat_penalty": cfg.model.parameters.ollama.repeat_penalty,
            "frequency_penalty": cfg.model.parameters.ollama.frequency_penalty,
            "presence_penalty": cfg.model.parameters.ollama.presence_penalty,
        }
        client = OllamaLLM(
            model=cfg.model.name,
            **common_params,
            **ollama_params,
            timeout=30
        )
        return client
    elif cfg.model.type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        # OpenAI 특수 파라미터
        openai_params = {
            "frequency_penalty": cfg.model.parameters.openai.frequency_penalty,
            "presence_penalty": cfg.model.parameters.openai.presence_penalty,
        }
        client = OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            model=cfg.model.name,
            **common_params,
            **openai_params,
        )
        return client
    elif cfg.model.type == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        # Anthropic 특수 파라미터
        anthropic_params = {
            "top_k": cfg.model.parameters.anthropic.top_k,
            "frequency_penalty": cfg.model.parameters.anthropic.frequency_penalty,
        }
        client = Anthropic(
            api_key=api_key,         
            model=cfg.model.name,
            **common_params,
            **anthropic_params,
            timeout=30
        )
        return client
    elif cfg.model.type == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        configure(api_key=api_key)
        # Gemini 특수 파라미터
        gemini_params = {
            "top_k": cfg.model.parameters.gemini.top_k,
        }
        client = GenerativeModel(
            model_name=cfg.model.name,
            generation_config={
                **common_params,
                **gemini_params,
            }
        )
        return client
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")