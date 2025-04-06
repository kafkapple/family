# integrations/llm_interface.py
import logging
from omegaconf import DictConfig
# Assuming llm_client object has a __call__ method or similar for prediction

logging.basicConfig(level=logging.INFO) # Basic config
logger = logging.getLogger(__name__)

def get_llm_response(prompt: str,  llm_client) -> str | None:
    """Sends prompt to the LLM client and returns the response."""
    response = None # Default to None
    try:
        # Adjust this call based on the actual LLM client library's API.
        # For langchain_ollama.OllamaLLM, it's likely callable directly.
        prompt_snippet = prompt[:100].replace('\n', ' ')
        if len(prompt) > 100:
            prompt_snippet += '...'
        logger.debug(f"Sending prompt: {prompt_snippet}")

        # Call the llm_client (assumed to be callable)
        response_data = llm_client(prompt) # Assuming this returns the string directly

        # Ensure response is a string
        if isinstance(response_data, str):
            response = response_data
            response_snippet = response[:100].replace('\n', ' ') + ('...' if len(response) > 100 else '') if response else "None"
            logger.debug(f"Received response: {response_snippet}")
        else:
            logger.warning(f"LLM client returned unexpected type: {type(response_data)}. Expected str.")
            response = None # Treat non-string response as error/None

    except Exception as e:
        logger.error(f"Error getting LLM response: {e}", exc_info=True)
        response = None # Ensure response is None on error

    return response 