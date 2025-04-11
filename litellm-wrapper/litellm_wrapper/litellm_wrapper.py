import functools
import logging
import os
import threading
import ujson
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LiteLLM Configuration ---
try:
    # Suppress UserWarning during litellm import if possible
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module='litellm')
        if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True" # Set before importing litellm
        import litellm

    litellm.drop_params = True  # Drop unsupported parameters automatically
    litellm.telemetry = False   # Disable telemetry

    # Configure LiteLLM disk caching
    disk_cache_dir = os.environ.get("LITELLM_CACHE_DIR", Path.home() / ".litellm_cache")
    try:
        # This requires the 'diskcache' package to be installed (pip install diskcache)
        litellm.cache = litellm.Cache(type="disk", disk_cache_dir=str(disk_cache_dir))
        logger.info(f"LiteLLM disk cache enabled at: {disk_cache_dir}")
    except ModuleNotFoundError:
        logger.error("Failed to initialize LiteLLM disk cache: The 'diskcache' package is not installed. Please run `pip install diskcache`.")
        litellm.cache = None # Disable cache if dependency is missing
    except Exception as e:
        logger.error(f"Failed to initialize LiteLLM disk cache: {e}")
        litellm.cache = None # Disable cache if initialization fails

except ImportError:
    logger.error("The LiteLLM package is not installed. Please run `pip install litellm`.")
    # Define a placeholder if litellm is essential for the script to even load
    class LitellmPlaceholder:
        def __getattr__(self, name):
            raise ImportError(
                "LiteLLM is not installed or failed to import."
            )
    litellm = LitellmPlaceholder()
except Exception as e:
    logger.error(f"An unexpected error occurred during LiteLLM setup: {e}")
    raise # Re-raise unexpected errors


# --- Constants ---
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1500


# --- Helper Functions for History Inspection ---
def _color_text(text: str, color_code: str, end: str = "\n") -> str:
    """Adds ANSI color codes to text."""
    return f"\x1b[{color_code}m{str(text)}\x1b[0m{end}"

def _green(text: str, end: str = "\n") -> str:
    """Formats text in green."""
    return _color_text(text, "32", end)

def _red(text: str, end: str = "\n") -> str:
    """Formats text in red."""
    return _color_text(text, "31", end)


# --- Main Wrapper Class ---
class LitellmModel:
    """
    A wrapper class for making calls to various LLMs using LiteLLM.

    Handles API requests, caching (via litellm's cache), token usage tracking,
    and history inspection.

    Attributes:
        model (str): The identifier for the model in litellm format (e.g., "openai/gpt-4o-mini", "groq/llama3-8b-8192").
        api_key (Optional[str]): Specific API key to use. If None, litellm will try environment variables.
        api_base (Optional[str]): Specific API base URL. If None, litellm will try environment variables or defaults.
        default_params (dict): Default parameters for litellm.completion calls (e.g., temperature, max_tokens).
        use_cache (bool): Whether to use litellm's caching by default for calls made by this instance.
        history (List[Dict]): A list storing records of past API calls and responses.
        prompt_tokens (int): Cumulative count of prompt tokens used by this instance.
        completion_tokens (int): Cumulative count of completion tokens used by this instance.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        use_cache: bool = True,
        **kwargs,
    ):
        """
        Initializes the LitellmModel instance.

        Args:
            model: The litellm model string.
            api_key: Optional API key.
            api_base: Optional API base URL.
            temperature: Default temperature for generations.
            max_tokens: Default maximum tokens for generations.
            use_cache: Default cache usage preference for calls from this instance.
            **kwargs: Additional default parameters to pass to litellm.completion.
                      These will be merged with/override temperature and max_tokens.
        """
        if not hasattr(litellm, "completion"):
             raise ImportError("LiteLLM is not installed or failed to import correctly.")

        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.default_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        self.use_cache = use_cache
        self.history: List[Dict[str, Any]] = []

        # Thread-safe token counting
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

        # Special check for OpenAI o1 models based on original comment
        if "o1-" in model:
            o1_temp = self.default_params.get("temperature")
            o1_max_tokens = self.default_params.get("max_tokens")
            if not (o1_temp == 1.0 and o1_max_tokens is not None and o1_max_tokens >= 5000):
                 logger.warning(
                     "OpenAI's o1-* models may require temperature=1.0 and max_tokens >= 5000. "
                     f"Current defaults: temperature={o1_temp}, max_tokens={o1_max_tokens}"
                 )

    def _prepare_messages(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Validates and prepares the messages list."""
        if messages is not None and prompt is not None:
            raise ValueError("Provide either 'prompt' or 'messages', not both.")
        if messages is None and prompt is None:
            raise ValueError("Must provide either 'prompt' or 'messages'.")

        if prompt is not None:
            # Convert a single prompt string to the standard messages format
            return [{"role": "user", "content": prompt}]
        elif messages is not None:
            # Ensure messages is a list of dicts with 'role' and 'content'
            if not isinstance(messages, list) or not all(
                isinstance(m, dict) and "role" in m and "content" in m for m in messages
            ):
                raise ValueError("'messages' must be a list of dictionaries, each with 'role' and 'content' keys.")
            return messages
        else:
             # Should be unreachable due to earlier checks, but provides explicit default
             return []

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> List[str]:
        """
        Makes a call to the LLM using litellm.completion.

        Args:
            prompt: A single user prompt string. Mutually exclusive with 'messages'.
            messages: A list of message dictionaries (e.g., [{"role": "user", "content": "..."}, ...]).
                      Mutually exclusive with 'prompt'.
            use_cache: Override the instance's default cache usage for this specific call.
                       If None, uses the instance's self.use_cache setting.
            **kwargs: Additional parameters to pass to litellm.completion, overriding defaults.

        Returns:
            A list of response strings (content of the generated messages). Typically contains one response,
            but can contain more if 'n' > 1 is requested.

        Raises:
            ValueError: If input arguments are invalid.
            litellm.exceptions.*: If the API call fails after retries.
        """
        prepared_messages = self._prepare_messages(prompt, messages)

        # Combine default params, call-specific kwargs, and essential info
        call_kwargs = {
            **self.default_params,
            **kwargs,
            "model": self.model,
            "messages": prepared_messages,
        }

        # Handle API key and base if provided explicitly for the instance
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        if self.api_base:
            call_kwargs["api_base"] = self.api_base

        # Determine caching for this call
        should_cache = self.use_cache if use_cache is None else use_cache

        # Prepare completion arguments, starting with the base ones from call_kwargs
        completion_args = {**call_kwargs}

        # --- ** FIX APPLIED HERE ** ---
        # Only explicitly pass the 'cache' parameter if we want to DISABLE caching.
        # Otherwise, let litellm use its global default setting (litellm.cache).
        # Avoids passing cache=None which caused the AttributeError in some litellm versions.
        if not should_cache:
            completion_args['cache'] = {"no-cache": True, "no-store": True}
            # logger.debug("LiteLLM Caching explicitly disabled for this call.") # Optional debug log
        # else:
            # logger.debug("LiteLLM Caching enabled for this call (using global settings).") # Optional debug log
        # --- ** END OF FIX ** ---

        response = None
        response_content = []
        response_dict_for_history = {} # Store response data for history even if errors occur
        usage_data = {}

        try:
            # Make the API call via litellm using the prepared args
            # Now 'cache' key is only present in completion_args if disabling cache
            logger.debug(f"Calling litellm.completion with args: { {k:v for k,v in completion_args.items() if k != 'messages'} }") # Log args except long messages
            response = litellm.completion(**completion_args)

            # Process successful response
            response_dict_for_history = response.dict() # Convert ModelResponse to dict for easier storage/serialization
            response_content = [
                choice.message.content
                for choice in response.choices
                if choice.message and choice.message.content is not None
            ]
            usage_data = response_dict_for_history.get("usage", {})
            self._log_usage(usage_data) # Log token usage on success

        except Exception as e:
            logger.error(f"LiteLLM completion call failed: {e}", exc_info=True) # Log traceback
            # Attempt to get usage data even from exceptions if available (some errors might include it)
            if hasattr(e, 'response') and hasattr(e.response, 'json'): # Check if error response is json parsable
                try:
                    error_data = e.response.json()
                    usage_data = error_data.get("usage", {})
                    response_dict_for_history = error_data # Store error details
                except Exception as json_e:
                    logger.warning(f"Could not parse usage data from error response JSON: {json_e}")
                    response_dict_for_history = {"error": str(e), "type": type(e).__name__, "raw_response": str(getattr(e, 'response', None))}
            else:
                # Store minimal info if response object itself is the error or not json
                 response_dict_for_history = {"error": str(e), "type": type(e).__name__}
            # Re-raise the exception after logging and attempting history update
            raise e

        finally:
            # Log history regardless of success or failure
            # Remove sensitive info like API key from logged kwargs
            kwargs_for_history = {k: v for k, v in completion_args.items() if k not in ["api_key", "api_base"]}
            # Messages are stored separately, remove from kwargs dict to avoid duplication
            if 'messages' in kwargs_for_history:
                del kwargs_for_history['messages']
            # Cache dict is internal detail, potentially remove from history kwargs
            if 'cache' in kwargs_for_history:
                del kwargs_for_history['cache']


            history_entry = {
                "prompt": prompt, # Keep original prompt if provided
                "messages": prepared_messages,
                "kwargs_used": kwargs_for_history, # Log the actual final kwargs used (excluding sensitive/redundant)
                "response": response_dict_for_history, # Store dict representation
                "outputs": response_content,
                "usage": usage_data,
                # Cost might be available in response_dict_for_history if litellm provides it
                "cost": response_dict_for_history.get("_hidden_params", {}).get("response_cost"),
                "cached_response": response_dict_for_history.get("_hidden_params",{}).get("cached_response", False) if response else None
            }
            self.history.append(history_entry)

        return response_content

    def _log_usage(self, usage_data: Dict[str, int]):
        """Safely increments token counters based on usage data."""
        if usage_data and isinstance(usage_data, dict):
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)
            # Ensure tokens are integers before adding
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                 with self._token_usage_lock:
                    self.prompt_tokens += prompt_tokens
                    self.completion_tokens += completion_tokens
            else:
                logger.warning(f"Invalid token usage data types received: {usage_data}")


    def get_usage_and_reset(self) -> Dict[str, Dict[str, int]]:
        """
        Retrieves the total token usage tracked by this instance and resets the counters.

        Returns:
            A dictionary containing the token counts for the model used by this instance.
            Example: {"openai/gpt-4o-mini": {"prompt_tokens": 100, "completion_tokens": 200}}
        """
        with self._token_usage_lock:
            usage = {
                self.model: {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                }
            }
            self.prompt_tokens = 0
            self.completion_tokens = 0
        return usage

    def inspect_history(self, n: int = 1, show_hidden: bool = False):
        """
        Prints the last n entries of the call history for inspection.

        Args:
            n: The number of history entries to display.
            show_hidden: If True, includes internal litellm params (_hidden_params) in the output.
        """
        if not self.history:
            print("History is empty.")
            return

        if n > len(self.history):
            n = len(self.history)
            print(f"Showing all {n} entries in history.")
        elif n <= 0:
            print("Specify a positive number for n.")
            return

        print(f"\n--- Inspecting Last {n} History Entries ---")

        # Iterate backwards through the history list
        for i, item in enumerate(reversed(self.history[-n:])):
            print("\n" + "="*80)
            print(f"History Entry {len(self.history) - i} (Most Recent = 1)")

            # Display Input Messages
            print(_red("Input Messages:"))
            messages = item.get("messages", [])
            if not messages and item.get("prompt"): # Handle case where only prompt was logged
                messages = [{"role": "user", "content": item["prompt"]}]

            for j, msg in enumerate(messages):
                role = msg.get('role', 'unknown').capitalize()
                content = msg.get('content', '')
                print(f"  ({j+1}) Role: {role}")
                print(f"      Content: {content.strip()}")

            # Display Call Kwargs Used (excluding messages, api_key, api_base, cache)
            print(_red("Call Kwargs Used:"))
            kwargs_used = item.get("kwargs_used", {})
            if kwargs_used:
                for key, value in kwargs_used.items():
                     print(f"  {key}: {value}")
            else:
                 print("  (No kwargs recorded)")


            # Display Response Status (Cached/Error)
            response_data = item.get("response", {})
            is_cached = item.get("cached_response", False)
            error_info = response_data.get("error") if isinstance(response_data, dict) else None

            status_color = _green if error_info is None else _red
            status_text = "Response"
            if is_cached:
                status_text += " (Cached)"
            if error_info:
                 status_text += f" (Error: {response_data.get('type', 'Unknown')})"


            print(status_color(f"{status_text}:"))

            # Display Outputs (if successful)
            outputs = item.get("outputs", [])
            if outputs:
                 for j, output in enumerate(outputs):
                    print(_green(f"  Output {j+1}:\n    {output.strip()}", end="\n\n"))
            elif error_info:
                 print(_red(f"  Error Details: {error_info}"))
                 # Optionally show raw response if available in case of error
                 raw_resp = response_data.get("raw_response")
                 if raw_resp:
                     print(_red(f"  Raw Error Response Snippet: {str(raw_resp)[:200]}...")) # Show snippet
            elif isinstance(response_data, dict) and not error_info:
                 print(_green("  (No text output found in response choices, check raw response if needed)"))


            # Display Usage and Cost
            usage = item.get("usage")
            cost = item.get("cost")
            usage_cost_str = "  Usage: "
            if usage and isinstance(usage, dict):
                usage_cost_str += f"Prompt Tokens: {usage.get('prompt_tokens', 'N/A')}, "
                usage_cost_str += f"Completion Tokens: {usage.get('completion_tokens', 'N/A')}, "
                usage_cost_str += f"Total Tokens: {usage.get('total_tokens', 'N/A')}"
            else:
                usage_cost_str += "N/A"

            if cost is not None:
                 usage_cost_str += f" | Cost: ${cost:.6f}"
            print(usage_cost_str)


            # Display Raw Response (optional)
            if show_hidden and isinstance(response_data, dict):
                hidden_params = response_data.get("_hidden_params")
                if hidden_params:
                    print(_red("  LiteLLM Hidden Params:"))
                    try:
                        import pprint
                        print(f"    {pprint.pformat(hidden_params)}")
                    except ImportError:
                        print(f"    {hidden_params}") # Fallback if pprint not available


            print("="*80)

        print("\n--- End of History Inspection ---")