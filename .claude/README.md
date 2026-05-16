
## Claude Code Setup (Optional)

If you want to use Claude Code with Google Gemini via LiteLLM:

1.  **Install LiteLLM:**
    ```bash
    pip install litellm
    ```

2.  **Configure LiteLLM for Gemini:**
    Create a `litellm_config.yml` file in the root of your project:
    ```yaml
    # litellm_config.yml
    model_list:
      - model_name: gemini-2.5-flash
        litellm_params:
          model: gemini/gemini-2.5-flash
          api_key: os.environ/GEMINI_API_KEY
    ```

3.  **Get a Gemini API Key:**
    Obtain a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

4.  **Add API Key to .env:**
    Uncomment the commented out lines, and add your Gemini API key to your `.env` file:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

5. **Start the LiteLLM Proxy in your terminal:**
    Run the LiteLLM Proxy within it's own terminal to run alongside your Claude Code session:
    ```bash
    litellm --config litellm_config.yml --port 4000
    ```

6.  **Launch Claude Code in another terminal window:**
    Run the following command in a separate, new terminal window to start using Claude Code with Google Gemini as the model:
    ```bash
    set -a && source .env && set +a 
    claude
    ```
    _You might get a message to use the detected ANTHROPIC API KEY key in the environment (the DUMMY key). Enter "No" to use the correct Google AI Studio API Key_

Now, Claude Code will use Google Gemini (via LiteLLM) for its responses, allowing you to leverage Gemini's capabilities within your development workflow.

