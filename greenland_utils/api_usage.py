import logging
import sys
import boto3
import json
from botocore.config import Config
from datetime import datetime
# from auto_prompt.prompt_bank.factoid_onboarding_final import factoid_onboarding_prompt_final as router_prompt_v0
try:
    from auto_prompt.prompt_bank.mod_onboarding import merchondemand_prompt_candidate as router_prompt_v0
except ImportError:
    # Fallback prompt so the module is importable/runnable without the auto_prompt package.
    router_prompt_v0 = (
        "You are an intent router for an e-commerce assistant. "
        "Given the customer's current query (and any context provided), classify it. "
        "Respond ONLY with a JSON object of the form "
        '{"predicted_intent_llmaj": "<intent>", "reasoning": "<short reason>"}.'
    )
import re

# Disable verbose logging
logging.disable(sys.maxsize)


class IntentRouterLLMaJAgent:

    def __init__(self, prompt=router_prompt_v0, model_id="us.amazon.nova-lite-v1:0",
                 region="us-east-1", profile_name="greenland-dev"):
        """Initialize the router agent with Bedrock client and configuration"""
        self.bedrock_config = Config(
            retries={'total_max_attempts': 3, 'mode': 'standard'}
        )
        # should help with credential error
        self.session = boto3.Session(profile_name=profile_name)
        self.bedrock_client = self.session.client(
            "bedrock-runtime",
            region_name=region,
            config=self.bedrock_config
        )
        self.model_id = model_id
        self.prompt = prompt
        self.conversation_history = []
        self.max_history_turns = 10

    def _prepare_advanced_context(self, customer_query, conversation_history=None,
                                  page_type=None, query_type=None,
                                  response_agent=None, response_text=None):
        """
        Consolidated context preparer that selectively adds sections based on available data.
        """
        messages = []

        # 1. System Prompt
        messages.append({
            "role": "system",
            "content": [{"text": self.prompt}]
        })

        # 2. History
        if conversation_history and len(conversation_history) > 0:
            history_text = "ROUTING HISTORY\n"
            history_text += "Previous queries and their routing results in this conversation:\n"
            for i, turn in enumerate(conversation_history, 1):
                history_text += f"Turn {i}:\n"
                history_text += f"  Query: {turn['query']}\n"
                history_text += f"  Routed to: {turn['intent']}\n"
                history_text += "\n"

            messages.append({
                "role": "user",
                "content": [{"text": history_text}]
            })

        # 3. Construct Context Block
        context_block = ""

        # A. Page Context
        if page_type and str(page_type).upper() not in ["UNKNOWN", "NONE", ""]:
            context_block += f"CURRENT PAGE CONTEXT: {page_type}\n"

        # B. Query Type Context
        if query_type and str(query_type).upper() not in ["UNKNOWN", "NONE", ""]:
            context_block += f"INFERRED QUERY TYPE: {query_type}\n"

        # C. Response Context (Agent + Text pair)
        if response_agent and response_text and str(response_agent).upper() != "NONE":
            context_block += f"PRODUCTION AGENT RESPONSE CONTEXT ({response_agent}): \"{str(response_text)[:300]}...\"\n"

        # D. The Query
        context_block += f"CURRENT QUERY: {customer_query}"

        messages.append({
            "role": "user",
            "content": [{"text": context_block}]
        })

        return messages

    def _should_use_converse_api(self):
        # All Anthropic Claude and Amazon Nova text models use the converse-style
        # path (real converse API when available, else the invoke_model fallback).
        # The "us."/"global." inference-profile prefixes still contain these substrings.
        converse_models = ["anthropic.claude", "amazon.nova"]
        return any(m in self.model_id.lower() for m in converse_models)

    def _call_bedrock_converse_LLMaJ(self, messages):
        """Call Bedrock using the converse API"""
        system_message = None
        converse_messages = []

        for message in messages:
            if message["role"] == "system":
                system_message = message["content"][0]["text"]
            elif message["role"] in ["user", "assistant"]:
                converse_messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })

        converse_params = {
            "modelId": self.model_id,
            "messages": converse_messages,
            "inferenceConfig": {"maxTokens": 1000, "temperature": 0}
        }

        if system_message:
            converse_params["system"] = [{"text": system_message}]

        # Older boto3 (e.g. 1.33.x on Python 3.7) lacks the converse API.
        # Fall back to invoke_model with a provider-specific request body.
        if hasattr(self.bedrock_client, "converse"):
            response = self.bedrock_client.converse(**converse_params)
            return response['output']['message']['content'][0]['text']
        return self._invoke_model_fallback(system_message, converse_messages)

    def _invoke_model_fallback(self, system_message, converse_messages):
        """Replicate converse() via invoke_model for boto3 versions without converse."""
        model_lower = self.model_id.lower()

        if "anthropic" in model_lower:
            messages = []
            for m in converse_messages:
                text = "".join(block.get("text", "") for block in m["content"])
                messages.append({"role": m["role"], "content": text})
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0,
                "messages": messages,
            }
            if system_message:
                body["system"] = system_message
            try:
                resp = self.bedrock_client.invoke_model(
                    modelId=self.model_id, body=json.dumps(body),
                    contentType="application/json", accept="application/json")
            except Exception as e:
                # Newer models (opus-4-7/4-8, ...) reject `temperature`; retry without it.
                if "temperature" in str(e).lower():
                    body.pop("temperature", None)
                    resp = self.bedrock_client.invoke_model(
                        modelId=self.model_id, body=json.dumps(body),
                        contentType="application/json", accept="application/json")
                else:
                    raise
            payload = json.loads(resp["body"].read())
            return "".join(b.get("text", "") for b in payload.get("content", []))

        # Amazon Nova: invoke_model body mirrors the converse request shape.
        body = {
            "messages": converse_messages,
            "inferenceConfig": {"maxTokens": 1000, "temperature": 0},
        }
        if system_message:
            body["system"] = [{"text": system_message}]
        resp = self.bedrock_client.invoke_model(
            modelId=self.model_id, body=json.dumps(body),
            contentType="application/json", accept="application/json")
        payload = json.loads(resp["body"].read())
        return payload["output"]["message"]["content"][0]["text"]

    def _call_bedrock_LLMaJ(self, messages):
        try:
            if self._should_use_converse_api():
                return self._call_bedrock_converse_LLMaJ(messages)
            else:
                raise NotImplementedError("Legacy invoke_model not updated for JSON output")
        except Exception as e:
            print(f"Bedrock API error: {str(e)}")
            raise

    def route_query_LLMaJ(self, customer_query, additional_context=None, conversation_history=None,
                          page_type=None, use_page_type=False,
                          query_type=None, use_query_type=False,
                          response_agent=None, response_text=None, use_response_context=False):
        """
        Route a customer query with flexible context flags.
        """
        try:
            history_to_use = conversation_history if conversation_history is not None else self.conversation_history

            # Prepare arguments based on flags.
            # If flag is False, we pass None to the preparer so it skips that section.
            _p_type = page_type if use_page_type else None
            _q_type = query_type if use_query_type else None
            _r_agent = response_agent if use_response_context else None
            _r_text = response_text if use_response_context else None

            messages = self._prepare_advanced_context(
                customer_query,
                conversation_history=history_to_use,
                page_type=_p_type,
                query_type=_q_type,
                response_agent=_r_agent,
                response_text=_r_text
            )

            # print(messages) ############ TESTING ############

            raw_response = self._call_bedrock_LLMaJ(messages)

            intent = "None"
            reasoning = ""

            try:
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    intent = data.get("predicted_intent_llmaj", "None")
                    reasoning = data.get("reasoning", "")
                else:
                    print(f"Warning: No JSON found in response: {raw_response[:100]}")
                    intent = raw_response.strip()

            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON from: {raw_response[:100]}")
                intent = "Error"

            result = {
                "predicted_intent_llmaj": intent,
                "reasoning": reasoning
            }

            if conversation_history is None:
                self.conversation_history.append({"query": customer_query, "intent": intent})
                if len(self.conversation_history) > self.max_history_turns:
                    self.conversation_history = self.conversation_history[-self.max_history_turns:]

            return result

        except Exception as e:
            print(f"CRITICAL ERROR in route_query: {str(e)}")
            return {"predicted_intent_llmaj": "Error", "reasoning": str(e)}


# Models verified callable on profile "greenland-dev" (account 339712697413, us-east-1)
# via invoke_model on 2026-06-17. IDs use the exact prefix required by Bedrock
# (bare model id where on-demand is supported, "us." cross-region inference profile
# otherwise). Pass any of these as model_id to IntentRouterLLMaJAgent.
AVAILABLE_MODELS = [
    # --- Anthropic Claude ---
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "us.anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-opus-4-7",   # rejects `temperature` (handled by fallback)
    "us.anthropic.claude-opus-4-8",   # rejects `temperature` (handled by fallback)
    # --- Amazon Nova ---
    "amazon.nova-micro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v1:0",
    "us.amazon.nova-premier-v1:0",
    "us.amazon.nova-2-lite-v1:0",
]

# Listed in the account but NOT callable with these credentials:
#   anthropic.claude-fable-5  -> requires non-default data retention mode (zero-retention)
# Non-text models excluded entirely: nova-canvas (image), nova-reel (video),
#   nova-sonic / nova-2-sonic (speech), nova-2-multimodal-embeddings (embeddings).


if __name__ == "__main__":
    import sys

    query = "do you have this shirt in size large?"

    # `python api_usage.py --all` smoke-tests every available model.
    # `python api_usage.py <model_id>` tests one.
    # Default: test Claude Opus 4.5 only.
    if "--all" in sys.argv:
        models = AVAILABLE_MODELS
    elif len(sys.argv) > 1:
        models = [sys.argv[1]]
    else:
        models = ["us.anthropic.claude-opus-4-5-20251101-v1:0"]

    print(f"Query: {query}\n")
    for mid in models:
        agent = IntentRouterLLMaJAgent(
            model_id=mid, region="us-east-1", profile_name="greenland-dev")
        result = agent.route_query_LLMaJ(query)
        print(f"[{mid}]")
        print("  ->", json.dumps(result, ensure_ascii=False))