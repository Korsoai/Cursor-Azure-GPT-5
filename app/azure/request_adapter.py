"""Request adaptation helpers for Azure Responses API.

This module defines RequestAdapter, which transforms incoming OpenAI-style
requests into Azure Responses API request parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List

from flask import Request, current_app

from ..exceptions import CursorConfigurationError, ServiceConfigurationError


class RequestAdapter:
    """Handle pre-request adaptation for the Azure Responses API.

    Transforms OpenAI Completions/Chat-style inputs into Azure Responses API
    request parameters suitable for streaming completions in this codebase.
    Returns request_kwargs for requests.request(**kwargs). Also sets
    per-request state on the adapter (model).
    """

    def __init__(self, adapter: Any) -> None:
        """Initialize the adapter with a reference to the AzureAdapter."""
        self.adapter = adapter  # AzureAdapter instance for shared config/env

    # ---- Helpers (kept local to minimize cross-module coupling) ----
    def _copy_request_headers_for_azure(
        self, src: Request, *, api_key: str
    ) -> Dict[str, str]:
        # Don't forward Cursor's headers to Azure — they're for our proxy, not
        # the upstream API. Azure only needs api-key; requests sets Content-Type
        # and Content-Length automatically when using json=.
        return {"api-key": api_key}

    def _messages_to_responses_input_and_instructions(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        instructions_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []

        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role in {"system", "developer"}:
                # content may be None or a list — coerce to string and skip blanks
                if isinstance(content, list):
                    content = " ".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                if content:
                    instructions_parts.append(content)
                continue
            # For user/assistant/tools as inputs
            if role == "tool":
                call_id = m.get("tool_call_id")

                # output must be a plain string; join list content if needed
                if isinstance(content, list):
                    output = " ".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                else:
                    output = content or ""

                item = {
                    "type": "function_call_output",
                    "output": output,
                    "status": "completed",
                    "call_id": call_id,
                }
                input_items.append(item)
            else:
                # Build content array only when there is actual text
                content_array = []
                if content is not None:
                    if isinstance(content, list):
                        # OpenAI multimodal list: convert type names to Responses API types
                        for part in content:
                            if not isinstance(part, dict):
                                continue
                            part_type = part.get("type")
                            if part_type == "text":
                                content_array.append({
                                    "type": "input_text" if role == "user" else "output_text",
                                    "text": part.get("text", ""),
                                })
                            elif part_type == "image_url":
                                url = (part.get("image_url") or {}).get("url", "")
                                content_array.append({
                                    "type": "input_image",
                                    "image_url": url,
                                })
                            else:
                                content_array.append(part)
                    else:
                        content_array = [
                            {
                                "type": "input_text" if role == "user" else "output_text",
                                "text": content,
                            }
                        ]

                item = {
                    "role": role or "user",
                    "content": content_array,
                }
                input_items.append(item)

                if tool_calls := m.get("tool_calls"):
                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        call_id = tool_call.get("id")
                        item = {
                            "type": "function_call",
                            "name": function.get("name"),
                            "arguments": function.get("arguments"),
                            "call_id": call_id,
                        }
                        input_items.append(item)

        instructions = "\n\n".join(instructions_parts) if instructions_parts else None
        return {
            "instructions": instructions,
            "input": input_items if input_items else None,
        }

    def _transform_tools_for_responses(self, tools: Any) -> Any:
        out: List[Dict[str, Any]] = []
        if not isinstance(tools, list):
            current_app.logger.debug(
                "Skipping tool transformation because tools payload is not a list: %r",
                tools,
            )
            return out

        for tool in tools:
            function = tool.get("function")
            if function:
                # Chat Completions format: {"type": "function", "function": {"name": ...}}
                transformed: Dict[str, Any] = {
                    "type": "function",
                    "name": function.get("name"),
                    "description": function.get("description"),
                    "parameters": function.get("parameters"),
                    "strict": False,
                }
            else:
                # Already in Responses API format: pass through as-is
                transformed = dict(tool)
            out.append(transformed)
        return out

    # ---- Main adaptation (always streaming completions-like) ----
    def adapt(self, req: Request) -> Dict[str, Any]:
        """Build requests.request kwargs for the Azure Responses API call.

        Maps inputs to the Responses schema and returns a dict suitable for
        requests.request(**kwargs).
        """
        # Reset per-request state
        self.adapter.inbound_model = None

        # Parse request body — use force=True so we always parse JSON regardless of
        # Content-Type (Cursor may send application/json; charset=utf-8 or similar).
        payload = req.get_json(silent=True, force=True) or {}
        if not payload:
            current_app.logger.warning(
                "Empty or non-JSON request body. "
                "Content-Type: %s | Raw body length: %d bytes",
                req.content_type,
                req.content_length or 0,
            )

        # Determine target model: prefer env AZURE_MODEL/AZURE_DEPLOYMENT
        inbound_model = payload.get("model")
        self.adapter.inbound_model = inbound_model

        settings = current_app.config

        upstream_headers = self._copy_request_headers_for_azure(
            req, api_key=settings["AZURE_API_KEY"]
        )

        # Map Chat/Completions to Responses (always streaming)
        # Cursor sends 'messages' (Chat Completions format) OR 'input' (Responses API
        # format). Only convert when 'messages' is present; otherwise forward 'input'
        # directly so we don't mangle a payload Cursor already formatted correctly.
        messages = payload.get("messages") or []
        if messages:
            responses_body = self._messages_to_responses_input_and_instructions(messages)
            # Use payload-level instructions if none were extracted from system messages
            if responses_body.get("instructions") is None:
                responses_body["instructions"] = payload.get("instructions")
        else:
            # No messages — Cursor is using Responses API format.
            # Pass input (string or array) and instructions through directly.
            responses_body = {
                "input": payload.get("input"),
                "instructions": payload.get("instructions"),
            }

        responses_body["model"] = settings["AZURE_DEPLOYMENT"]

        # Transform tools and tool choice
        responses_body["tools"] = self._transform_tools_for_responses(
            payload.get("tools", [])
        )
        responses_body["tool_choice"] = payload.get("tool_choice")

        responses_body["prompt_cache_key"] = payload.get("user")

        # Always streaming
        responses_body["stream"] = True

        reasoning_effort = (inbound_model or "").replace("gpt-", "").lower()
        if reasoning_effort not in {"high", "medium", "low", "minimal"}:
            reasoning_effort = "high"

        responses_body["reasoning"] = {
            "effort": reasoning_effort,
        }

        # Concise is not supported by GPT-5,
        # but allowing it for now to be able to test it on other models
        if settings["AZURE_SUMMARY_LEVEL"] in {"auto", "detailed", "concise"}:
            responses_body["reasoning"]["summary"] = settings["AZURE_SUMMARY_LEVEL"]
        else:
            raise ServiceConfigurationError(
                "AZURE_SUMMARY_LEVEL must be either auto, detailed, or concise."
                f"\n\nGot: {settings['AZURE_SUMMARY_LEVEL']}"
            )

        # No need to pass verbosity if it's set to medium, as it's the model's default
        if settings["AZURE_VERBOSITY_LEVEL"] in {"low", "high"}:
            responses_body["text"] = {"verbosity": settings["AZURE_VERBOSITY_LEVEL"]}

        responses_body["store"] = False
        responses_body["stream_options"] = {"include_obfuscation": False}

        if settings["AZURE_TRUNCATION"] == "auto":
            responses_body["truncation"] = settings["AZURE_TRUNCATION"]

        request_kwargs: Dict[str, Any] = {
            "method": "POST",
            "url": settings["AZURE_RESPONSES_API_URL"],
            "headers": upstream_headers,
            "json": responses_body,
            "data": None,
            "stream": True,
            "timeout": (60, None),
        }
        return request_kwargs
