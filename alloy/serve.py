"""OpenAI-compatible HTTP API server for Alloy models.

Usage:
    python -m alloy.serve --model Zyphra/Zamba2-1.2B
    python -m alloy.serve --model Zyphra/Zamba2-1.2B --quantize 4 --port 8000

Endpoints:
    POST /v1/chat/completions
    POST /v1/completions
    GET  /v1/models
    GET  /health
"""

import argparse
import json
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock

import mlx.core as mx


# Global state
_model = None
_tokenizer = None
_model_id = "alloy"
_lock = Lock()


def _generate_response(prompt: str, max_tokens: int, temperature: float,
                        top_p: float, stream: bool):
    """Generate tokens from a prompt."""
    from alloy.generate import stream_generate

    ids = _tokenizer.encode(prompt)
    prompt_ids = mx.array([ids])
    tokens = []

    for token in stream_generate(
        _model, prompt_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    ):
        t = token.item()
        if t == _tokenizer.eos_token_id:
            break
        tokens.append(t)
        if stream:
            yield _tokenizer.decode([t])

    if not stream:
        yield _tokenizer.decode(tokens)


class AlloyHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenAI-compatible API."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_sse(self, data):
        self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length > 0 else {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{
                    "id": _model_id,
                    "object": "model",
                    "owned_by": "alloy",
                }],
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif self.path == "/v1/completions":
            self._handle_completions()
        else:
            self._send_json({"error": "not found"}, 404)

    def _handle_chat_completions(self):
        body = self._read_body()
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        stream = body.get("stream", False)

        # Build prompt using tokenizer's chat template if available
        if hasattr(_tokenizer, 'apply_chat_template'):
            prompt = _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            # Fallback: simple concatenation
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"{content}\n"

        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        t0 = time.time()

        with _lock:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                for chunk_text in _generate_response(prompt, max_tokens, temperature, top_p, stream=True):
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(t0),
                        "model": _model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None,
                        }],
                    }
                    self._send_sse(chunk)

                # Send final chunk
                self._send_sse({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(t0),
                    "model": _model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                })
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                text = next(_generate_response(prompt, max_tokens, temperature, top_p, stream=False))
                self._send_json({
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(t0),
                    "model": _model_id,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": len(_tokenizer.encode(prompt)),
                        "completion_tokens": len(_tokenizer.encode(text)),
                        "total_tokens": len(_tokenizer.encode(prompt)) + len(_tokenizer.encode(text)),
                    },
                })

    def _handle_completions(self):
        body = self._read_body()
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        stream = body.get("stream", False)

        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        t0 = time.time()

        with _lock:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                for chunk_text in _generate_response(prompt, max_tokens, temperature, top_p, stream=True):
                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(t0),
                        "model": _model_id,
                        "choices": [{"text": chunk_text, "index": 0, "finish_reason": None}],
                    }
                    self._send_sse(chunk)

                self._send_sse({
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(t0),
                    "model": _model_id,
                    "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
                })
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                text = next(_generate_response(prompt, max_tokens, temperature, top_p, stream=False))
                self._send_json({
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(t0),
                    "model": _model_id,
                    "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": len(_tokenizer.encode(prompt)),
                        "completion_tokens": len(_tokenizer.encode(text)),
                        "total_tokens": len(_tokenizer.encode(prompt)) + len(_tokenizer.encode(text)),
                    },
                })


def main():
    global _model, _tokenizer, _model_id

    parser = argparse.ArgumentParser(description="Alloy model server (OpenAI-compatible)")
    parser.add_argument("--model", type=str, required=True, help="HF model ID or local path")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=None, help="Quantize bits")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    args = parser.parse_args()

    # Load model
    from alloy.convert_cli import download_model
    from alloy.convert import load_pretrained

    print(f"Loading model: {args.model}")
    model_dir = download_model(args.model)
    _model = load_pretrained(model_dir)
    _model_id = args.model

    if args.bf16:
        print("Converting to bfloat16...")
        _model.to_bfloat16()

    if args.quantize:
        print(f"Quantizing to {args.quantize}-bit...")
        _model.quantize(bits=args.quantize)

    from mlx.utils import tree_flatten
    mem = sum(p.nbytes for _, p in tree_flatten(_model.parameters()))
    print(f"Model ready: {mem / 1e9:.1f} GB")

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except ImportError:
        print("Error: transformers required. pip install transformers")
        sys.exit(1)

    # Start server
    server = HTTPServer((args.host, args.port), AlloyHandler)
    print(f"\nAlloy server running at http://{args.host}:{args.port}")
    print(f"  POST /v1/chat/completions")
    print(f"  POST /v1/completions")
    print(f"  GET  /v1/models")
    print(f"  GET  /health")
    print(f"\nExample:")
    print(f'  curl http://localhost:{args.port}/v1/chat/completions \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model": "{args.model}", "messages": [{{"role": "user", "content": "Hello"}}]}}\'')
    print(f"\nCtrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
