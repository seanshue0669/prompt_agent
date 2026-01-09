# main.py
import os
import shutil
import subprocess
import sys
from urllib.parse import urlparse
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
INPUT_PROMPT_PATH = ROOT_DIR / "user" / "input_prompt.txt"
OUTPUT_PROMPT_PATH = ROOT_DIR / "user" / "output_prompt.txt"
CONTAINERFILE_PATH = ROOT_DIR / "Containerfile"
DEFAULT_IMAGE_NAME = "prompt-agent:local"
DEFAULT_RECURSION_LIMIT = 200
DEFAULT_CONTAINER_BASE_URL = "http://host.containers.internal:11434/v1"
DEFAULT_HOST_BASE_URL = "http://localhost:11434/v1"


def load_input_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input prompt file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Input prompt file is empty: {path}")

    return content


def resolve_base_url(in_container: bool) -> str:
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    if base_url:
        return base_url
    return DEFAULT_CONTAINER_BASE_URL if in_container else DEFAULT_HOST_BASE_URL


def is_connection_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "Connection error" in message
        or "ConnectionError" in message
        or "Failed to connect" in message
    )


def build_llm_client():
    from agentcore import LLMClient

    model = os.getenv("LLM_MODEL", "hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M")
    in_container = os.getenv("PROMPT_AGENT_IN_CONTAINER") == "1"
    base_url = resolve_base_url(in_container).strip()
    api_key = os.getenv("LLM_API_KEY", "")

    if not base_url:
        base_url = None

    try:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    except ValueError:
        temperature = 0.7

    try:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    except ValueError:
        max_tokens = 1000

    llm_config = {
        "model": model,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }

    return LLMClient(
        api_key=api_key,
        base_url=base_url,
        default_config=llm_config,
    )


def run_native() -> int:
    from agents.orchestrator import Orchestrator
    from cli.cli_interface import CLIInterface
    from config.runtime_config import RuntimeConfig

    print("Prompt Optimization Runner")
    print(f"Input prompt:  {INPUT_PROMPT_PATH}")
    print(f"Output prompt: {OUTPUT_PROMPT_PATH}")
    print("Edit the input prompt file before starting.")
    input("Press Enter to start...\n")

    input_prompt = load_input_prompt(INPUT_PROMPT_PATH)

    RuntimeConfig.cli_interface = CLIInterface()

    llm_client = build_llm_client()
    orchestrator = Orchestrator(llm_client, input_prompt)
    compiled = orchestrator.compile()

    recursion_limit = DEFAULT_RECURSION_LIMIT
    env_limit = os.getenv("PROMPT_AGENT_RECURSION_LIMIT", "").strip()
    if env_limit:
        try:
            parsed = int(env_limit)
            if parsed > 0:
                recursion_limit = parsed
        except ValueError:
            pass

    try:
        result = compiled.invoke({}, config={"recursion_limit": recursion_limit})
    except Exception as exc:
        if is_connection_error(exc):
            base_url = resolve_base_url(os.getenv("PROMPT_AGENT_IN_CONTAINER") == "1")
            raise RuntimeError(
                "LLM connection failed. Ensure the LLM server is running and reachable. "
                f"Base URL: {base_url}. If running in Podman, set LLM_BASE_URL to a host "
                "address (not localhost)."
            ) from exc
        raise
    output_prompt = (result.get("current_prompt") or "").strip()
    if not output_prompt:
        raise ValueError("Final output prompt is empty.")

    OUTPUT_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PROMPT_PATH.write_text(output_prompt, encoding="utf-8")
    print(f"\nSaved output to: {OUTPUT_PROMPT_PATH}")
    return 0


def podman_image_exists(podman_path: str, image_name: str) -> bool:
    result = subprocess.run(
        [podman_path, "image", "exists", image_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def build_podman_image(podman_path: str, image_name: str) -> None:
    subprocess.run(
        [
            podman_path,
            "build",
            "-t",
            image_name,
            "-f",
            str(CONTAINERFILE_PATH),
            str(ROOT_DIR),
        ],
        check=True,
    )


def resolve_container_base_url() -> str:
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    if not base_url:
        return DEFAULT_CONTAINER_BASE_URL
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    if hostname in ("localhost", "127.0.0.1"):
        return DEFAULT_CONTAINER_BASE_URL
    return base_url


def run_in_podman(podman_path: str) -> int:
    if not CONTAINERFILE_PATH.exists():
        raise FileNotFoundError(f"Missing Containerfile: {CONTAINERFILE_PATH}")

    image_name = os.getenv("PROMPT_AGENT_IMAGE", DEFAULT_IMAGE_NAME)

    if not podman_image_exists(podman_path, image_name):
        build_podman_image(podman_path, image_name)

    user_dir = INPUT_PROMPT_PATH.parent
    user_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = ROOT_DIR.resolve()
    volume_arg = f"{repo_dir}:/app"
    env_args = ["-e", "PROMPT_AGENT_IN_CONTAINER=1"]
    env_args.extend(["-e", f"LLM_BASE_URL={resolve_container_base_url()}"])
    passthrough_keys = [
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
    ]
    for key in passthrough_keys:
        value = os.getenv(key)
        if value:
            env_args.extend(["-e", f"{key}={value}"])

    subprocess.run(
        [
            podman_path,
            "run",
            "--rm",
            "-it",
            "--add-host",
            "host.containers.internal:host-gateway",
            "-v",
            volume_arg,
            *env_args,
            image_name,
        ],
        check=True,
    )
    return 0


def main() -> int:
    if os.getenv("PROMPT_AGENT_IN_CONTAINER") == "1":
        return run_native()

    if os.getenv("PROMPT_AGENT_NATIVE") == "1":
        return run_native()

    podman_path = shutil.which("podman")
    if podman_path:
        return run_in_podman(podman_path)

    try:
        return run_native()
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python dependencies. Install them or run with Podman."
        ) from exc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print("\nContainer run failed.")
        print("If the error mentions an LLM connection issue, ensure the LLM server is running")
        print("and set LLM_BASE_URL to a host-reachable address (not localhost).")
        print(f"\nError: {exc}")
        sys.exit(1)
    except Exception as exc:
        if is_connection_error(exc):
            print("\nLLM connection failed. Ensure the LLM server is running and reachable.")
        print(f"\nError: {exc}")
        sys.exit(1)
