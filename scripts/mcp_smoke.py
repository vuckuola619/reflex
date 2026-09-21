"""Exercise the running loopback MCP server with the official SDK client."""

import asyncio
import json
from pathlib import Path

from mcp import Client


async def main() -> None:
    async with Client("http://127.0.0.1:8000/mcp") as client:
        result = await client.call_tool(
            "decide_choice",
            {
                "state": "git status --short",
                "question": "How should this command be handled?",
                "options": ["allow", "review", "deny"],
                "risk": 0.1,
            },
        )
        content = result.structured_content
        if not isinstance(content, dict) or content.get("value") != "allow":
            raise RuntimeError("MCP smoke response failed validation")
        Path("artifacts").mkdir(exist_ok=True)
        Path("artifacts/mcp-smoke.json").write_text(json.dumps(content, indent=2))
        print(json.dumps(content, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
