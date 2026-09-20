from __future__ import annotations

import argparse
import importlib
import os


def _factory(spec: str):
    if ":" not in spec:
        raise SystemExit("factory must look like module:function")
    module,name=spec.split(":",1)
    fn=getattr(importlib.import_module(module),name)
    engine=fn()
    return engine


def main() -> None:
    p=argparse.ArgumentParser(prog="agent-reflex")
    sub=p.add_subparsers(dest="cmd",required=True)
    serve=sub.add_parser("serve",help="Serve authenticated REST API")
    serve.add_argument("--factory",required=True,help="module:function returning ReflexEngine")
    serve.add_argument("--host",default="127.0.0.1")
    serve.add_argument("--port",type=int,default=8080)
    serve.add_argument("--rpm",type=int,default=600)
    serve.add_argument("--allow-unauthenticated",action="store_true")
    mcp=sub.add_parser("mcp",help="Serve MCP Streamable HTTP")
    mcp.add_argument("--factory",required=True)
    args=p.parse_args()
    engine=_factory(args.factory)
    if args.cmd=="serve":
        try:
            import uvicorn
            from agent_reflex.api import create_app
        except ImportError as exc:
            raise SystemExit("Install agent-reflex[server]") from exc
        app=create_app(engine,requests_per_minute=args.rpm,allow_unauthenticated=args.allow_unauthenticated)
        uvicorn.run(app,host=args.host,port=args.port,proxy_headers=False)
    elif args.cmd=="mcp":
        from agent_reflex.mcp import create_mcp_server
        server=create_mcp_server(engine)
        # MCP v2 serves 2026-07-28 requests statelessly by protocol design.
        # stateless_http=True also makes legacy clients horizontally scalable.
        server.run(transport="streamable-http",stateless_http=True,json_response=True)


if __name__=="__main__":
    main()
