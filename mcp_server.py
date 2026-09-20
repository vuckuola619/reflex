from examples.server_app import build_engine
from agent_reflex.mcp import create_mcp_server

mcp=create_mcp_server(build_engine())

if __name__=="__main__":
    mcp.run(transport="streamable-http",stateless_http=True,json_response=True)
