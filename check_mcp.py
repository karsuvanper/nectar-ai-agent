from mcp.server import MCPServer
import inspect
src = inspect.getsource(MCPServer.__init__)
print(src[:2000])