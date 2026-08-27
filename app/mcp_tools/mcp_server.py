"""MCP Server registry for Nectar Facility Operations AI Agent.

Registers all MCP tools (telemetry reading + action/service request)
with a centralized MCP server instance.  LLM discovery via tool
descriptions and structured input/output schemas.
"""

from mcp.server import MCPServer
from mcp.types import Tool, ToolAnnotations

from app.mcp_tools.telemetry import (
    get_asset_details,
    get_asset_status,
    get_sensor_data,
    get_energy_consumption,
    get_active_alerts,
    get_asset_relationships,
)
from app.mcp_tools.actions import create_service_request, update_service_request


def _tool(name: str, description: str, fn) -> Tool:
    """Helper to create a Tool instance from a bound function."""
    return Tool(
        name=name,
        title=name.replace("_", " ").title(),
        description=description,
        inputSchema={"type": "object", "properties": {}},
        # We'll let MCP auto-detect structured output where possible
        structured_output=None,
    )


# Pre-register all 8 MCP tools so the server can advertise them.
_MCP_TOOLS: list[Tool] = [
    _tool(
        "get_asset_details",
        "Retrieve detailed information about a facility asset, including "
        "its location, building, model, and equipment specifications. "
        "Use this to look up asset metadata before performing actions.",
        get_asset_details,
    ),
    _tool(
        "get_asset_status",
        "Retrieve the current operational status of a facility asset, "
        "including running state, health state, and current mode. "
        "Use this to check asset health before creating service requests.",
        get_asset_status,
    ),
    _tool(
        "get_sensor_data",
        "Retrieve real-time sensor readings for a facility asset, including "
        "temperature, pressure, airflow, and vibration measurements. "
        "Use this to monitor asset conditions and detect anomalies.",
        get_sensor_data,
    ),
    _tool(
        "get_energy_consumption",
        "Retrieve energy consumption data for a facility asset, including "
        "current power draw (kW), total energy used (kWh), and power usage "
        "description. Use this to track operational costs and efficiency.",
        get_energy_consumption,
    ),
    _tool(
        "get_active_alerts",
        "Retrieve active warning logs and critical alarms for a given asset "
        "or building. Use this to identify outstanding issues that may require "
        "maintenance attention before creating service requests.",
        get_active_alerts,
    ),
    _tool(
        "get_asset_relationships",
        "Retrieve connectivity relationships for a facility asset, including "
        "which AHUs, chillers, and valves are connected. Use this to understand "
        "the system topology before performing maintenance or troubleshooting.",
        get_asset_relationships,
    ),
    _tool(
        "create_service_request",
        "Create a new service maintenance request for a facility asset. "
        "IMPORTANT: If confirmed is False, the tool will NOT create the ticket "
        "but will return a requires_confirmation=True flag with a prompt message. "
        "The caller must confirm with 'YES' to proceed. If confirmed is True, "
        "a unique ticket_id will be generated and the request will be created.",
        create_service_request,
    ),
    _tool(
        "update_service_request",
        "Update an existing service request ticket status and optional notes. "
        "IMPORTANT: This tool requires confirmation before updating active "
        "service tickets. If confirmed is False, the tool will return "
        "requires_confirmation=True with a prompt message. If confirmed is "
        "True, the ticket will be updated with the new status and notes.",
        update_service_request,
    ),
]


def create_mcp_server() -> MCPServer:
    """Build and return an MCPServer instance with all 8 tools registered."""
    server = MCPServer(
        name="nectar-facility-mcp",
        title="Nectar Facility Operations MCP Server",
        description="MCP tool registry for HVAC facility asset management, "
                    "telemetry reading, and maintenance service requests.",
        version="1.0.0",
        tools=_MCP_TOOLS,
    )
    return server