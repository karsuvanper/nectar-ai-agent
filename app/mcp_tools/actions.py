import re
from datetime import datetime, timezone

from app.mcp_tools.models import MaintenanceRequest


_ticket_counter = 1000


def _generate_ticket_id() -> str:
    global _ticket_counter
    _ticket_counter += 1
    return f"TICK-{_ticket_counter}"


def create_service_request(asset_id: str, issue: str, priority: str = "Medium", confirmed: bool = False) -> MaintenanceRequest:
    if not confirmed:
        return MaintenanceRequest(
            ticket_id="",
            asset_id=asset_id,
            issue_description=issue,
            priority=priority,
            requires_confirmation=True,
            is_created=False,
            message=f"Are you sure you want to create a {priority} priority service request for {asset_id} regarding '{issue}'? Please confirm with YES to proceed.",
        )

    ticket_id = _generate_ticket_id()
    return MaintenanceRequest(
        ticket_id=ticket_id,
        asset_id=asset_id,
        issue_description=issue,
        priority=priority,
        requires_confirmation=False,
        is_created=True,
        message=f"Service request created successfully. Ticket ID: {ticket_id}",
    )


def update_service_request(ticket_id: str, status: str, notes: str = "", confirmed: bool = False) -> dict:
    if not confirmed:
        return {
            "requires_confirmation": True,
            "is_updated": False,
            "message": f"Are you sure you want to update service ticket {ticket_id} to status '{status}'? Please confirm with YES to proceed.",
        }

    return {
        "ticket_id": ticket_id,
        "status": status,
        "notes": notes,
        "is_updated": True,
        "message": f"Service request {ticket_id} updated successfully to status: {status}",
    }