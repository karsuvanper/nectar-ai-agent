from pydantic import BaseModel, Field


class TelemetryData(BaseModel):
    asset_id: str = Field(description="Unique identifier for the facility asset")
    temperature_c: float = Field(description="Temperature in Celsius")
    pressure_psi: float = Field(description="Pressure in psi")
    energy_kw: float = Field(description="Energy consumption in kW")
    status: str = Field(description="Operational status of the asset")
    timestamp: str = Field(description="ISO format timestamp")


class MaintenanceRequest(BaseModel):
    ticket_id: str = Field(description="Unique ticket identifier")
    asset_id: str = Field(description="Asset associated with the maintenance request")
    issue_description: str = Field(description="Description of the issue")
    priority: str = Field(description="Priority level (Low, Medium, High, Critical)")
    requires_confirmation: bool = Field(description="Whether confirmation is required")
    is_created: bool = Field(description="Whether the ticket was created")
    message: str = Field(description="Status message")