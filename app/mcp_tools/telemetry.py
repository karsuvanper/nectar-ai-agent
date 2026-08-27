from app.mcp_tools.models import TelemetryData


def get_asset_details(asset_id: str) -> dict:
    assets = {
        "Chiller-01": {
            "asset_id": "Chiller-01",
            "location": "Building A - Mechanical Room",
            "building": "Building A",
            "model": "Centrifugal Chiller CT-100",
            "specs": "100 kW cooling capacity, R-134a refrigerant",
        },
        "Chiller-02": {
            "asset_id": "Chiller-02",
            "location": "Building B - Mechanical Room",
            "building": "Building B",
            "model": "Screw Chiller SC-150",
            "specs": "150 kW cooling capacity, R-410A refrigerant",
        },
        "AHU-02": {
            "asset_id": "AHU-02",
            "location": "Building A - North Wing",
            "building": "Building A",
            "model": "Air Handling Unit AHU-5000",
            "specs": "5000 CFM, MERV 13 filters",
        },
        "AHU-03": {
            "asset_id": "AHU-03",
            "location": "Building B - South Wing",
            "building": "Building B",
            "model": "Air Handling Unit AHU-5000",
            "specs": "5000 CFM, MERV 13 filters",
        },
        "Valve-01": {
            "asset_id": "Valve-01",
            "location": "Building A - Chilled Water Loop",
            "building": "Building A",
            "model": "Control Valve VK-200",
            "specs": "2-inch NPS, equal percentage, 24VAC actuator",
        },
        "Valve-02": {
            "asset_id": "Valve-02",
            "location": "Building B - Condenser Water Loop",
            "building": "Building B",
            "model": "Control Valve VK-200",
            "specs": "2-inch NPS, equal percentage, 24VAC actuator",
        },
    }

    if asset_id in assets:
        return assets[asset_id]

    return {"error": f"Asset details unavailable for asset ID: {asset_id}"}


def get_asset_status(asset_id: str) -> dict:
    statuses = {
        "Chiller-01": {
            "operational_status": "Running",
            "running_state": "Enabled",
            "health": "Good",
            "current_mode": "Cooling",
        },
        "Chiller-02": {
            "operational_status": "Standby",
            "running_state": "Disabled",
            "health": "Good",
            "current_mode": "None",
        },
        "AHU-02": {
            "operational_status": "Warning",
            "running_state": "Enabled",
            "health": "Low Airflow",
            "current_mode": "Heating",
        },
        "AHU-03": {
            "operational_status": "Running",
            "running_state": "Enabled",
            "health": "Good",
            "current_mode": "Cooling",
        },
        "Valve-01": {
            "operational_status": "Open",
            "running_state": "Active",
            "health": "Good",
            "current_mode": "Modulating",
        },
        "Valve-02": {
            "operational_status": "Closed",
            "running_state": "Active",
            "health": "Good",
            "current_mode": "Modulating",
        },
    }

    if asset_id in statuses:
        return statuses[asset_id]

    return {"error": f"Asset status unavailable for asset ID: {asset_id}"}


def get_sensor_data(asset_id: str) -> dict:
    sensors = {
        "Chiller-01": {
            "temperature": 7.0,
            "pressure": 120.5,
            "airflow": None,
            "vibration": 2.1,
        },
        "Chiller-02": {
            "temperature": 8.2,
            "pressure": 115.0,
            "airflow": None,
            "vibration": 1.8,
        },
        "AHU-02": {
            "temperature": 22.0,
            "pressure": 65.0,
            "airflow": 85.0,
            "vibration": 3.5,
        },
        "AHU-03": {
            "temperature": 24.0,
            "pressure": 68.0,
            "airflow": 92.0,
            "vibration": 2.0,
        },
        "Valve-01": {
            "temperature": None,
            "pressure": None,
            "airflow": None,
            "vibration": None,
        },
        "Valve-02": {
            "temperature": None,
            "pressure": None,
            "airflow": None,
            "vibration": None,
        },
    }

    if asset_id in sensors:
        return sensors[asset_id]

    return {"error": f"Sensor data unavailable for asset ID: {asset_id}"}


def get_energy_consumption(asset_id: str) -> dict:
    energy = {
        "Chiller-01": {
            "kW": 105.0,
            "kWh": 840.0,
            "power_usage": "105 kW continuous",
        },
        "Chiller-02": {
            "kW": 140.0,
            "kWh": 1120.0,
            "power_usage": "140 kW continuous",
        },
        "AHU-02": {
            "kW": 45.0,
            "kWh": 360.0,
            "power_usage": "45 kW operational",
        },
        "AHU-03": {
            "kW": 48.0,
            "kWh": 384.0,
            "power_usage": "48 kW operational",
        },
        "Valve-01": {
            "kW": 2.0,
            "kWh": 16.0,
            "power_usage": "2 kW standby",
        },
        "Valve-02": {
            "kW": 2.0,
            "kWh": 16.0,
            "power_usage": "2 kW standby",
        },
    }

    if asset_id in energy:
        return energy[asset_id]

    return {"error": f"Energy consumption data unavailable for asset ID: {asset_id}"}


def get_active_alerts(asset_id_or_building: str) -> dict:
    alerts = {
        "Chiller-01": {
            "warnings": [],
            "critical_alarms": [],
        },
        "Chiller-02": {
            "warnings": ["Condenser water temp approaching limit"],
            "critical_alarms": [],
        },
        "AHU-02": {
            "warnings": ["Low airflow detected"],
            "critical_alarms": [],
        },
        "AHU-03": {
            "warnings": [],
            "critical_alarms": [],
        },
        "Valve-01": {
            "warnings": [],
            "critical_alarms": [],
        },
        "Valve-02": {
            "warnings": [],
            "critical_alarms": [],
        },
        "Building A": {
            "warnings": ["AHU-02 low airflow"],
            "critical_alarms": [],
        },
        "Building B": {
            "warnings": ["Chiller-02 condenser warning"],
            "critical_alarms": [],
        },
    }

    if asset_id_or_building in alerts:
        return alerts[asset_id_or_building]

    return {"error": f"Alerts unavailable for: {asset_id_or_building}"}


def get_asset_relationships(asset_id: str) -> dict:
    relationships = {
        "Chiller-01": {
            "connected_ahus": ["AHU-02", "AHU-03"],
            "connected_chillers": [],
            "connected_valves": ["Valve-01"],
        },
        "Chiller-02": {
            "connected_ahus": ["AHU-02", "AHU-03"],
            "connected_chillers": [],
            "connected_valves": ["Valve-02"],
        },
        "AHU-02": {
            "connected_ahus": [],
            "connected_chillers": ["Chiller-01", "Chiller-02"],
            "connected_valves": ["Valve-01"],
        },
        "AHU-03": {
            "connected_ahus": [],
            "connected_chillers": ["Chiller-01", "Chiller-02"],
            "connected_valves": ["Valve-01"],
        },
        "Valve-01": {
            "connected_ahus": ["AHU-02", "AHU-03"],
            "connected_chillers": ["Chiller-01", "Chiller-02"],
            "connected_valves": [],
        },
        "Valve-02": {
            "connected_ahus": [],
            "connected_chillers": ["Chiller-02"],
            "connected_valves": [],
        },
    }

    if asset_id in relationships:
        return relationships[asset_id]

    return {"error": f"Asset relationships unavailable for asset ID: {asset_id}"}