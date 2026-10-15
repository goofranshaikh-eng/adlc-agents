import re
import logging
from typing import Optional
from agent.src.mcp_client import MCPClient

logger = logging.getLogger(__name__)


class AgentCore:
    def __init__(self, mcp_client: MCPClient):
        self._mcp_client = mcp_client

    def process_query(self, query: str) -> str:
        query_lower = query.lower().strip()

        # Parse intent and extract parameters
        if "list" in query_lower and "sales" in query_lower:
            return self._handle_list_sales_orders()

        elif "list" in query_lower and "employee" in query_lower:
            return self._handle_list_employees()

        elif "get" in query_lower and "sales" in query_lower:
            order_id = self._extract_id(query.strip())
            if not order_id:
                return "Error: No sales order ID provided. Use 'get sales order <id>'"
            return self._handle_get_sales_order(order_id)

        elif "get" in query_lower and "employee" in query_lower:
            employee_id = self._extract_id(query.strip())
            if not employee_id:
                return "Error: No employee ID provided. Use 'get employee <id>'"
            return self._handle_get_employee(employee_id)

        else:
            return "Error: Unable to understand query. Try 'list sales orders', 'list employees', 'get sales order <id>', or 'get employee <id>'"

    def _extract_id(self, query: str) -> Optional[str]:
        # Extract ID from the original query (alphanumeric + hyphens), preserving case.
        match = re.search(r'\b([A-Za-z0-9\-]+)\s*$', query)
        return match.group(1) if match else None

    def _handle_list_sales_orders(self) -> str:
        result = self._mcp_client.invoke_list_sales_order()

        if result is None:
            return "Error: Unable to fetch sales orders"

        data = result.get("data", [])
        if not data:
            return "No sales orders found"

        count = len(data)
        ids = [item.get("id", "unknown") for item in data[:5]]
        summary = ", ".join(ids)
        if count > 5:
            summary += f", ... ({count - 5} more)"

        return f"Found {count} sales orders: {summary}"

    def _handle_list_employees(self) -> str:
        result = self._mcp_client.invoke_list_employees()

        if result is None:
            return "Error: Unable to fetch employees"

        data = result.get("data", [])
        if not data:
            return "No employees found"

        count = len(data)
        names = [item.get("name", "unknown") for item in data[:5]]
        summary = ", ".join(names)
        if count > 5:
            summary += f", ... ({count - 5} more)"

        return f"Found {count} employees: {summary}"

    def _handle_get_sales_order(self, order_id: str) -> str:
        result = self._mcp_client.invoke_get_sales_order_by_id(order_id)

        if result is None:
            return f"Error: Sales order {order_id} not found"

        data = result.get("data", {})
        order_id_val = data.get("id", "unknown")
        amount = data.get("amount", "unknown")

        return f"Sales Order {order_id_val}: Amount: {amount}"

    def _handle_get_employee(self, employee_id: str) -> str:
        result = self._mcp_client.invoke_get_employee_by_id(employee_id)

        if result is None:
            return f"Error: Employee {employee_id} not found"

        data = result.get("data", {})
        name = data.get("name", "unknown")
        emp_id = data.get("id", "unknown")
        department = data.get("department", "unknown")

        return f"Employee {name} (ID: {emp_id}): Department: {department}"
