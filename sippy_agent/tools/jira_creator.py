"""
Tool for creating Jira tickets in the TRT project and other projects.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import Field, validator
import httpx

from .base_tool import SippyBaseTool, SippyToolInput

logger = logging.getLogger(__name__)


class SippyJiraTicketCreatorTool(SippyBaseTool):
    """Tool for creating Jira tickets."""
    
    name: str = "create_jira_ticket"
    description: str = (
        "Create a Jira ticket. Returns a JSON object with the new ticket's key and URL upon successful creation. "
        "IMPORTANT: This tool MUST ask for user confirmation before creating any ticket. "
        "Never create more than one ticket per chat session. Use this only when the user explicitly requests "
        "ticket creation or when you've identified a critical issue that needs tracking."
    )
    
    # Jira configuration
    jira_url: str = Field(default="https://issues.redhat.com", description="Jira instance URL")
    jira_token: Optional[str] = Field(default=None, description="Jira API token")
    
    # Session tracking to prevent multiple ticket creation
    _tickets_created_in_session: int = 0
    
    class JiraTicketInput(SippyToolInput):
        project: str = Field(description="Jira project key (e.g., 'TRT')")
        title: str = Field(description="Ticket title/summary")
        description: str = Field(description="Ticket description (use Wiki Markup for formatting)")
        issue_type: str = Field(default="Story", description="Issue type (Story, Bug, Task, etc.)")
        priority: str = Field(default="Normal", description="Priority (Critical, High, Normal, Low)")
        labels: Optional[List[str]] = Field(default=None, description="Optional labels to add to the ticket")
        confirm_creation: bool = Field(default=False, description="User confirmation required - must be explicitly set to True")
        
        @validator('project')
        def validate_project(cls, v):
            if not v or not v.strip():
                raise ValueError("Project is required")
            return v.strip().upper()
        
        @validator('title')
        def validate_title(cls, v):
            if not v or not v.strip():
                raise ValueError("Title is required")
            if len(v.strip()) > 255:
                raise ValueError("Title must be 255 characters or less")
            return v.strip()
        
        @validator('description')
        def validate_description(cls, v):
            if not v or not v.strip():
                raise ValueError("Description is required")
            return v.strip()
        
        @validator('issue_type')
        def validate_issue_type(cls, v):
            valid_types = ["Story", "Bug", "Task", "Epic", "Sub-task"]
            if v not in valid_types:
                logger.warning(f"Issue type '{v}' may not be valid. Common types: {', '.join(valid_types)}")
            return v
        
        @validator('priority')
        def validate_priority(cls, v):
            valid_priorities = ["Critical", "High", "Normal", "Low"]
            if v not in valid_priorities:
                logger.warning(f"Priority '{v}' may not be valid. Valid priorities: {', '.join(valid_priorities)}")
            return v
    
    args_schema: Type[SippyToolInput] = JiraTicketInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Get Jira token from environment
        self.jira_token = os.getenv('JIRA_TOKEN')
    
    def _run(self, *args, **kwargs: Any) -> Union[str, Dict[str, Any]]:
        """Create a Jira ticket with the specified parameters."""
        
        input_data = {}
        if args and isinstance(args[0], dict):
            input_data.update(args[0])
        input_data.update(kwargs)

        try:
            # Pydantic model will have validated and filled in defaults
            params = self.JiraTicketInput(**input_data)
        except Exception as e:
            return f"❌ **Error**: Invalid input parameters: {e}"

        # Check session limit
        if self._tickets_created_in_session >= 1:
            return (
                "❌ **Ticket Creation Blocked**: I can only create one Jira ticket per chat session. "
                "This is a safety measure to prevent accidental spam. Please start a new conversation "
                "if you need to create another ticket."
            )
        
        # Require user confirmation
        if not params.confirm_creation:
            # Prepare ticket summary for user review
            summary = self._format_ticket_preview(
                params.project, params.title, params.description, 
                params.issue_type, params.priority, params.labels
            )
            return (
                f"{summary}\n\n"
                "⚠️ **User Confirmation Required**: I need your explicit approval before creating this Jira ticket. "
                "Please review the details above carefully. If you want to proceed, please respond with confirmation "
                "and I'll create the ticket with these exact details.\n\n"
                "**This is a safety measure - I will never create tickets without your permission.**"
            )
        
        # Check for Jira token
        if not self.jira_token:
            logger.error("No Jira token found in environment variable JIRA_TOKEN")
            return {
                "error": "Authentication Error: No Jira token found. Please set the JIRA_TOKEN environment variable."
            }
        
        try:
            # Create the ticket
            logger.info(f"Attempting to create Jira ticket in project '{params.project}' with title '{params.title[:50]}...'")
            ticket_key = self._create_jira_ticket(
                params.project, params.title, params.description, 
                params.issue_type, params.priority, params.labels
            )
            
            # Increment session counter
            self._tickets_created_in_session += 1
            
            # Format success response as JSON
            jira_base = self.jira_url.rstrip('/')
            return {
                "message": "Jira Ticket Created Successfully!",
                "ticket_key": ticket_key,
                "project": params.project,
                "title": params.title,
                "url": f"{jira_base}/browse/{ticket_key}",
            }
            
        except Exception as e:
            logger.error(f"Error creating Jira ticket: {e}")
            return {"error": f"Error Creating Ticket: {str(e)}"}
    
    def _format_ticket_preview(
        self, 
        project: str, 
        title: str, 
        description: str, 
        issue_type: str, 
        priority: str, 
        labels: Optional[List[str]]
    ) -> str:
        """Format a preview of the ticket to be created."""
        preview = f"📋 **Ticket Preview**\n\n"
        preview += f"**Project**: {project}\n"
        preview += f"**Type**: {issue_type}\n"
        preview += f"**Priority**: {priority}\n"
        preview += f"**Title**: {title}\n\n"
        preview += f"**Description**:\n{description}\n"
        
        if labels:
            preview += f"\n**Labels**: {', '.join(labels)}\n"
        
        return preview
    
    def _create_jira_ticket(
        self, 
        project: str, 
        title: str, 
        description: str, 
        issue_type: str, 
        priority: str, 
        labels: Optional[List[str]]
    ) -> str:
        """Create the actual Jira ticket and return the ticket key."""
        
        # Construct the Jira REST API endpoint
        endpoint = f"{self.jira_url.rstrip('/')}/rest/api/2/issue"
        
        # Build the issue payload
        issue_data = {
            "fields": {
                "project": {
                    "key": project
                },
                "summary": title,
                "description": description,
                "issuetype": {
                    "name": issue_type
                },
                "priority": {
                    "name": priority
                }
            }
        }
        
        # Add labels if provided
        if labels:
            issue_data["fields"]["labels"] = labels
        
        try:
            # Prepare authentication
            username = os.getenv('JIRA_USERNAME')
            if not self.jira_token:
                raise Exception("Jira token is required for authentication")
            if not username:
                logger.error("JIRA_USERNAME environment variable is not set")
                raise Exception("JIRA_USERNAME environment variable is required. Set it to your Red Hat email address.")
            
            logger.info(f"Making Jira API request to {endpoint}")
            logger.info(f"Using username: {username}")
            logger.info(f"Token length: {len(self.jira_token) if self.jira_token else 0}")
            logger.info(f"Issue data: {json.dumps(issue_data, indent=2)}")
            
            # Make the API request - try Bearer token first
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.jira_token}'
            }
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    endpoint,
                    json=issue_data,
                    headers=headers
                )
                
                if response.status_code == 201:
                    # Ticket created successfully
                    response_data = response.json()
                    return response_data.get('key', 'Unknown')
                else:
                    # Handle errors
                    error_msg = f"HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        if 'errors' in error_data:
                            error_details = []
                            for field, error in error_data['errors'].items():
                                error_details.append(f"{field}: {error}")
                            error_msg += f" - {'; '.join(error_details)}"
                        elif 'errorMessages' in error_data:
                            error_msg += f" - {'; '.join(error_data['errorMessages'])}"
                    except:
                        error_msg += f" - {response.text}"
                    
                    raise Exception(error_msg)
                    
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise Exception("Authentication failed. Check JIRA_TOKEN environment variable.")
            elif e.response.status_code == 403:
                raise Exception(f"Access denied. You may not have permission to create issues in project {project}.")
            else:
                raise Exception(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise Exception(f"Failed to connect to Jira: {str(e)}")


def create_release_payload_incident_ticket(
    title: str,
    description: str,
    user_confirmed: bool = False
) -> str:
    """
    Helper function to create a TRT incident ticket for release payload issues.
    This is called by the agent when it identifies release payload problems.
    """
    
    tool = SippyJiraTicketCreatorTool()
    
    # Format description with Wiki Markup
    formatted_description = f"""
h2. Release Payload Issue Detected

{description}

h3. Next Steps
* Investigate the root cause of the failures
* Determine if this affects multiple payloads
* Implement fixes or workarounds as needed
* Update this ticket with findings and resolution

_This ticket was automatically created by Sippy Agent based on analysis of rejected release payloads._
    """.strip()
    
    return tool._run(
        project="TRT",
        title=title,
        description=formatted_description,
        issue_type="Story",
        priority="Critical",
        labels=["trt-incident"],
        confirm_creation=user_confirmed
    ) 