"""
Tool for querying Jira for known open incidents in the TRT project.
"""

import json
import logging
from typing import Any, Dict, Optional, Type
from pydantic import Field
import httpx

from .base_tool import SippyBaseTool, SippyToolInput

logger = logging.getLogger(__name__)


class SippyJiraIncidentTool(SippyBaseTool):
    """Tool for querying Jira for known open incidents in the TRT project."""
    
    name: str = "check_known_incidents"
    description: str = "Check Jira for known open TRT incidents. ONLY use this when job errors suggest a correlation. Use specific keywords that match actual errors found in logs."
    
    # Add Jira configuration as proper fields
    jira_url: str = Field(default="https://issues.redhat.com", description="Jira instance URL")
    jira_username: Optional[str] = Field(default=None, description="Jira username")
    jira_token: Optional[str] = Field(default=None, description="Jira API token")
    
    class JiraIncidentInput(SippyToolInput):
        search_terms: Optional[str] = Field(
            default=None,
            description="Optional search terms to filter incidents (e.g., 'registry', 'build11', 'timeout')"
        )
        jira_url: Optional[str] = Field(default=None, description="Jira URL (optional, uses config if not provided)")
    
    args_schema: Type[SippyToolInput] = JiraIncidentInput
    
    def _run(self, search_terms: Optional[str] = None, jira_url: Optional[str] = None) -> str:
        """Query Jira for known open incidents."""
        # Use provided URL or fall back to instance URL
        api_url = jira_url or self.jira_url
        
        if not api_url:
            return "Error: No Jira URL configured. Please set JIRA_URL environment variable or provide jira_url parameter."
        
        # Clean up search terms - filter out common LLM artifacts
        clean_search_terms = None
        if search_terms:
            # Remove common LLM response artifacts and clean up the input
            search_terms = str(search_terms).strip()
            
            # Skip if it contains common LLM artifacts or is too long
            skip_patterns = [
                "none", "null", "no job", "let's", "we can", "this time",
                "search for all", "open incidents", "trt project", "default value"
            ]
            
            if (len(search_terms) > 50 or
                any(pattern in search_terms.lower() for pattern in skip_patterns) or
                search_terms.lower() in ["none", "null", ""]):
                clean_search_terms = None
            else:
                # Extract meaningful keywords
                import re
                # Look for technical terms that might be relevant
                tech_terms = re.findall(r'\b(registry|build\d+|timeout|error|fail|503|502|infrastructure|node|cluster|network)\b', search_terms.lower())
                if tech_terms:
                    clean_search_terms = ','.join(tech_terms[:3])  # Limit to 3 terms
                else:
                    clean_search_terms = None
        
        # Construct the Jira REST API endpoint
        endpoint = f"{api_url.rstrip('/')}/rest/api/2/search"
        
        # Build JQL query for TRT project incidents
        jql_parts = [
            'project = "TRT"',
            'labels = "trt-incident"',
            'status not in (Closed, Done, Resolved)'
        ]
        
        # Add search terms if we have clean ones
        if clean_search_terms:
            # Split search terms and add them as text search
            terms = [term.strip() for term in clean_search_terms.split(',') if term.strip()]
            if terms:
                text_search = ' OR '.join([f'text ~ "{term}"' for term in terms])
                jql_parts.append(f'({text_search})')
        
        jql = ' AND '.join(jql_parts)
        
        try:
            # Prepare request parameters
            params = {
                'jql': jql,
                'fields': 'key,summary,status,priority,created,updated,description,labels',
                'maxResults': 20  # Limit results
            }
            
            # Prepare authentication if available
            auth = None
            if self.jira_username and self.jira_token:
                auth = (self.jira_username, self.jira_token)
            
            if clean_search_terms:
                logger.info(f"Querying Jira with search terms: {clean_search_terms}")
            else:
                logger.info("Querying Jira for all open TRT incidents")
            logger.info(f"JQL: {jql}")
            
            # Make the API request
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    endpoint,
                    params=params,
                    auth=auth,
                    headers={'Accept': 'application/json'}
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Format the response
                return self._format_jira_incidents(data, clean_search_terms)
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error querying Jira: {e}")
            if e.response.status_code == 401:
                return "Error: Jira authentication failed. Check JIRA_USERNAME and JIRA_TOKEN environment variables."
            elif e.response.status_code == 403:
                return "Error: Access denied to Jira. You may need authentication or permissions to view TRT project."
            else:
                return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            logger.error(f"Request error querying Jira: {e}")
            return f"Error: Failed to connect to Jira at {api_url} - {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return f"Error: Invalid JSON response from Jira API"
        except Exception as e:
            logger.error(f"Unexpected error querying Jira: {e}")
            return f"Error: Unexpected error - {str(e)}"

    def _format_jira_incidents(self, data: Dict[str, Any], search_terms: Optional[str] = None) -> str:
        """Format the Jira incidents for display."""
        issues = data.get('issues', [])
        total = data.get('total', 0)

        if not issues:
            if search_terms:
                return f"No open TRT incidents found matching search terms: {search_terms}"
            else:
                return "No open TRT incidents found with 'trt-incident' label"

        result = f"**Known Open Incidents**\n\n"
        if search_terms:
            result += f"**Search Terms:** {search_terms}\n"
        result += f"**Found {len(issues)} of {total} total incidents:**\n\n"

        for issue in issues:
            key = issue.get('key', 'Unknown')
            fields = issue.get('fields', {})
            summary = fields.get('summary', 'No summary')
            status = fields.get('status', {}).get('name', 'Unknown')
            priority = fields.get('priority', {}).get('name', 'Unknown')
            created = fields.get('created', '')
            updated = fields.get('updated', '')
            description = fields.get('description', '')
            labels = fields.get('labels', [])

            # Format dates
            created_date = self._format_jira_date(created)
            updated_date = self._format_jira_date(updated)

            result += f"**🎫 {key}** - {summary}\n"
            result += f"📊 Status: {status} | Priority: {priority}\n"
            result += f"📅 Created: {created_date} | Updated: {updated_date}\n"

            if labels:
                relevant_labels = [label for label in labels if 'trt' in label.lower() or 'incident' in label.lower()]
                if relevant_labels:
                    result += f"🏷️ Labels: {', '.join(relevant_labels)}\n"

            # Include description snippet if available
            if description:
                # Clean and truncate description
                clean_desc = description.replace('\n', ' ').replace('\r', '').strip()
                if len(clean_desc) > 200:
                    clean_desc = clean_desc[:200] + "..."
                result += f"📝 Description: {clean_desc}\n"

            # Add Jira link
            jira_base = self.jira_url.rstrip('/')
            result += f"🔗 [View in Jira]({jira_base}/browse/{key})\n\n"

        if total > len(issues):
            result += f"... and {total - len(issues)} more incidents (use more specific search terms to narrow results)\n"

        return result

    def _format_jira_date(self, date_str: str) -> str:
        """Format Jira date string to a more readable format."""
        if not date_str:
            return "Unknown"

        try:
            # Jira dates are in ISO format like "2024-01-15T10:30:45.000+0000"
            from datetime import datetime
            # Remove timezone info for simple parsing
            clean_date = date_str.split('T')[0]
            date_obj = datetime.strptime(clean_date, '%Y-%m-%d')
            return date_obj.strftime('%Y-%m-%d')
        except Exception:
            return date_str.split('T')[0] if 'T' in date_str else date_str
