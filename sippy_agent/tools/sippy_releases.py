"""
Tool for getting OpenShift release information from Sippy API.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
from pydantic import Field
import httpx

from .base_tool import SippyBaseTool, SippyToolInput

logger = logging.getLogger(__name__)


class SippyReleasesTool(SippyBaseTool):
    """Tool for getting OpenShift release information from Sippy API."""
    
    name: str = "get_release_info"
    description: str = "Get OpenShift release information including available releases, GA dates, and development start dates. Can answer questions like 'What's the most recent release?', 'When did 4.18 go GA?', 'When did development start on 4.16?'. Input: optional release_version to get specific release info, or leave empty for all releases"
    
    # Add sippy_api_url as a proper field
    sippy_api_url: Optional[str] = Field(default=None, description="Sippy API base URL")
    
    class ReleasesInput(SippyToolInput):
        release_version: Optional[str] = Field(
            default=None, 
            description="Specific release version to get info for (e.g., '4.18', '4.20'). Leave empty to get all releases."
        )
        query_type: Optional[str] = Field(
            default="general",
            description="Type of query: 'general' (default), 'latest', 'ga_date', 'dev_start', 'all_releases'"
        )
        sippy_api_url: Optional[str] = Field(default=None, description="Sippy API base URL (optional, uses config if not provided)")
    
    args_schema: Type[SippyToolInput] = ReleasesInput
    
    def _run(
        self, 
        release_version: Optional[str] = None,
        query_type: Optional[str] = "general",
        sippy_api_url: Optional[str] = None
    ) -> str:
        """Get release information from Sippy API."""
        # Use provided URL or fall back to instance URL
        api_url = sippy_api_url or self.sippy_api_url
        
        if not api_url:
            return "Error: No Sippy API URL configured. Please set SIPPY_API_URL environment variable or provide sippy_api_url parameter."
        
        # Construct the API endpoint
        endpoint = f"{api_url.rstrip('/')}/api/releases"
        
        try:
            logger.info(f"Making request to {endpoint}")
            
            with httpx.Client(timeout=30.0) as client:
                response = client.get(endpoint)
                response.raise_for_status()
                
                data = response.json()
                
                # Format the response based on query type and release version
                return self._format_release_response(data, release_version, query_type)
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting release info: {e}")
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            logger.error(f"Request error getting release info: {e}")
            return f"Error: Failed to connect to Sippy API at {api_url} - {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return f"Error: Invalid JSON response from Sippy API"
        except Exception as e:
            logger.error(f"Unexpected error getting release info: {e}")
            return f"Error: Unexpected error - {str(e)}"
    
    def _format_release_response(
        self, 
        data: Dict[str, Any], 
        release_version: Optional[str],
        query_type: str
    ) -> str:
        """Format the release response data for display."""
        if not data:
            return "No data returned from Sippy API"
        
        releases = data.get("releases", [])
        ga_dates = data.get("ga_dates", {})
        dates = data.get("dates", {})
        last_updated = data.get("last_updated", "")
        
        if not releases:
            return "No releases found in Sippy API response"
        
        # Handle specific release version queries
        if release_version:
            return self._format_specific_release(
                release_version, releases, ga_dates, dates, query_type
            )
        
        # Handle different query types for all releases
        if query_type == "latest":
            return self._format_latest_release(releases, ga_dates, dates)
        elif query_type == "all_releases":
            return self._format_all_releases(releases, ga_dates, dates, last_updated)
        else:
            # Default general response
            return self._format_general_response(releases, ga_dates, dates, last_updated)
    
    def _format_specific_release(
        self,
        release_version: str,
        releases: List[str],
        ga_dates: Dict[str, str],
        dates: Dict[str, Dict[str, str]],
        query_type: str
    ) -> str:
        """Format response for a specific release version."""
        # Clean the release version
        clean_version = release_version.strip()
        
        if clean_version not in releases:
            available = ", ".join(releases[:10])  # Show first 10
            return f"Error: Release '{clean_version}' not found. Available releases: {available}"
        
        result = f"**OpenShift Release {clean_version} Information**\n\n"
        
        # Get GA date
        ga_date = ga_dates.get(clean_version)
        if ga_date:
            formatted_ga = self._format_date(ga_date)
            result += f"**GA Date:** {formatted_ga}\n"
        else:
            result += f"**GA Date:** Not yet released\n"
        
        # Get development start date
        release_dates = dates.get(clean_version, {})
        dev_start = release_dates.get("development_start")
        if dev_start:
            formatted_dev = self._format_date(dev_start)
            result += f"**Development Start:** {formatted_dev}\n"
        
        # Add status information
        if ga_date:
            result += f"**Status:** ✅ Generally Available\n"
        else:
            result += f"**Status:** 🚧 In Development\n"
        
        # Add position in release list
        try:
            position = releases.index(clean_version) + 1
            result += f"**Position:** #{position} in release list\n"
        except ValueError:
            pass
        
        return result
    
    def _format_latest_release(
        self,
        releases: List[str],
        ga_dates: Dict[str, str],
        dates: Dict[str, Dict[str, str]]
    ) -> str:
        """Format response for latest release query."""
        if not releases:
            return "No releases available"
        
        # The first release in the list is the most recent
        latest_release = releases[0]
        
        result = f"**🎯 Latest OpenShift Release: {latest_release}**\n\n"
        
        # Check if it's GA or in development
        ga_date = ga_dates.get(latest_release)
        if ga_date:
            formatted_ga = self._format_date(ga_date)
            result += f"**Status:** ✅ Generally Available (GA: {formatted_ga})\n"
        else:
            result += f"**Status:** 🚧 In Development\n"
            
            # Show development start if available
            release_dates = dates.get(latest_release, {})
            dev_start = release_dates.get("development_start")
            if dev_start:
                formatted_dev = self._format_date(dev_start)
                result += f"**Development Started:** {formatted_dev}\n"
        
        result += f"\n**Quick Answer:** The most recent OpenShift release is {latest_release}.\n"
        
        return result
    
    def _format_all_releases(
        self,
        releases: List[str],
        ga_dates: Dict[str, str],
        dates: Dict[str, Dict[str, str]],
        last_updated: str
    ) -> str:
        """Format response showing all releases."""
        result = f"**📋 All OpenShift Releases**\n\n"
        result += f"**Total Releases:** {len(releases)}\n"
        
        if last_updated:
            formatted_updated = self._format_date(last_updated)
            result += f"**Last Updated:** {formatted_updated}\n"
        
        result += f"\n**Release List:**\n"
        
        for i, release in enumerate(releases, 1):
            ga_date = ga_dates.get(release)
            status_emoji = "✅" if ga_date else "🚧"
            status_text = "GA" if ga_date else "Dev"
            
            result += f"{i:2d}. **{release}** {status_emoji} {status_text}"
            
            if ga_date:
                formatted_ga = self._format_date(ga_date)
                result += f" (GA: {formatted_ga})"
            else:
                # Show dev start if available
                release_dates = dates.get(release, {})
                dev_start = release_dates.get("development_start")
                if dev_start:
                    formatted_dev = self._format_date(dev_start)
                    result += f" (Dev: {formatted_dev})"
            
            result += "\n"
        
        return result
    
    def _format_general_response(
        self,
        releases: List[str],
        ga_dates: Dict[str, str],
        dates: Dict[str, Dict[str, str]],
        last_updated: str
    ) -> str:
        """Format general response with summary information."""
        result = f"**OpenShift Release Information Summary**\n\n"
        
        # Latest release info
        if releases:
            latest_release = releases[0]
            ga_date = ga_dates.get(latest_release)
            
            result += f"**🎯 Latest Release:** {latest_release}"
            if ga_date:
                formatted_ga = self._format_date(ga_date)
                result += f" ✅ (GA: {formatted_ga})"
            else:
                result += f" 🚧 (In Development)"
            result += "\n\n"
        
        # Statistics
        ga_count = len(ga_dates)
        dev_count = len(releases) - ga_count
        
        result += f"**📊 Release Statistics:**\n"
        result += f"• Total Releases: {len(releases)}\n"
        result += f"• Generally Available: {ga_count}\n"
        result += f"• In Development: {dev_count}\n\n"
        
        # Recent GA releases
        recent_ga = []
        for release in releases:
            if release in ga_dates:
                recent_ga.append(release)
                if len(recent_ga) >= 5:  # Show last 5 GA releases
                    break
        
        if recent_ga:
            result += f"**📅 Recent GA Releases:**\n"
            for release in recent_ga:
                ga_date = ga_dates[release]
                formatted_ga = self._format_date(ga_date)
                result += f"• {release}: {formatted_ga}\n"
            result += "\n"
        
        if last_updated:
            formatted_updated = self._format_date(last_updated)
            result += f"**Last Updated:** {formatted_updated}\n"
        
        result += f"\n💡 **Tip:** Use specific release version (e.g., '4.18') to get detailed information about that release.\n"
        
        return result
    
    def _format_date(self, date_str: str) -> str:
        """Format ISO date string to readable format."""
        try:
            # Parse ISO format date
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except Exception:
            # Return original if parsing fails
            return date_str
