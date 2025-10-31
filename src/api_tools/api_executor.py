"""
API Executor - Execute HTTP calls to banking APIs with JWT authentication
"""

import httpx
import json
import asyncio
from typing import Dict, Optional, List
from datetime import datetime
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class APIExecutor:
    """
    Execute HTTP API calls to banking backend
    Features:
    - JWT authentication
    - Retry logic for failed requests
    - Parallel execution support
    - Detailed logging
    """

    def __init__(
        self,
        jwt_token: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        verify_ssl: Optional[bool] = None
    ):
        """
        Initialize API executor

        Args:
            jwt_token: JWT authentication token (defaults to settings)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            verify_ssl: Verify SSL certificates (False for self-signed certs)
        """
        self.jwt_token = jwt_token or settings.banking_api_jwt_token
        self.timeout = timeout or settings.banking_api_timeout
        self.max_retries = max_retries or settings.banking_api_max_retries
        self.verify_ssl = verify_ssl if verify_ssl is not None else settings.banking_api_verify_ssl

        if not self.jwt_token:
            logger.warning("No JWT token configured for API executor")

        if not self.verify_ssl:
            logger.warning("SSL verification is DISABLED - use only in development")

        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            verify=self.verify_ssl,
            headers={
                "Authorization": f"Bearer {self.jwt_token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )

    async def execute(
        self,
        api_definition: Dict,
        parameters: Optional[Dict] = None,
        user_context: Optional[Dict] = None
    ) -> Dict:
        """
        Execute a single API call

        Args:
            api_definition: API definition from registry
            parameters: Request parameters
            user_context: User context (user_id, session_id, etc.)

        Returns:
            {
                "success": bool,
                "data": dict (if success),
                "error": str (if failed),
                "status_code": int,
                "execution_time_ms": float,
                "api_name": str
            }
        """
        start_time = datetime.now()

        # Extract API details
        api_name = api_definition["api_name"]
        url = api_definition["endpoint_url"]
        method = api_definition.get("http_method", "POST").upper()

        # Handle empty parameters - use request_schema as template
        if not parameters or parameters == {}:
            request_schema = api_definition.get("request_schema", {})
            if isinstance(request_schema, str):
                try:
                    parameters = json.loads(request_schema)
                except json.JSONDecodeError:
                    parameters = {}
            else:
                parameters = request_schema

        logger.info(f"Executing API: {api_name} [{method}] {url}")
        logger.debug(f"Request body: {json.dumps(parameters, indent=2)[:500]}")

        # Prepare request
        request_kwargs = {"url": url, "headers": self.client.headers.copy()}

        # Add user context to headers
        if user_context:
            if "user_id" in user_context:
                request_kwargs["headers"]["X-User-Id"] = str(user_context["user_id"])
            if "session_id" in user_context:
                request_kwargs["headers"]["X-Session-Id"] = str(user_context["session_id"])

        # Add parameters based on HTTP method
        if method in ["GET", "DELETE"]:
            request_kwargs["params"] = parameters
        elif method in ["POST", "PUT", "PATCH"]:
            request_kwargs["json"] = parameters

        # Execute with retry logic
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Make HTTP request
                if method == "GET":
                    response = await self.client.get(**request_kwargs)
                elif method == "POST":
                    response = await self.client.post(**request_kwargs)
                elif method == "PUT":
                    response = await self.client.put(**request_kwargs)
                elif method == "PATCH":
                    response = await self.client.patch(**request_kwargs)
                elif method == "DELETE":
                    response = await self.client.delete(**request_kwargs)
                else:
                    return self._build_error_result(
                        api_name, f"Unsupported HTTP method: {method}", 0, start_time
                    )

                execution_time = (datetime.now() - start_time).total_seconds() * 1000

                # Check response status
                if 200 <= response.status_code < 300:
                    # Success
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        data = {"raw_response": response.text}

                    logger.info(f"API SUCCESS: {api_name}, status={response.status_code}, time={execution_time:.0f}ms")
                    logger.debug(f"Response: {json.dumps(data, indent=2)[:500]}")

                    return {
                        "success": True,
                        "data": data,
                        "status_code": response.status_code,
                        "execution_time_ms": execution_time,
                        "api_name": api_name
                    }
                else:
                    # HTTP error
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.error(f"API FAILED: {api_name}, status={response.status_code}, response={response.text[:200]}")

                    # Retry on 5xx errors
                    if response.status_code >= 500 and attempt < self.max_retries:
                        logger.warning(f"Retrying API call (attempt {attempt + 1}/{self.max_retries})...")
                        last_error = error_msg
                        await asyncio.sleep(1 * attempt)  # Exponential backoff
                        continue

                    return self._build_error_result(
                        api_name, error_msg, response.status_code, start_time
                    )

            except httpx.TimeoutException:
                last_error = f"Request timeout ({self.timeout}s)"
                logger.error(f"API TIMEOUT: {api_name}")
                if attempt < self.max_retries:
                    logger.warning(f"Retrying (attempt {attempt + 1}/{self.max_retries})...")
                    await asyncio.sleep(1 * attempt)
                    continue

            except httpx.ConnectError as e:
                last_error = f"Connection error: {str(e)}"
                logger.error(f"API CONNECTION ERROR: {api_name}, error={str(e)}")
                if attempt < self.max_retries:
                    logger.warning(f"Retrying (attempt {attempt + 1}/{self.max_retries})...")
                    await asyncio.sleep(1 * attempt)
                    continue

            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"API UNEXPECTED ERROR: {api_name}, error={str(e)}", exc_info=True)
                if attempt < self.max_retries:
                    logger.warning(f"Retrying (attempt {attempt + 1}/{self.max_retries})...")
                    await asyncio.sleep(1 * attempt)
                    continue

        # All retries exhausted
        return self._build_error_result(api_name, last_error or "Unknown error", 0, start_time)

    async def execute_multiple(
        self,
        api_calls: List[Dict],
        user_context: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Execute multiple API calls in parallel

        Args:
            api_calls: List of:
                {
                    "api_definition": dict,
                    "parameters": dict
                }
            user_context: User context

        Returns:
            List of execution results
        """
        logger.info(f"Executing {len(api_calls)} APIs in parallel")

        # Create tasks for parallel execution
        tasks = [
            self.execute(
                api_definition=call["api_definition"],
                parameters=call.get("parameters"),
                user_context=user_context
            )
            for call in api_calls
        ]

        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                api_name = api_calls[idx]["api_definition"]["api_name"]
                processed_results.append({
                    "success": False,
                    "error": f"Exception: {str(result)}",
                    "status_code": 0,
                    "execution_time_ms": 0,
                    "api_name": api_name
                })
                logger.error(f"API exception: {api_name}, error={str(result)}")
            else:
                processed_results.append(result)

        successful = sum(1 for r in processed_results if r.get("success"))
        logger.info(f"Parallel execution complete: {successful}/{len(api_calls)} successful")

        return processed_results

    def _build_error_result(
        self,
        api_name: str,
        error_msg: str,
        status_code: int,
        start_time: datetime
    ) -> Dict:
        """Build standardized error result"""
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        return {
            "success": False,
            "error": error_msg,
            "status_code": status_code,
            "execution_time_ms": execution_time,
            "api_name": api_name
        }

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
        logger.info("API executor client closed")

    async def __aenter__(self):
        """Context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()


async def execute_api_call(
    api_definition: Dict,
    parameters: Optional[Dict] = None,
    user_context: Optional[Dict] = None
) -> Dict:
    """
    Convenience function for single API execution

    Usage:
        result = await execute_api_call(api_def, {"account_id": "123"})
        if result["success"]:
            print(result["data"])
        else:
            print(result["error"])
    """
    async with APIExecutor() as executor:
        return await executor.execute(api_definition, parameters, user_context)
