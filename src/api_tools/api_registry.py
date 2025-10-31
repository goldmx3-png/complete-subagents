"""
API Registry - Database access for available banking APIs
"""

import asyncpg
import json
from typing import List, Dict, Optional
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class APIRegistry:
    """
    Manages API definitions stored in PostgreSQL database

    Database schema:
        - product: Banking product (payments, accounts, cards, loans)
        - api_name: Unique API identifier
        - api_description: What the API does
        - http_method: GET, POST, etc.
        - endpoint_url: API endpoint
        - request_schema: JSON schema for request
        - response_schema: JSON schema for response
        - example_request/response: Examples
        - is_active: Enable/disable API
    """

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or settings.database_url
        self._conn = None
        self._pool = None

    async def _get_connection(self) -> asyncpg.Connection:
        """Get database connection"""
        if self._conn is None or self._conn.is_closed():
            try:
                self._conn = await asyncpg.connect(self.db_url)
                logger.info("API Registry database connection established")
            except Exception as e:
                logger.error(f"Database connection error: {str(e)}")
                raise
        return self._conn

    async def close(self):
        """Close database connection"""
        if self._conn and not self._conn.is_closed():
            await self._conn.close()
            logger.info("API Registry database connection closed")

    async def get_apis_by_product(
        self,
        product: str,
        active_only: bool = True
    ) -> List[Dict]:
        """
        Get all available APIs for a banking product

        Args:
            product: Product identifier (payments, accounts, cards, loans)
            active_only: Only return active APIs (default: True)

        Returns:
            List of API definitions

        Example:
            apis = await registry.get_apis_by_product("accounts")
            # Returns: [
            #   {"api_name": "get_account_balance", "api_description": "...", ...},
            #   {"api_name": "get_account_details", "api_description": "...", ...}
            # ]
        """
        try:
            conn = await self._get_connection()

            query = """
                SELECT
                    id, product, api_name, api_description,
                    http_method, endpoint_url,
                    request_schema, response_schema,
                    example_request, example_response,
                    list_formatting_template,
                    is_active, requires_auth, rate_limit,
                    created_at, updated_at
                FROM api_registry
                WHERE product = $1
            """

            if active_only:
                query += " AND is_active = true"

            query += " ORDER BY api_name"

            rows = await conn.fetch(query, product)

            apis = []
            for row in rows:
                api = dict(row)

                # Parse JSON fields if they're strings
                for field in ['request_schema', 'response_schema', 'example_request', 'example_response']:
                    if api.get(field) and isinstance(api[field], str):
                        try:
                            api[field] = json.loads(api[field])
                        except json.JSONDecodeError:
                            pass

                apis.append(api)

            logger.info(f"Found {len(apis)} APIs for product '{product}'")
            return apis

        except Exception as e:
            logger.error(f"Error fetching APIs for product '{product}': {str(e)}")
            return []

    async def get_api_by_name(
        self,
        product: str,
        api_name: str
    ) -> Optional[Dict]:
        """
        Get specific API by product and name

        Args:
            product: Product identifier
            api_name: API name

        Returns:
            API definition dict or None if not found
        """
        try:
            conn = await self._get_connection()

            query = """
                SELECT
                    id, product, api_name, api_description,
                    http_method, endpoint_url,
                    request_schema, response_schema,
                    example_request, example_response,
                    list_formatting_template,
                    is_active, requires_auth, rate_limit
                FROM api_registry
                WHERE product = $1 AND api_name = $2 AND is_active = true
            """

            row = await conn.fetchrow(query, product, api_name)

            if not row:
                logger.warning(f"API not found: {product}/{api_name}")
                return None

            api = dict(row)

            # Parse JSON fields
            for field in ['request_schema', 'response_schema', 'example_request', 'example_response']:
                if api.get(field) and isinstance(api[field], str):
                    try:
                        api[field] = json.loads(api[field])
                    except json.JSONDecodeError:
                        pass

            return api

        except Exception as e:
            logger.error(f"Error fetching API {product}/{api_name}: {str(e)}")
            return None

    async def search_apis_by_keywords(
        self,
        product: str,
        keywords: List[str]
    ) -> List[Dict]:
        """
        Search APIs by keywords in description

        Args:
            product: Product identifier
            keywords: List of keywords to search

        Returns:
            List of matching APIs
        """
        try:
            conn = await self._get_connection()

            # Build keyword search condition (case-insensitive)
            keyword_conditions = " OR ".join(
                [f"LOWER(api_description) LIKE LOWER('%{kw}%')" for kw in keywords]
            )

            query = f"""
                SELECT
                    id, product, api_name, api_description,
                    http_method, endpoint_url,
                    request_schema, response_schema,
                    example_request, example_response,
                    list_formatting_template,
                    is_active, requires_auth, rate_limit
                FROM api_registry
                WHERE product = $1
                AND is_active = true
                AND ({keyword_conditions})
                ORDER BY api_name
            """

            rows = await conn.fetch(query, product)

            apis = [dict(row) for row in rows]
            logger.info(f"Found {len(apis)} APIs matching keywords for product '{product}'")
            return apis

        except Exception as e:
            logger.error(f"Error searching APIs: {str(e)}")
            return []

    async def get_all_products(self) -> List[str]:
        """
        Get list of all available products

        Returns:
            List of product names
        """
        try:
            conn = await self._get_connection()

            query = """
                SELECT DISTINCT product
                FROM api_registry
                WHERE is_active = true
                ORDER BY product
            """

            rows = await conn.fetch(query)
            products = [row['product'] for row in rows]

            logger.info(f"Found {len(products)} products: {products}")
            return products

        except Exception as e:
            logger.error(f"Error fetching products: {str(e)}")
            return []

    async def get_api_count(self, product: Optional[str] = None) -> int:
        """
        Get count of active APIs

        Args:
            product: Optional product filter

        Returns:
            Number of APIs
        """
        try:
            conn = await self._get_connection()

            if product:
                query = "SELECT COUNT(*) FROM api_registry WHERE product = $1 AND is_active = true"
                count = await conn.fetchval(query, product)
            else:
                query = "SELECT COUNT(*) FROM api_registry WHERE is_active = true"
                count = await conn.fetchval(query)

            return count or 0

        except Exception as e:
            logger.error(f"Error counting APIs: {str(e)}")
            return 0

    async def health_check(self) -> bool:
        """
        Check if database connection is healthy

        Returns:
            True if connection is healthy
        """
        try:
            conn = await self._get_connection()
            await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False


# Singleton instance
_registry_instance = None


async def get_api_registry() -> APIRegistry:
    """
    Get singleton API Registry instance

    Usage:
        registry = await get_api_registry()
        apis = await registry.get_apis_by_product("accounts")
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = APIRegistry()
    return _registry_instance
