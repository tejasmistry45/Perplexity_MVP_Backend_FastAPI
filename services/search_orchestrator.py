import time
from typing import Dict, Any

from models.schemas import SearchRequest, SearchResponse, SearchResult, WebSearchResults
from services.tavily_service import TavilyService
from services.groq_service import GroqService
from services.content_synthesizer import ContentSynthesizer
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SearchOrchestrator:
    """Main orchestrator that coordinates query analysis and web search"""
    
    def __init__(self):
        self.groq_service = GroqService()
        self.tavily_service = TavilyService()
        self.content_synthesizer = ContentSynthesizer()

    async def execute_search(self, request: SearchRequest) -> SearchResponse:
        """Execute complete search pipeline: Analysis + Web Search + Synthesis"""

        start_time = time.time()
        analysis = None
        web_results = None  # Fixed variable name

        try:
            # Step 1: Analyze Query (directly with groq)
            logger.info(f"Step 1: Analyzing Query: '{request.query}'")
            analysis = await self.groq_service.analyze_query(request.query)

            # Step 2: Execute web searches
            logger.info(f"Step 2: Executing web search")
            web_results = await self._execute_web_search(analysis, request.query)

            # Step 3: Synthesize Response
            logger.info(f"Step 3: Synthesizing Response")
            synthesized_response = await self.content_synthesizer.synthesize_response(
                query=request.query,
                analysis=analysis,
                web_results=web_results
            )

            total_duration = time.time() - start_time
            logger.info(f"Total Search completed in {total_duration:.2f}s")

            # Create comprehensive response 
            response = SearchResponse(
                original_query=request.query,
                analysis=analysis,
                web_results=web_results,  # Fixed: was web_result
                synthesized_response=synthesized_response,  # Fixed: spelling
                status="search_completed",
                timestamp=datetime.now().isoformat()  # Fixed: spelling
            )

            return response
        
        except Exception as e:
            logger.error(f"Search Pipeline Failed: {e}")
            # Return partial response with error handling
            return self._create_error_response(request.query, analysis, web_results, str(e))
        
    async def _execute_web_search(self, analysis, original_query: str) -> WebSearchResults:
        """Execute web search using analyzed query data"""

        search_start = time.time()

        # Use suggested searches from analysis
        search_terms = analysis.suggested_searches  

        # Add original query as fallback if not in suggestions
        if original_query.lower() not in [term.lower() for term in search_terms]:
            search_terms = [original_query] + search_terms[:2]  

        # Limit search based on complexity
        max_searches = self._get_max_searches(analysis.complexity_score)
        search_terms = search_terms[:max_searches]
        logger.info(f"Using {len(search_terms)} search terms: {search_terms}")

        # Execute searches via tavily
        raw_results = await self.tavily_service.search_multiple(
            search_terms=search_terms,
            max_results_per_search=2  # 2 results per search term
        )

        # Convert to our schema
        search_results = self._convert_raw_results_to_schema(raw_results)
        search_duration = time.time() - search_start

        return WebSearchResults(
            results=search_results,
            total_results=len(search_results),
            query=original_query,
            search_terms_used=search_terms,
            search_duration=round(search_duration, 2)
        )

    def _convert_raw_results_to_schema(self, raw_results: list) -> list[SearchResult]:
        """Convert raw Tavily results to SearchResult objects"""

        search_results = []
        
        for result in raw_results:
            try:
                search_result = SearchResult(
                    title=result.get('title', 'No title'),
                    url=result.get('url', ''),
                    content=result.get('content', ''),
                    score=result.get('score', 0.0),
                    calculated_score=result.get('calculated_score'),
                    published_date=result.get('published_date')
                )
                search_results.append(search_result)
            except Exception as e:
                logger.warning(f"Failed to parse search result: {e}")
                continue
        
        return search_results

    def _get_max_searches(self, complexity_score: int) -> int:
        """Determine maximum number of searches based on query complexity"""
        if complexity_score <= 3:
            return 2  # Simple queries: 2 searches
        elif complexity_score <= 6:
            return 3  # Moderate queries: 3 searches
        else:
            return 3  # Complex queries: 3 searches
    
    def _create_error_response(self, query: str, analysis, web_results, error: str) -> SearchResponse:
        """Create error response with partial data if available"""
        return SearchResponse(
            original_query=query,
            analysis=analysis,
            web_results=web_results,
            synthesized_response=None,
            status=f"error: {error}",
            timestamp=datetime.now().isoformat()
        )
