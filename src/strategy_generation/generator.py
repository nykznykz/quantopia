"""Main strategy generator using LLM."""

from typing import List, Dict, Any, Optional
import logging
import time

from .llm_client import LLMClient
from .prompt_templates import (
    STRATEGY_GENERATION_SYSTEM_PROMPT,
    create_strategy_generation_prompt,
    create_refinement_prompt,
    create_strategy_family_prompt,
    get_indicator_descriptions
)
from .strategy_parser import StrategyParser, StrategyParseError

logger = logging.getLogger(__name__)


class StrategyGenerator:
    """Generates trading strategies using LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        available_indicators: List[str],
        max_retries: int = 3,
        db: Optional[Any] = None
    ):
        """Initialize strategy generator.

        Args:
            llm_client: LLM client instance
            available_indicators: List of valid indicator names
            max_retries: Maximum retries for failed generations
            db: Optional StrategyDatabase instance for persistent storage
        """
        self.llm_client = llm_client
        self.available_indicators = available_indicators
        self.max_retries = max_retries
        self.parser = StrategyParser(available_indicators)
        self.db = db

        # Track generated strategies to avoid duplicates
        self.generated_strategies = []

        logger.info(
            f"Initialized StrategyGenerator with {len(available_indicators)} indicators"
            f"{' (with database)' if db else ''}"
        )

    def generate_strategy(
        self,
        strategy_type: Optional[str] = None,
        avoid_similar_to: Optional[List[Dict]] = None,
        temperature: float = 0.8
    ) -> Dict[str, Any]:
        """Generate a single trading strategy.

        Args:
            strategy_type: Desired strategy type (optional)
            avoid_similar_to: List of existing strategies to avoid
            temperature: LLM temperature (higher = more creative)

        Returns:
            Parsed strategy dict

        Raises:
            StrategyParseError: If generation fails after retries
        """
        avoid_list = avoid_similar_to or self.generated_strategies[-10:]  # Last 10

        prompt = create_strategy_generation_prompt(
            strategy_type=strategy_type,
            num_strategies=1,
            indicators_list=self.available_indicators,
            avoid_similar_to=avoid_list
        )

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Generating strategy (attempt {attempt}/{self.max_retries})"
                )

                # Generate with LLM
                response_json = self.llm_client.generate_json(
                    prompt=prompt,
                    system_prompt=STRATEGY_GENERATION_SYSTEM_PROMPT,
                    temperature=temperature
                )

                # Parse and validate
                import json as json_module
                strategy = self.parser.parse(json_module.dumps(response_json))

                # Add metadata
                strategy['generated_at'] = time.time()
                strategy['generation_attempt'] = attempt

                # Track generated strategy
                self.generated_strategies.append(strategy)

                # Store in database if available
                if self.db:
                    try:
                        strategy_id = self.db.store_strategy(
                            name=strategy['strategy_name'],
                            strategy_type=strategy['strategy_type'],
                            indicators=strategy.get('indicators', []),
                            description=strategy.get('description'),
                            parameters=strategy.get('parameters'),
                            llm_model=self.llm_client.model if hasattr(self.llm_client, 'model') else None,
                            status='generated'
                        )
                        strategy['db_id'] = strategy_id
                        logger.info(f"Stored strategy in database with ID: {strategy_id}")
                    except Exception as e:
                        logger.warning(f"Failed to store strategy in database: {e}")

                logger.info(
                    f"✓ Successfully generated: {strategy['strategy_name']} "
                    f"({strategy['strategy_type']})"
                )

                return strategy

            except (StrategyParseError, ValueError) as e:
                last_error = e
                logger.warning(f"Attempt {attempt} failed: {e}")

                if attempt < self.max_retries:
                    # Add feedback to prompt for retry
                    prompt += f"\n\nPrevious attempt failed with error: {e}\nPlease fix and try again."
                    time.sleep(1)  # Brief delay between retries

        # All retries failed
        raise StrategyParseError(
            f"Failed to generate valid strategy after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def generate_batch(
        self,
        num_strategies: int = 10,
        strategy_types: Optional[List[str]] = None,
        temperature: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Generate a batch of diverse strategies.

        Args:
            num_strategies: Number of strategies to generate
            strategy_types: Optional list of strategy types to focus on
            temperature: LLM temperature

        Returns:
            List of parsed strategy dicts
        """
        strategies = []

        logger.info(f"Generating batch of {num_strategies} strategies")

        # If strategy types specified, distribute evenly
        if strategy_types:
            strategies_per_type = num_strategies // len(strategy_types)
            remaining = num_strategies % len(strategy_types)

            for i, strategy_type in enumerate(strategy_types):
                count = strategies_per_type + (1 if i < remaining else 0)

                for _ in range(count):
                    try:
                        strategy = self.generate_strategy(
                            strategy_type=strategy_type,
                            temperature=temperature
                        )
                        strategies.append(strategy)
                    except StrategyParseError as e:
                        logger.error(f"Failed to generate {strategy_type} strategy: {e}")

        else:
            # Generate without type constraint
            for i in range(num_strategies):
                try:
                    strategy = self.generate_strategy(temperature=temperature)
                    strategies.append(strategy)
                except StrategyParseError as e:
                    logger.error(f"Failed to generate strategy {i+1}: {e}")

        logger.info(
            f"✓ Batch generation complete: {len(strategies)}/{num_strategies} successful"
        )

        return strategies

    def refine_strategy(
        self,
        failed_strategy: Dict[str, Any],
        backtest_results: Dict[str, Any],
        failure_analysis: str,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Refine a failed strategy based on backtest results.

        Args:
            failed_strategy: Original strategy that failed
            backtest_results: Backtest metrics
            failure_analysis: Analysis of failure modes
            temperature: LLM temperature

        Returns:
            Refined strategy dict

        Raises:
            StrategyParseError: If refinement fails
        """
        prompt = create_refinement_prompt(
            strategy=failed_strategy,
            backtest_results=backtest_results,
            failure_analysis=failure_analysis
        )

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Refining strategy '{failed_strategy['strategy_name']}' "
                    f"(attempt {attempt}/{self.max_retries})"
                )

                # Generate refinement
                response_json = self.llm_client.generate_json(
                    prompt=prompt,
                    system_prompt=STRATEGY_GENERATION_SYSTEM_PROMPT,
                    temperature=temperature
                )

                # Parse and validate
                refined_strategy = self.parser.parse(str(response_json))

                # Add metadata
                refined_strategy['parent_strategy'] = failed_strategy['strategy_name']
                refined_strategy['refinement_iteration'] = attempt
                refined_strategy['generated_at'] = time.time()

                # Store in database if available
                if self.db:
                    try:
                        parent_id = failed_strategy.get('db_id')
                        parent_generation = 0
                        if parent_id:
                            parent_data = self.db.get_strategy_by_id(parent_id)
                            if parent_data:
                                parent_generation = parent_data.get('generation', 0)

                        strategy_id = self.db.store_strategy(
                            name=refined_strategy['strategy_name'],
                            strategy_type=refined_strategy['strategy_type'],
                            indicators=refined_strategy.get('indicators', []),
                            description=refined_strategy.get('description'),
                            parameters=refined_strategy.get('parameters'),
                            parent_id=parent_id,
                            generation=parent_generation + 1,
                            refinement_type='refinement',
                            llm_model=self.llm_client.model if hasattr(self.llm_client, 'model') else None,
                            status='generated'
                        )
                        refined_strategy['db_id'] = strategy_id
                        logger.info(f"Stored refined strategy in database with ID: {strategy_id}")
                    except Exception as e:
                        logger.warning(f"Failed to store refined strategy in database: {e}")

                logger.info(f"✓ Successfully refined strategy: {refined_strategy['strategy_name']}")

                return refined_strategy

            except (StrategyParseError, ValueError) as e:
                last_error = e
                logger.warning(f"Refinement attempt {attempt} failed: {e}")
                time.sleep(1)

        raise StrategyParseError(
            f"Failed to refine strategy after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def generate_strategy_family(
        self,
        base_strategy: Dict[str, Any],
        num_variations: int = 3,
        temperature: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Generate variations of a successful strategy.

        Args:
            base_strategy: Base strategy to create variations of
            num_variations: Number of variations to generate
            temperature: LLM temperature

        Returns:
            List of strategy variation dicts
        """
        prompt = create_strategy_family_prompt(
            base_strategy=base_strategy,
            num_variations=num_variations
        )

        try:
            logger.info(
                f"Generating {num_variations} variations of "
                f"'{base_strategy['strategy_name']}'"
            )

            # Generate variations
            response_json = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=STRATEGY_GENERATION_SYSTEM_PROMPT,
                temperature=temperature
            )

            # Parse batch
            variations = self.parser.parse_batch(str(response_json))

            # Add metadata and store in database
            parent_id = base_strategy.get('db_id')
            parent_generation = 0
            if self.db and parent_id:
                parent_data = self.db.get_strategy_by_id(parent_id)
                if parent_data:
                    parent_generation = parent_data.get('generation', 0)

            for i, variation in enumerate(variations):
                variation['parent_strategy'] = base_strategy['strategy_name']
                variation['variation_number'] = i + 1
                variation['generated_at'] = time.time()

                # Store in database if available
                if self.db:
                    try:
                        strategy_id = self.db.store_strategy(
                            name=variation['strategy_name'],
                            strategy_type=variation['strategy_type'],
                            indicators=variation.get('indicators', []),
                            description=variation.get('description'),
                            parameters=variation.get('parameters'),
                            parent_id=parent_id,
                            generation=parent_generation + 1,
                            refinement_type='family_member',
                            llm_model=self.llm_client.model if hasattr(self.llm_client, 'model') else None,
                            status='generated'
                        )
                        variation['db_id'] = strategy_id
                        logger.info(f"Stored family variation in database with ID: {strategy_id}")
                    except Exception as e:
                        logger.warning(f"Failed to store variation in database: {e}")

            logger.info(f"✓ Generated {len(variations)} strategy variations")

            return variations

        except (StrategyParseError, ValueError) as e:
            logger.error(f"Failed to generate strategy family: {e}")
            return []

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about generated strategies.

        Returns:
            Dict with generation statistics
        """
        if not self.generated_strategies:
            return {
                'total_generated': 0,
                'by_type': {},
                'by_regime': {}
            }

        # Count by type
        type_counts = {}
        for strategy in self.generated_strategies:
            strategy_type = strategy.get('strategy_type', 'unknown')
            type_counts[strategy_type] = type_counts.get(strategy_type, 0) + 1

        # Count by regime
        regime_counts = {}
        for strategy in self.generated_strategies:
            regime = strategy.get('market_regime', 'any')
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        return {
            'total_generated': len(self.generated_strategies),
            'by_type': type_counts,
            'by_regime': regime_counts,
            'last_generated': self.generated_strategies[-1]['strategy_name'] if self.generated_strategies else None
        }
