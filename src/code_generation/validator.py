"""Code validator for generated strategy classes."""

import ast
import sys
import os
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CodeValidator:
    """Validates generated strategy code."""

    def __init__(self):
        """Initialize validator."""
        pass

    def validate_comprehensive(
        self,
        code: str,
        strategy_metadata: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Comprehensive validation of generated code.

        Args:
            code: Python code string
            strategy_metadata: Original strategy metadata

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Run all validation checks
        errors.extend(self._check_syntax(code))
        errors.extend(self._check_imports(code))
        errors.extend(self._check_class_structure(code))
        errors.extend(self._check_methods(code))
        errors.extend(self._check_indicators(code, strategy_metadata))
        errors.extend(self._check_logic_completeness(code, strategy_metadata))

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("✓ Comprehensive validation passed")
        else:
            logger.error(f"✗ Validation failed with {len(errors)} errors")
            for error in errors:
                logger.error(f"  - {error}")

        return is_valid, errors

    def _check_syntax(self, code: str) -> List[str]:
        """Check for syntax errors."""
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        return errors

    def _check_imports(self, code: str) -> List[str]:
        """Check for required imports."""
        errors = []

        required_imports = {
            'pd': ['pandas', 'pd'],
            'np': ['numpy', 'np'],
            'BaseStrategy': ['BaseStrategy', 'strategy_base'],
            'INDICATOR_REGISTRY': ['INDICATOR_REGISTRY', 'indicators'],
        }

        for name, patterns in required_imports.items():
            if not any(pattern in code for pattern in patterns):
                errors.append(f"Missing required import: {name}")

        return errors

    def _check_class_structure(self, code: str) -> List[str]:
        """Check class structure and inheritance."""
        errors = []

        # Parse AST
        try:
            tree = ast.parse(code)
        except:
            return errors  # Syntax errors already caught

        # Find class definitions
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        if not classes:
            errors.append("No class definition found")
            return errors

        # Check that at least one class inherits from BaseStrategy
        has_base_strategy = False
        for cls in classes:
            for base in cls.bases:
                if isinstance(base, ast.Name) and base.id == 'BaseStrategy':
                    has_base_strategy = True
                    break

        if not has_base_strategy:
            errors.append("No class inherits from BaseStrategy")

        return errors

    def _check_methods(self, code: str) -> List[str]:
        """Check for required methods."""
        errors = []

        required_methods = [
            '__init__',
            'calculate_indicators',
            'should_enter',
            'should_exit',
            'stop_loss',
            'take_profit'
        ]

        try:
            tree = ast.parse(code)
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

            if not classes:
                return errors

            # Get methods from the main strategy class
            strategy_class = classes[0]  # Assume first class is the strategy
            method_names = [node.name for node in strategy_class.body if isinstance(node, ast.FunctionDef)]

            for required_method in required_methods:
                if required_method not in method_names:
                    errors.append(f"Missing required method: {required_method}")

        except Exception as e:
            errors.append(f"Error checking methods: {e}")

        return errors

    def _check_indicators(self, code: str, strategy_metadata: Dict[str, Any]) -> List[str]:
        """Check that indicators from metadata are properly initialized."""
        errors = []

        indicators = strategy_metadata.get('indicators', [])
        indicator_names = [ind.get('name') for ind in indicators if ind.get('name')]

        for indicator_name in indicator_names:
            # Check if indicator is accessed from INDICATOR_REGISTRY
            if f"INDICATOR_REGISTRY['{indicator_name}']" not in code and \
               f'INDICATOR_REGISTRY["{indicator_name}"]' not in code:
                errors.append(
                    f"Indicator '{indicator_name}' from metadata not initialized from INDICATOR_REGISTRY"
                )

        return errors

    def _check_logic_completeness(self, code: str, strategy_metadata: Dict[str, Any]) -> List[str]:
        """Check that entry/exit rules are implemented."""
        errors = []

        # Check should_enter has logic
        if 'def should_enter' in code:
            should_enter_start = code.find('def should_enter')
            should_enter_end = code.find('\n    def ', should_enter_start + 1)
            if should_enter_end == -1:
                should_enter_end = len(code)

            should_enter_body = code[should_enter_start:should_enter_end]

            # Check if method just has pass or is empty
            lines = [line.strip() for line in should_enter_body.split('\n') if line.strip()]
            # Filter out docstrings, comments, and def line
            logic_lines = [
                line for line in lines
                if not line.startswith('#') and
                   not line.startswith('"""') and
                   not line.startswith("'''") and
                   not line.startswith('def ')
            ]

            if len(logic_lines) == 0 or all(line == 'pass' for line in logic_lines):
                errors.append("should_enter method has no logic (only 'pass')")

        # Check should_exit has logic
        if 'def should_exit' in code:
            should_exit_start = code.find('def should_exit')
            should_exit_end = code.find('\n    def ', should_exit_start + 1)
            if should_exit_end == -1:
                should_exit_end = len(code)

            should_exit_body = code[should_exit_start:should_exit_end]

            lines = [line.strip() for line in should_exit_body.split('\n') if line.strip()]
            logic_lines = [
                line for line in lines
                if not line.startswith('#') and
                   not line.startswith('"""') and
                   not line.startswith("'''") and
                   not line.startswith('def ')
            ]

            if len(logic_lines) == 0 or all(line == 'pass' for line in logic_lines):
                errors.append("should_exit method has no logic (only 'pass')")

        return errors

    def can_import_strategy(self, code: str, strategy_name: str) -> Tuple[bool, Optional[str]]:
        """Test if strategy can be imported and instantiated.

        Args:
            code: Python code string
            strategy_name: Expected strategy class name

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Create a temporary module
            import types
            module = types.ModuleType('temp_strategy')

            # Execute code in module namespace
            exec(code, module.__dict__)

            # Try to find the strategy class
            strategy_class = None
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and name.endswith('Strategy'):
                    strategy_class = obj
                    break

            if strategy_class is None:
                return False, "No strategy class found in generated code"

            logger.info(f"✓ Successfully imported strategy class: {strategy_class.__name__}")
            return True, None

        except Exception as e:
            return False, f"Failed to import strategy: {str(e)}"
