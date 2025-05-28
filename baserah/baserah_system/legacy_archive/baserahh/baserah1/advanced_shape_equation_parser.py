#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Shape Equation Parser for Basira System

This module provides a comprehensive parser for the "Advanced Shape Equation" format,
which represents shapes, their properties, styles, and animations in a text-based format.
The parser converts these text equations into structured data that can be used by other
components of the Basira System.

Author: Basira System Development Team
Version: 1.0.0
"""

import re
import ast
import copy
from typing import Dict, List, Tuple, Union, Optional, Any
from enum import Enum, auto
import pyparsing as pp

class TokenType(Enum):
    """Enumeration of token types used in the shape equation parsing."""
    SHAPE_TYPE = auto()
    PARAMETER = auto()
    RANGE = auto()
    STYLE = auto()
    ANIMATION = auto()
    COMMENT = auto()
    REFERENCE = auto()
    VARIABLE = auto()
    FUNCTION = auto()
    OPERATOR = auto()

class AnimationType(Enum):
    """Enumeration of animation types supported in shape equations."""
    TRANSFORM = auto()
    PARAMETER = auto()
    STYLE = auto()
    VISIBILITY = auto()
    CUSTOM = auto()

class ParseError(Exception):
    """Exception raised for errors during parsing of shape equations."""
    def __init__(self, message: str, position: Optional[int] = None, 
                 line: Optional[int] = None, column: Optional[int] = None):
        """
        Initialize a new ParseError.
        
        Args:
            message: Error description
            position: Character position in the input string where the error occurred
            line: Line number where the error occurred
            column: Column number where the error occurred
        """
        self.message = message
        self.position = position
        self.line = line
        self.column = column
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message with position information if available."""
        if self.line is not None and self.column is not None:
            return f"Parse error at line {self.line}, column {self.column}: {self.message}"
        elif self.position is not None:
            return f"Parse error at position {self.position}: {self.message}"
        else:
            return f"Parse error: {self.message}"

class AdvancedShapeEquationParser:
    """
    Parser for Advanced Shape Equations.
    
    This class provides methods to parse text-based shape equations into structured data
    that represents shapes, their parameters, styles, and animations.
    """
    
    def __init__(self):
        """Initialize the parser with grammar rules."""
        self.parser = self._setup_parser()
        self._reset_state()
    
    def _reset_state(self):
        """Reset the internal state of the parser."""
        self.current_position = 0
        self.current_line = 1
        self.current_column = 1
        self.input_text = ""
    
    def _update_position(self, string: str, loc: int, toks: List):
        """
        Update the current position, line, and column during parsing.
        
        Args:
            string: The input string being parsed
            loc: The current location in the string
            toks: The tokens found at this location
            
        Returns:
            The tokens unchanged
        """
        self.current_position = loc
        # Calculate line and column from the input text up to this position
        text_so_far = string[:loc]
        self.current_line = text_so_far.count('\n') + 1
        if '\n' in text_so_far:
            self.current_column = loc - text_so_far.rindex('\n')
        else:
            self.current_column = loc + 1
        return toks
    
    def _setup_parser(self) -> pp.ParserElement:
        """
        Set up the pyparsing grammar for shape equations.
        
        Returns:
            A pyparsing parser element that can parse shape equations
        """
        # Define basic elements
        pp.ParserElement.setDefaultWhitespaceChars(" \t\r\n")
        
        # Helper for tracking position
        def track_pos(expr):
            return expr.addParseAction(self._update_position)
        
        # Comments
        comment = pp.Suppress(pp.Literal("#") + pp.restOfLine)
        
        # Numbers
        integer = pp.Regex(r"[+-]?\d+").setParseAction(lambda t: int(t[0]))
        float_number = pp.Regex(r"[+-]?\d+\.\d*([eE][+-]?\d+)?").setParseAction(lambda t: float(t[0]))
        number = float_number | integer
        
        # Strings
        quoted_string = pp.QuotedString('"', escChar='\\') | pp.QuotedString("'", escChar='\\')
        
        # Colors (hex format)
        hex_color = pp.Regex(r"#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?")
        
        # Boolean values
        boolean = (pp.Keyword("true") | pp.Keyword("false")).setParseAction(
            lambda t: t[0].lower() == "true")
        
        # Basic value types
        value = track_pos(number | quoted_string | hex_color | boolean)
        
        # Tuples and lists
        lpar = pp.Suppress(pp.Literal("("))
        rpar = pp.Suppress(pp.Literal(")"))
        lbrack = pp.Suppress(pp.Literal("["))
        rbrack = pp.Suppress(pp.Literal("]"))
        comma = pp.Suppress(pp.Literal(","))
        
        # Forward declaration for recursive definitions
        expr = pp.Forward()
        
        # Tuple definition
        tuple_value = pp.Group(lpar + pp.Optional(expr + pp.ZeroOrMore(comma + expr)) + rpar)
        tuple_value.setParseAction(lambda t: tuple(t[0]))
        
        # List definition
        list_value = pp.Group(lbrack + pp.Optional(expr + pp.ZeroOrMore(comma + expr)) + rbrack)
        list_value.setParseAction(lambda t: list(t[0]))
        
        # Update expr with complex types
        expr << (tuple_value | list_value | value)
        
        # Parameter name
        param_name = pp.Word(pp.alphas + "_", pp.alphanums + "_")
        
        # Parameter assignment
        equals = pp.Suppress(pp.Literal("="))
        parameter = pp.Group(param_name + equals + expr).setResultsName("parameters", listAllMatches=True)
        
        # Range definition
        range_op = pp.Suppress(pp.Literal(".."))
        range_def = pp.Group(number + range_op + number).setResultsName("ranges", listAllMatches=True)
        
        # Style properties
        style_name = pp.Word(pp.alphas + "_", pp.alphanums + "_")
        style_value = expr
        style_assign = pp.Suppress(pp.Literal(":"))
        style_property = pp.Group(style_name + style_assign + style_value)
        
        # Style block
        lcurly = pp.Suppress(pp.Literal("{"))
        rcurly = pp.Suppress(pp.Literal("}"))
        style_block = pp.Group(
            lcurly + 
            pp.Optional(style_property + pp.ZeroOrMore(comma + style_property)) + 
            rcurly
        ).setResultsName("styles", listAllMatches=True)
        
        # Animation keyframe
        time_value_pair = pp.Group(number + comma + expr)
        keyframe_list = pp.Group(
            lbrack + 
            pp.Optional(time_value_pair + pp.ZeroOrMore(comma + time_value_pair)) + 
            rbrack
        )
        
        # Animation property
        animation_marker = pp.Suppress(pp.Literal("@"))
        animation_property = pp.Group(
            animation_marker + param_name + equals + keyframe_list
        ).setResultsName("animations", listAllMatches=True)
        
        # Shape type
        shape_type = pp.Word(pp.alphas + "_", pp.alphanums + "_").setResultsName("shape_type")
        
        # Complete shape equation
        shape_equation = (
            shape_type + 
            pp.Optional(lpar + pp.Optional(parameter + pp.ZeroOrMore(comma + parameter)) + rpar) +
            pp.Optional(range_def) +
            pp.Optional(style_block) +
            pp.ZeroOrMore(animation_property)
        )
        
        # Full parser with optional comments
        parser = pp.Optional(comment) + shape_equation + pp.Optional(comment)
        
        return parser
    
    def parse(self, equation_text: str) -> Dict[str, Any]:
        """
        Parse a shape equation string into a structured dictionary.
        
        Args:
            equation_text: The shape equation text to parse
            
        Returns:
            A dictionary containing the parsed shape information
            
        Raises:
            ParseError: If the equation cannot be parsed correctly
        """
        self._reset_state()
        self.input_text = equation_text
        
        try:
            # Parse the equation
            parse_results = self.parser.parseString(equation_text, parseAll=True)
            
            # Extract the results into a structured dictionary
            result = self._extract_parse_results(parse_results)
            
            return result
        
        except pp.ParseException as e:
            # Convert pyparsing exception to our custom ParseError
            raise ParseError(
                message=str(e),
                position=e.loc,
                line=e.lineno,
                column=e.col
            )
        except Exception as e:
            # Handle other exceptions
            raise ParseError(f"Unexpected error during parsing: {str(e)}")
    
    def _extract_parse_results(self, parse_results) -> Dict[str, Any]:
        """
        Extract and structure the parse results.
        
        Args:
            parse_results: The pyparsing parse results
            
        Returns:
            A structured dictionary with the parsed information
        """
        result = {
            "shape_type": parse_results.shape_type,
            "parameters": {},
            "ranges": [],
            "styles": {},
            "animations": {}
        }
        
        # Extract parameters
        if hasattr(parse_results, "parameters"):
            for param in parse_results.parameters:
                name, value = param[0], param[1]
                result["parameters"][name] = value
        
        # Extract ranges
        if hasattr(parse_results, "ranges"):
            for range_item in parse_results.ranges:
                start, end = range_item[0], range_item[1]
                result["ranges"].append((start, end))
        
        # Extract styles
        if hasattr(parse_results, "styles"):
            for style_block in parse_results.styles:
                for style_item in style_block:
                    name, value = style_item[0], style_item[1]
                    result["styles"][name] = value
        
        # Extract animations
        if hasattr(parse_results, "animations"):
            for anim in parse_results.animations:
                prop_name, keyframes = anim[0], anim[1]
                # Convert keyframes to list of (time, value) tuples
                keyframe_list = []
                for kf in keyframes:
                    time, value = kf[0], kf[1]
                    keyframe_list.append((time, value))
                result["animations"][prop_name] = keyframe_list
        
        return result
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a file containing multiple shape equations.
        
        Args:
            file_path: Path to the file containing shape equations
            
        Returns:
            A list of dictionaries, each containing a parsed shape
            
        Raises:
            FileNotFoundError: If the file cannot be found
            ParseError: If any equation cannot be parsed correctly
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split the content by semicolons or newlines
            equations = re.split(r';|\n\s*\n', content)
            
            results = []
            for eq in equations:
                eq = eq.strip()
                if eq:  # Skip empty equations
                    try:
                        parsed = self.parse(eq)
                        results.append(parsed)
                    except ParseError as e:
                        # Add file information to the error
                        raise ParseError(
                            f"Error in file {file_path}: {e.message}",
                            position=e.position,
                            line=e.line,
                            column=e.column
                        )
            
            return results
        
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    def parse_style(self, style_text: str) -> Dict[str, Any]:
        """
        Parse a style block string into a dictionary.
        
        Args:
            style_text: The style block text to parse (e.g., "{color: #FF0000, width: 2}")
            
        Returns:
            A dictionary of style properties
            
        Raises:
            ParseError: If the style block cannot be parsed correctly
        """
        try:
            # Create a mini-parser just for style blocks
            style_name = pp.Word(pp.alphas + "_", pp.alphanums + "_")
            
            # Numbers
            integer = pp.Regex(r"[+-]?\d+").setParseAction(lambda t: int(t[0]))
            float_number = pp.Regex(r"[+-]?\d+\.\d*([eE][+-]?\d+)?").setParseAction(lambda t: float(t[0]))
            number = float_number | integer
            
            # Strings
            quoted_string = pp.QuotedString('"', escChar='\\') | pp.QuotedString("'", escChar='\\')
            
            # Colors (hex format)
            hex_color = pp.Regex(r"#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?")
            
            # Boolean values
            boolean = (pp.Keyword("true") | pp.Keyword("false")).setParseAction(
                lambda t: t[0].lower() == "true")
            
            # Basic value types
            value = number | quoted_string | hex_color | boolean
            
            # Style property
            style_assign = pp.Suppress(pp.Literal(":"))
            comma = pp.Suppress(pp.Literal(","))
            style_property = pp.Group(style_name + style_assign + value)
            
            # Style block
            lcurly = pp.Suppress(pp.Literal("{"))
            rcurly = pp.Suppress(pp.Literal("}"))
            style_block = (
                lcurly + 
                pp.Optional(style_property + pp.ZeroOrMore(comma + style_property)) + 
                rcurly
            )
            
            # Parse the style text
            parse_results = style_block.parseString(style_text, parseAll=True)
            
            # Convert to dictionary
            style_dict = {}
            for item in parse_results:
                name, value = item[0], item[1]
                style_dict[name] = value
            
            return style_dict
        
        except pp.ParseException as e:
            raise ParseError(
                message=f"Invalid style block: {str(e)}",
                position=e.loc,
                line=e.lineno,
                column=e.col
            )
    
    def parse_animation(self, animation_text: str) -> Dict[str, List[Tuple[float, Any]]]:
        """
        Parse animation definitions into a structured format.
        
        Args:
            animation_text: The animation text to parse (e.g., "@x=[(0,10), (1,20)]")
            
        Returns:
            A dictionary mapping property names to lists of (time, value) tuples
            
        Raises:
            ParseError: If the animation text cannot be parsed correctly
        """
        try:
            # Create a mini-parser just for animation properties
            param_name = pp.Word(pp.alphas + "_", pp.alphanums + "_")
            
            # Numbers
            integer = pp.Regex(r"[+-]?\d+").setParseAction(lambda t: int(t[0]))
            float_number = pp.Regex(r"[+-]?\d+\.\d*([eE][+-]?\d+)?").setParseAction(lambda t: float(t[0]))
            number = float_number | integer
            
            # Strings
            quoted_string = pp.QuotedString('"', escChar='\\') | pp.QuotedString("'", escChar='\\')
            
            # Colors (hex format)
            hex_color = pp.Regex(r"#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?")
            
            # Boolean values
            boolean = (pp.Keyword("true") | pp.Keyword("false")).setParseAction(
                lambda t: t[0].lower() == "true")
            
            # Basic value types
            value = number | quoted_string | hex_color | boolean
            
            # Punctuation
            lpar = pp.Suppress(pp.Literal("("))
            rpar = pp.Suppress(pp.Literal(")"))
            lbrack = pp.Suppress(pp.Literal("["))
            rbrack = pp.Suppress(pp.Literal("]"))
            comma = pp.Suppress(pp.Literal(","))
            equals = pp.Suppress(pp.Literal("="))
            
            # Time-value pair
            time_value_pair = pp.Group(lpar + number + comma + value + rpar)
            
            # Keyframe list
            keyframe_list = pp.Group(
                lbrack + 
                pp.Optional(time_value_pair + pp.ZeroOrMore(comma + time_value_pair)) + 
                rbrack
            )
            
            # Animation property
            animation_marker = pp.Suppress(pp.Literal("@"))
            animation_property = pp.Group(
                animation_marker + param_name + equals + keyframe_list
            )
            
            # Multiple animation properties
            animations = pp.OneOrMore(animation_property)
            
            # Parse the animation text
            parse_results = animations.parseString(animation_text, parseAll=True)
            
            # Convert to dictionary
            animation_dict = {}
            for anim in parse_results:
                prop_name, keyframes = anim[0], anim[1]
                # Convert keyframes to list of (time, value) tuples
                keyframe_list = []
                for kf in keyframes:
                    time, value = kf[0], kf[1]
                    keyframe_list.append((time, value))
                animation_dict[prop_name] = keyframe_list
            
            return animation_dict
        
        except pp.ParseException as e:
            raise ParseError(
                message=f"Invalid animation definition: {str(e)}",
                position=e.loc,
                line=e.lineno,
                column=e.col
            )
    
    def get_animation_properties(self, parsed_shape: Dict[str, Any]) -> List[str]:
        """
        Get a list of all animated properties in a parsed shape.
        
        Args:
            parsed_shape: A dictionary containing parsed shape information
            
        Returns:
            A list of property names that have animations defined
        """
        if "animations" in parsed_shape:
            return list(parsed_shape["animations"].keys())
        return []
    
    def get_animation_timespan(self, parsed_shape: Dict[str, Any]) -> Tuple[float, float]:
        """
        Get the total timespan covered by all animations in a parsed shape.
        
        Args:
            parsed_shape: A dictionary containing parsed shape information
            
        Returns:
            A tuple of (start_time, end_time) representing the animation timespan
        """
        if "animations" not in parsed_shape or not parsed_shape["animations"]:
            return (0.0, 0.0)
        
        min_time = float('inf')
        max_time = float('-inf')
        
        for prop, keyframes in parsed_shape["animations"].items():
            for time, _ in keyframes:
                min_time = min(min_time, time)
                max_time = max(max_time, time)
        
        return (min_time, max_time) if min_time <= max_time else (0.0, 0.0)
    
    def interpolate_value_at_time(self, keyframes: List[Tuple[float, Any]], 
                                 time: float) -> Any:
        """
        Interpolate a value at a specific time based on keyframes.
        
        Args:
            keyframes: A list of (time, value) tuples
            time: The time at which to interpolate
            
        Returns:
            The interpolated value at the specified time
            
        Raises:
            ValueError: If keyframes is empty or if values cannot be interpolated
        """
        if not keyframes:
            raise ValueError("Cannot interpolate with empty keyframes")
        
        # Sort keyframes by time
        sorted_keyframes = sorted(keyframes, key=lambda kf: kf[0])
        
        # If time is before first keyframe, return first value
        if time <= sorted_keyframes[0][0]:
            return sorted_keyframes[0][1]
        
        # If time is after last keyframe, return last value
        if time >= sorted_keyframes[-1][0]:
            return sorted_keyframes[-1][1]
        
        # Find the two keyframes that surround the target time
        for i in range(len(sorted_keyframes) - 1):
            t1, v1 = sorted_keyframes[i]
            t2, v2 = sorted_keyframes[i + 1]
            
            if t1 <= time < t2:
                # Calculate the interpolation factor
                factor = (time - t1) / (t2 - t1) if t2 != t1 else 0
                
                # Interpolate based on value type
                try:
                    # Numeric values
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        return v1 + factor * (v2 - v1)
                    
                    # Color values (hex strings)
                    elif isinstance(v1, str) and isinstance(v2, str) and v1.startswith('#') and v2.startswith('#'):
                        # Parse hex colors
                        r1, g1, b1 = int(v1[1:3], 16), int(v1[3:5], 16), int(v1[5:7], 16)
                        r2, g2, b2 = int(v2[1:3], 16), int(v2[3:5], 16), int(v2[5:7], 16)
                        
                        # Interpolate RGB values
                        r = int(r1 + factor * (r2 - r1))
                        g = int(g1 + factor * (g2 - g1))
                        b = int(b1 + factor * (b2 - b1))
                        
                        # Handle alpha if present
                        if len(v1) >= 9 and len(v2) >= 9:
                            a1, a2 = int(v1[7:9], 16), int(v2[7:9], 16)
                            a = int(a1 + factor * (a2 - a1))
                            return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
                        else:
                            return f"#{r:02x}{g:02x}{b:02x}"
                    
                    # Boolean values - step function (no interpolation)
                    elif isinstance(v1, bool) and isinstance(v2, bool):
                        return v1 if factor < 0.5 else v2
                    
                    # String values - step function (no interpolation)
                    elif isinstance(v1, str) and isinstance(v2, str):
                        return v1 if factor < 0.5 else v2
                    
                    # Tuples/lists of numbers
                    elif (isinstance(v1, (tuple, list)) and isinstance(v2, (tuple, list)) and
                          len(v1) == len(v2) and all(isinstance(x, (int, float)) for x in v1 + v2)):
                        return tuple(v1[i] + factor * (v2[i] - v1[i]) for i in range(len(v1)))
                    
                    # Default: step function for incompatible types
                    else:
                        return v1 if factor < 0.5 else v2
                
                except Exception as e:
                    raise ValueError(f"Error interpolating between {v1} and {v2}: {str(e)}")
        
        # This should not happen if the keyframes are properly sorted
        return sorted_keyframes[-1][1]
    
    def get_shape_at_time(self, parsed_shape: Dict[str, Any], time: float) -> Dict[str, Any]:
        """
        Get the state of a shape at a specific time by interpolating all animated properties.
        
        Args:
            parsed_shape: A dictionary containing parsed shape information
            time: The time at which to get the shape state
            
        Returns:
            A new dictionary representing the shape state at the specified time
        """
        # Create a deep copy of the original shape
        result = copy.deepcopy(parsed_shape)
        
        # If there are no animations, return the original shape
        if "animations" not in parsed_shape or not parsed_shape["animations"]:
            return result
        
        # Process each animated property
        for prop, keyframes in parsed_shape["animations"].items():
            # Skip empty keyframes
            if not keyframes:
                continue
            
            # Determine where this property belongs (parameters or styles)
            if prop in result["parameters"]:
                # Animated parameter
                result["parameters"][prop] = self.interpolate_value_at_time(keyframes, time)
            elif prop in result["styles"]:
                # Animated style
                result["styles"][prop] = self.interpolate_value_at_time(keyframes, time)
            else:
                # This could be a special property or a new property
                # For now, we'll add it to parameters
                result["parameters"][prop] = self.interpolate_value_at_time(keyframes, time)
        
        return result


# Example usage
if __name__ == "__main__":
    parser = AdvancedShapeEquationParser()
    
    # Example shape equation with animation
    example_equation = """
    Circle(x=100, y=150, radius=50) {color: #FF0000, stroke_width: 2, fill: true}
    @x=[(0, 100), (1, 200), (2, 100)]
    @radius=[(0, 50), (1, 75), (2, 50)]
    @color=[(0, "#FF0000"), (1, "#00FF00"), (2, "#0000FF")]
    """
    
    try:
        result = parser.parse(example_equation)
        print("Parsed shape:")
        print(f"Type: {result['shape_type']}")
        print(f"Parameters: {result['parameters']}")
        print(f"Styles: {result['styles']}")
        print(f"Animations: {result['animations']}")
        
        # Get animation timespan
        start_time, end_time = parser.get_animation_timespan(result)
        print(f"\nAnimation timespan: {start_time} to {end_time}")
        
        # Get shape at specific time
        time = 0.5
        shape_at_time = parser.get_shape_at_time(result, time)
        print(f"\nShape at time {time}:")
        print(f"Parameters: {shape_at_time['parameters']}")
        print(f"Styles: {shape_at_time['styles']}")
        
    except ParseError as e:
        print(f"Error: {e}")
