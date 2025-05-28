#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Foundation Module for Basira System

This module provides the core mathematical tools and functions for handling
the mathematical aspects of equations used in the Basira System. It enables
symbolic analysis, simplification, evaluation, and algebraic, differential,
and integral operations on equations.

The module works closely with shape equations parsed by AdvancedShapeEquationParser
and behavioral/mathematical equations used by other modules like EquationEvolutionService_EEE
or LogicalReasoningEngine.

Author: Basira System Development Team
Version: 1.0.0
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Union, Callable
from collections import defaultdict
import re
import math


class SymbolicExpression:
    """
    A wrapper around SymPy expressions to facilitate handling and storing
    additional information about them.
    
    This class provides methods for evaluating, simplifying, differentiating,
    integrating, and solving symbolic expressions, as well as converting them
    to string and LaTeX representations.
    """
    
    def __init__(self, expression_str: Optional[str] = None, 
                 sympy_obj: Optional[sp.Expr] = None,
                 variables: Optional[Dict[str, sp.Symbol]] = None):
        """
        Initialize a SymbolicExpression.
        
        Args:
            expression_str: Optional string representation of the expression
            sympy_obj: Optional SymPy expression object
            variables: Optional dictionary mapping variable names to SymPy symbols
            
        Raises:
            ValueError: If neither expression_str nor sympy_obj is provided,
                       or if expression_str cannot be parsed
        """
        self.variables = variables or {}
        
        if sympy_obj is not None:
            # Use the provided SymPy object directly
            self.sympy_expr = sympy_obj
        elif expression_str is not None:
            # Parse the string expression
            try:
                # If variables are provided, use them for parsing
                if self.variables:
                    # Create a dictionary of local variables for sympify
                    local_dict = {name: symbol for name, symbol in self.variables.items()}
                    self.sympy_expr = sp.sympify(expression_str, locals=local_dict)
                else:
                    # Parse without predefined variables
                    self.sympy_expr = sp.sympify(expression_str)
                    
                    # Extract symbols from the parsed expression
                    for symbol in self.sympy_expr.free_symbols:
                        self.variables[symbol.name] = symbol
            except Exception as e:
                raise ValueError(f"Failed to parse expression: {expression_str}. Error: {str(e)}")
        else:
            raise ValueError("Either expression_str or sympy_obj must be provided")
        
        # Ensure all free symbols are in the variables dictionary
        for symbol in self.sympy_expr.free_symbols:
            if symbol.name not in self.variables:
                self.variables[symbol.name] = symbol
    
    def evaluate(self, assignments: Dict[str, float]) -> Optional[float]:
        """
        Evaluate the expression with the given variable assignments.
        
        Args:
            assignments: Dictionary mapping variable names to values
            
        Returns:
            The numerical result of the evaluation, or None if evaluation fails
            
        Example:
            >>> expr = SymbolicExpression(expression_str="x**2 + y")
            >>> expr.evaluate({"x": 2, "y": 3})
            7.0
        """
        try:
            # Create substitution dictionary using the stored symbols
            subs_dict = {}
            for var_name, value in assignments.items():
                if var_name in self.variables:
                    subs_dict[self.variables[var_name]] = value
                else:
                    # If the variable is not in self.variables, try to create a new symbol
                    symbol = sp.Symbol(var_name)
                    subs_dict[symbol] = value
            
            # Substitute and evaluate
            result = float(self.sympy_expr.subs(subs_dict).evalf())
            return result
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return None
    
    def simplify(self) -> 'SymbolicExpression':
        """
        Simplify the expression.
        
        Returns:
            A new SymbolicExpression with the simplified form
            
        Example:
            >>> expr = SymbolicExpression(expression_str="x**2 + x**2")
            >>> simplified = expr.simplify()
            >>> print(simplified.to_string())
            2*x**2
        """
        simplified_expr = sp.simplify(self.sympy_expr)
        return SymbolicExpression(sympy_obj=simplified_expr, variables=self.variables.copy())
    
    def differentiate(self, var_name: str) -> 'SymbolicExpression':
        """
        Compute the derivative of the expression with respect to a variable.
        
        Args:
            var_name: Name of the variable to differentiate with respect to
            
        Returns:
            A new SymbolicExpression representing the derivative
            
        Raises:
            ValueError: If the variable is not found
            
        Example:
            >>> expr = SymbolicExpression(expression_str="x**2 + y*x")
            >>> derivative = expr.differentiate("x")
            >>> print(derivative.to_string())
            2*x + y
        """
        if var_name not in self.variables:
            raise ValueError(f"Variable {var_name} not found in expression")
        
        derivative = sp.diff(self.sympy_expr, self.variables[var_name])
        return SymbolicExpression(sympy_obj=derivative, variables=self.variables.copy())
    
    def integrate(self, var_name: str, 
                 limits: Optional[Tuple[Any, Any]] = None) -> 'SymbolicExpression':
        """
        Compute the integral of the expression with respect to a variable.
        
        Args:
            var_name: Name of the variable to integrate with respect to
            limits: Optional tuple (lower_bound, upper_bound) for definite integration
            
        Returns:
            A new SymbolicExpression representing the integral
            
        Raises:
            ValueError: If the variable is not found
            
        Example:
            >>> expr = SymbolicExpression(expression_str="2*x")
            >>> indefinite_integral = expr.integrate("x")
            >>> print(indefinite_integral.to_string())
            x**2
            >>> definite_integral = expr.integrate("x", (0, 1))
            >>> print(definite_integral.to_string())
            1
        """
        if var_name not in self.variables:
            raise ValueError(f"Variable {var_name} not found in expression")
        
        if limits is None:
            # Indefinite integration
            integral = sp.integrate(self.sympy_expr, self.variables[var_name])
        else:
            # Definite integration
            lower, upper = limits
            integral = sp.integrate(self.sympy_expr, (self.variables[var_name], lower, upper))
        
        return SymbolicExpression(sympy_obj=integral, variables=self.variables.copy())
    
    def solve_for(self, var_name: str) -> List['SymbolicExpression']:
        """
        Solve the equation for a specific variable.
        
        This method assumes the expression is an equation (Eq) or can be set to zero.
        
        Args:
            var_name: Name of the variable to solve for
            
        Returns:
            A list of SymbolicExpression objects representing the solutions
            
        Raises:
            ValueError: If the variable is not found
            
        Example:
            >>> eq = SymbolicExpression(expression_str="Eq(x**2 - 4, 0)")
            >>> solutions = eq.solve_for("x")
            >>> [sol.to_string() for sol in solutions]
            ['-2', '2']
        """
        if var_name not in self.variables:
            raise ValueError(f"Variable {var_name} not found in expression")
        
        # Check if the expression is an equation
        if isinstance(self.sympy_expr, sp.Eq):
            equation = self.sympy_expr
        else:
            # If not an equation, set it equal to zero
            equation = sp.Eq(self.sympy_expr, 0)
        
        # Solve the equation
        solutions = sp.solve(equation, self.variables[var_name])
        
        # Convert solutions to SymbolicExpression objects
        result = []
        for sol in solutions:
            result.append(SymbolicExpression(sympy_obj=sol, variables=self.variables.copy()))
        
        return result
    
    def get_free_symbols(self) -> Set[sp.Symbol]:
        """
        Get the set of free symbols in the expression.
        
        Returns:
            A set of SymPy symbols
            
        Example:
            >>> expr = SymbolicExpression(expression_str="x**2 + y*z")
            >>> symbols = expr.get_free_symbols()
            >>> sorted([str(s) for s in symbols])
            ['x', 'y', 'z']
        """
        return self.sympy_expr.free_symbols
    
    def to_string(self) -> str:
        """
        Convert the expression to a string representation.
        
        Returns:
            String representation of the expression
            
        Example:
            >>> expr = SymbolicExpression(expression_str="x**2 + y")
            >>> expr.to_string()
            'x**2 + y'
        """
        return str(self.sympy_expr)
    
    def to_latex(self) -> str:
        """
        Convert the expression to a LaTeX representation.
        
        Returns:
            LaTeX representation of the expression
            
        Example:
            >>> expr = SymbolicExpression(expression_str="x**2 + y/z")
            >>> expr.to_latex()
            'x^{2} + \\frac{y}{z}'
        """
        return sp.latex(self.sympy_expr)
    
    def get_complexity_score(self) -> float:
        """
        Calculate a complexity score for the expression.
        
        The score is based on the number of operations, depth of the expression tree,
        and number of unique symbols.
        
        Returns:
            A numerical score representing the complexity
            
        Example:
            >>> expr1 = SymbolicExpression(expression_str="x + y")
            >>> expr2 = SymbolicExpression(expression_str="sin(x)**2 + cos(x)**2 + tan(x)")
            >>> expr1.get_complexity_score() < expr2.get_complexity_score()
            True
        """
        # Count operations
        operations_count = len([node for node in sp.preorder_traversal(self.sympy_expr) 
                               if not node.is_Symbol and not node.is_number])
        
        # Count unique symbols
        symbols_count = len(self.get_free_symbols())
        
        # Calculate expression depth
        def get_depth(expr):
            if expr.is_Symbol or expr.is_number:
                return 0
            return 1 + max([get_depth(arg) for arg in expr.args], default=0)
        
        depth = get_depth(self.sympy_expr)
        
        # Combine factors with weights
        complexity = (operations_count * 1.0 + 
                     symbols_count * 0.5 + 
                     depth * 2.0)
        
        return complexity
    
    def __str__(self) -> str:
        """String representation of the expression."""
        return self.to_string()
    
    def __repr__(self) -> str:
        """Detailed string representation of the expression."""
        return f"SymbolicExpression({self.to_string()})"


class ShapeEquationMathAnalyzer:
    """
    Analyzer for the mathematical aspects of shape equations.
    
    This class provides methods for extracting and analyzing the mathematical
    properties of shape components parsed by AdvancedShapeEquationParser.
    """
    
    def __init__(self, parsed_shape_component: Dict[str, Any]):
        """
        Initialize a ShapeEquationMathAnalyzer.
        
        Args:
            parsed_shape_component: Dictionary representing a parsed shape component
        """
        self.component = parsed_shape_component
        
        # Define common symbols
        self.t_sym = sp.Symbol('t')  # Time symbol for animation
        self.x_sym = sp.Symbol('x')  # X coordinate
        self.y_sym = sp.Symbol('y')  # Y coordinate
        self.z_sym = sp.Symbol('z')  # Z coordinate (for 3D shapes)
        
        # Parameter name mappings for different shape types
        self.param_mappings = {
            'circle': ['cx', 'cy', 'radius'],
            'rectangle': ['x', 'y', 'width', 'height'],
            'ellipse': ['cx', 'cy', 'rx', 'ry'],
            'line': ['x1', 'y1', 'x2', 'y2'],
            'path': ['commands'],
            'polygon': ['points'],
            'text': ['x', 'y', 'text']
        }
        
        # Get shape type
        self.shape_type = self._get_shape_type()
    
    def _get_shape_type(self) -> str:
        """
        Extract the shape type from the component.
        
        Returns:
            The shape type as a string
        """
        # Check if shape_type is directly in params
        if 'params' in self.component and isinstance(self.component['params'], dict):
            if 'shape_type' in self.component['params']:
                return self.component['params']['shape_type']
        
        # If not found or params is not a dictionary, return 'unknown'
        return 'unknown'
    
    def get_parameter_as_symbolic_expression(self, param_name: str, 
                                           time_symbol: Optional[sp.Symbol] = None) -> Optional[SymbolicExpression]:
        """
        Get a symbolic expression for a parameter, which may be animated.
        
        Args:
            param_name: Name of the parameter
            time_symbol: Optional symbol to use for time (defaults to self.t_sym)
            
        Returns:
            A SymbolicExpression representing the parameter, or None if not found
            
        Example:
            >>> analyzer = ShapeEquationMathAnalyzer(parsed_component)
            >>> radius_expr = analyzer.get_parameter_as_symbolic_expression('radius')
            >>> if radius_expr:
            >>>     print(radius_expr.to_string())
            1 + 4*t  # For radius with keyframes [(0,1), (1,5)]
        """
        if time_symbol is None:
            time_symbol = self.t_sym
        
        # Check if the parameter is animated
        animations = self.component.get('animations', {})
        if param_name in animations:
            keyframes = animations[param_name]
            return self._create_interpolation_expression(keyframes, time_symbol)
        
        # If not animated, check for static parameter
        params = self.component.get('params', {})
        if isinstance(params, dict) and param_name in params:
            # Parameter is directly accessible by name
            value = params[param_name]
            return SymbolicExpression(sympy_obj=sp.sympify(value), 
                                     variables={'t': time_symbol})
        elif isinstance(params, dict) and self.shape_type in self.param_mappings:
            # Try to find parameter by index in the mapping
            param_names = self.param_mappings[self.shape_type]
            if param_name in param_names and param_name in params:
                value = params[param_name]
                return SymbolicExpression(sympy_obj=sp.sympify(value), 
                                         variables={'t': time_symbol})
        
        # Check in style properties
        style = self.component.get('style', {})
        if param_name in style:
            value = style[param_name]
            # Only return for numeric values
            if isinstance(value, (int, float)):
                return SymbolicExpression(sympy_obj=sp.sympify(value), 
                                         variables={'t': time_symbol})
        
        # Parameter not found
        return None
    
    def _create_interpolation_expression(self, keyframes: List[Dict[str, Any]], 
                                       time_symbol: sp.Symbol) -> Optional[SymbolicExpression]:
        """
        Create a symbolic expression for interpolating between keyframes.
        
        Args:
            keyframes: List of keyframes, each with 'time' and 'value'
            time_symbol: Symbol to use for time
            
        Returns:
            A SymbolicExpression representing the interpolation function
        """
        if not keyframes:
            return None
        
        # Sort keyframes by time
        sorted_keyframes = sorted(keyframes, key=lambda kf: kf['time'])
        
        # If only one keyframe, return constant value
        if len(sorted_keyframes) == 1:
            value = sorted_keyframes[0]['value']
            return SymbolicExpression(sympy_obj=sp.sympify(value), 
                                     variables={'t': time_symbol})
        
        # For multiple keyframes, create a piecewise function
        pieces = []
        
        for i in range(len(sorted_keyframes) - 1):
            t0 = sorted_keyframes[i]['time']
            t1 = sorted_keyframes[i + 1]['time']
            v0 = sorted_keyframes[i]['value']
            v1 = sorted_keyframes[i + 1]['value']
            
            # Linear interpolation: v0 + (v1 - v0) * (t - t0) / (t1 - t0)
            if t1 - t0 != 0:
                slope = (v1 - v0) / (t1 - t0)
                expr = v0 + slope * (time_symbol - t0)
                condition = sp.And(time_symbol >= t0, time_symbol < t1)
                pieces.append((expr, condition))
        
        # Add the last piece for t >= last_keyframe_time
        last_value = sorted_keyframes[-1]['value']
        last_time = sorted_keyframes[-1]['time']
        pieces.append((last_value, time_symbol >= last_time))
        
        # Create piecewise function
        piecewise_expr = sp.Piecewise(*pieces)
        
        return SymbolicExpression(sympy_obj=piecewise_expr, 
                                 variables={'t': time_symbol})
    
    def get_shape_boundary_equations(self, time_value: Optional[float] = None,
                                   x_sym: Optional[sp.Symbol] = None, 
                                   y_sym: Optional[sp.Symbol] = None,
                                   z_sym: Optional[sp.Symbol] = None) -> List[SymbolicExpression]:
        """
        Get the symbolic equations that describe the shape's boundary.
        
        Args:
            time_value: Optional specific time value to evaluate animated parameters
            x_sym: Optional symbol for x coordinate (defaults to self.x_sym)
            y_sym: Optional symbol for y coordinate (defaults to self.y_sym)
            z_sym: Optional symbol for z coordinate (defaults to self.z_sym)
            
        Returns:
            A list of SymbolicExpression objects representing the boundary equations
            
        Example:
            >>> analyzer = ShapeEquationMathAnalyzer(circle_component)
            >>> boundary_eqs = analyzer.get_shape_boundary_equations()
            >>> if boundary_eqs:
            >>>     print(boundary_eqs[0].to_string())
            Eq((x - cx)**2 + (y - cy)**2, radius**2)  # Circle equation
        """
        if x_sym is None:
            x_sym = self.x_sym
        if y_sym is None:
            y_sym = self.y_sym
        if z_sym is None:
            z_sym = self.z_sym
        
        # Dictionary to store symbols
        symbols = {'x': x_sym, 'y': y_sym, 'z': z_sym, 't': self.t_sym}
        
        # Get shape type
        shape_type = self.shape_type
        
        # Handle different shape types
        if shape_type == 'circle':
            return self._get_circle_boundary_equations(time_value, symbols)
        elif shape_type == 'rectangle':
            return self._get_rectangle_boundary_equations(time_value, symbols)
        elif shape_type == 'ellipse':
            return self._get_ellipse_boundary_equations(time_value, symbols)
        elif shape_type == 'line':
            return self._get_line_boundary_equations(time_value, symbols)
        elif shape_type == 'polygon':
            return self._get_polygon_boundary_equations(time_value, symbols)
        elif shape_type == 'path':
            return self._get_path_boundary_equations(time_value, symbols)
        else:
            # Unknown or unsupported shape type
            return []
    
    def _get_circle_boundary_equations(self, time_value: Optional[float], 
                                     symbols: Dict[str, sp.Symbol]) -> List[SymbolicExpression]:
        """
        Get boundary equations for a circle.
        
        Args:
            time_value: Optional time value for evaluation
            symbols: Dictionary of symbols
            
        Returns:
            List of boundary equations
        """
        # Get parameters
        cx_expr = self.get_parameter_as_symbolic_expression('cx')
        cy_expr = self.get_parameter_as_symbolic_expression('cy')
        radius_expr = self.get_parameter_as_symbolic_expression('radius')
        
        if not (cx_expr and cy_expr and radius_expr):
            return []
        
        # If time_value is provided, evaluate parameters at that time
        if time_value is not None:
            cx = cx_expr.evaluate({'t': time_value})
            cy = cy_expr.evaluate({'t': time_value})
            radius = radius_expr.evaluate({'t': time_value})
            
            # Create equation with numeric values
            equation = sp.Eq((symbols['x'] - cx)**2 + (symbols['y'] - cy)**2, radius**2)
        else:
            # Create equation with symbolic expressions
            equation = sp.Eq(
                (symbols['x'] - cx_expr.sympy_expr)**2 + 
                (symbols['y'] - cy_expr.sympy_expr)**2, 
                radius_expr.sympy_expr**2
            )
        
        return [SymbolicExpression(sympy_obj=equation, variables={
            'x': symbols['x'], 
            'y': symbols['y'], 
            't': symbols['t']
        })]
    
    def _get_rectangle_boundary_equations(self, time_value: Optional[float], 
                                        symbols: Dict[str, sp.Symbol]) -> List[SymbolicExpression]:
        """
        Get boundary equations for a rectangle.
        
        Args:
            time_value: Optional time value for evaluation
            symbols: Dictionary of symbols
            
        Returns:
            List of boundary equations
        """
        # Get parameters
        x_expr = self.get_parameter_as_symbolic_expression('x')
        y_expr = self.get_parameter_as_symbolic_expression('y')
        width_expr = self.get_parameter_as_symbolic_expression('width')
        height_expr = self.get_parameter_as_symbolic_expression('height')
        
        if not (x_expr and y_expr and width_expr and height_expr):
            return []
        
        # If time_value is provided, evaluate parameters at that time
        if time_value is not None:
            x = x_expr.evaluate({'t': time_value})
            y = y_expr.evaluate({'t': time_value})
            width = width_expr.evaluate({'t': time_value})
            height = height_expr.evaluate({'t': time_value})
            
            # Create equations for the four sides
            left = sp.Eq(symbols['x'], x)
            right = sp.Eq(symbols['x'], x + width)
            top = sp.Eq(symbols['y'], y)
            bottom = sp.Eq(symbols['y'], y + height)
        else:
            # Create equations with symbolic expressions
            left = sp.Eq(symbols['x'], x_expr.sympy_expr)
            right = sp.Eq(symbols['x'], x_expr.sympy_expr + width_expr.sympy_expr)
            top = sp.Eq(symbols['y'], y_expr.sympy_expr)
            bottom = sp.Eq(symbols['y'], y_expr.sympy_expr + height_expr.sympy_expr)
        
        # Create SymbolicExpression objects for each equation
        equations = []
        for eq in [left, right, top, bottom]:
            equations.append(SymbolicExpression(sympy_obj=eq, variables={
                'x': symbols['x'], 
                'y': symbols['y'], 
                't': symbols['t']
            }))
        
        return equations
    
    def _get_ellipse_boundary_equations(self, time_value: Optional[float], 
                                      symbols: Dict[str, sp.Symbol]) -> List[SymbolicExpression]:
        """
        Get boundary equations for an ellipse.
        
        Args:
            time_value: Optional time value for evaluation
            symbols: Dictionary of symbols
            
        Returns:
            List of boundary equations
        """
        # Get parameters
        cx_expr = self.get_parameter_as_symbolic_expression('cx')
        cy_expr = self.get_parameter_as_symbolic_expression('cy')
        rx_expr = self.get_parameter_as_symbolic_expression('rx')
        ry_expr = self.get_parameter_as_symbolic_expression('ry')
        
        if not (cx_expr and cy_expr and rx_expr and ry_expr):
            return []
        
        # If time_value is provided, evaluate parameters at that time
        if time_value is not None:
            cx = cx_expr.evaluate({'t': time_value})
            cy = cy_expr.evaluate({'t': time_value})
            rx = rx_expr.evaluate({'t': time_value})
            ry = ry_expr.evaluate({'t': time_value})
            
            # Create ellipse equation: (x-cx)²/rx² + (y-cy)²/ry² = 1
            equation = sp.Eq(
                (symbols['x'] - cx)**2 / rx**2 + 
                (symbols['y'] - cy)**2 / ry**2, 
                1
            )
        else:
            # Create equation with symbolic expressions
            equation = sp.Eq(
                (symbols['x'] - cx_expr.sympy_expr)**2 / rx_expr.sympy_expr**2 + 
                (symbols['y'] - cy_expr.sympy_expr)**2 / ry_expr.sympy_expr**2, 
                1
            )
        
        return [SymbolicExpression(sympy_obj=equation, variables={
            'x': symbols['x'], 
            'y': symbols['y'], 
            't': symbols['t']
        })]
    
    def _get_line_boundary_equations(self, time_value: Optional[float], 
                                   symbols: Dict[str, sp.Symbol]) -> List[SymbolicExpression]:
        """
        Get boundary equations for a line.
        
        Args:
            time_value: Optional time value for evaluation
            symbols: Dictionary of symbols
            
        Returns:
            List of boundary equations
        """
        # Get parameters
        x1_expr = self.get_parameter_as_symbolic_expression('x1')
        y1_expr = self.get_parameter_as_symbolic_expression('y1')
        x2_expr = self.get_parameter_as_symbolic_expression('x2')
        y2_expr = self.get_parameter_as_symbolic_expression('y2')
        
        if not (x1_expr and y1_expr and x2_expr and y2_expr):
            return []
        
        # If time_value is provided, evaluate parameters at that time
        if time_value is not None:
            x1 = x1_expr.evaluate({'t': time_value})
            y1 = y1_expr.evaluate({'t': time_value})
            x2 = x2_expr.evaluate({'t': time_value})
            y2 = y2_expr.evaluate({'t': time_value})
            
            # Create parametric equations for the line
            # x = x1 + u*(x2-x1), y = y1 + u*(y2-y1) where u is in [0,1]
            u_sym = sp.Symbol('u')
            x_eq = sp.Eq(symbols['x'], x1 + u_sym * (x2 - x1))
            y_eq = sp.Eq(symbols['y'], y1 + u_sym * (y2 - y1))
        else:
            # Create equations with symbolic expressions
            u_sym = sp.Symbol('u')
            x_eq = sp.Eq(
                symbols['x'], 
                x1_expr.sympy_expr + u_sym * (x2_expr.sympy_expr - x1_expr.sympy_expr)
            )
            y_eq = sp.Eq(
                symbols['y'], 
                y1_expr.sympy_expr + u_sym * (y2_expr.sympy_expr - y1_expr.sympy_expr)
            )
        
        # Create SymbolicExpression objects for each equation
        equations = []
        for eq in [x_eq, y_eq]:
            equations.append(SymbolicExpression(sympy_obj=eq, variables={
                'x': symbols['x'], 
                'y': symbols['y'], 
                't': symbols['t'],
                'u': u_sym
            }))
        
        return equations
    
    def _get_polygon_boundary_equations(self, time_value: Optional[float], 
                                      symbols: Dict[str, sp.Symbol]) -> List[SymbolicExpression]:
        """
        Get boundary equations for a polygon.
        
        Args:
            time_value: Optional time value for evaluation
            symbols: Dictionary of symbols
            
        Returns:
            List of boundary equations (one for each edge)
        """
        # For polygons, we need to get the points parameter
        params = self.component.get('params', {})
        if not isinstance(params, dict) or 'points' not in params:
            return []
        
        points = params['points']
        if not points or len(points) < 3:
            return []
        
        # Create equations for each edge
        equations = []
        u_sym = sp.Symbol('u')
        
        for i in range(len(points)):
            # Get current and next point
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            
            # Create parametric equations for the edge
            # x = x1 + u*(x2-x1), y = y1 + u*(y2-y1) where u is in [0,1]
            x1, y1 = p1
            x2, y2 = p2
            
            x_eq = sp.Eq(symbols['x'], x1 + u_sym * (x2 - x1))
            y_eq = sp.Eq(symbols['y'], y1 + u_sym * (y2 - y1))
            
            # Add constraint that u is between 0 and 1
            u_constraint = sp.And(u_sym >= 0, u_sym <= 1)
            
            # Create SymbolicExpression objects for each equation
            for eq in [x_eq, y_eq]:
                equations.append(SymbolicExpression(sympy_obj=eq, variables={
                    'x': symbols['x'], 
                    'y': symbols['y'], 
                    'u': u_sym
                }))
            
            # Add the constraint as a separate equation
            equations.append(SymbolicExpression(sympy_obj=u_constraint, variables={
                'u': u_sym
            }))
        
        return equations
    
    def _get_path_boundary_equations(self, time_value: Optional[float], 
                                   symbols: Dict[str, sp.Symbol]) -> List[SymbolicExpression]:
        """
        Get boundary equations for a path.
        
        Args:
            time_value: Optional time value for evaluation
            symbols: Dictionary of symbols
            
        Returns:
            List of boundary equations
        """
        # For paths, we need to get the commands parameter
        params = self.component.get('params', {})
        if not isinstance(params, dict) or 'commands' not in params:
            return []
        
        commands = params['commands']
        if not commands:
            return []
        
        # Create equations for each segment in the path
        equations = []
        u_sym = sp.Symbol('u')
        
        # Keep track of the current point
        current_x, current_y = 0, 0
        
        for cmd in commands:
            command = cmd.get('command', '')
            points = cmd.get('points', [])
            
            if command == 'M':  # Move to
                if len(points) >= 2:
                    current_x, current_y = points[0], points[1]
            
            elif command == 'L':  # Line to
                if len(points) >= 2:
                    x2, y2 = points[0], points[1]
                    
                    # Create parametric equations for the line
                    x_eq = sp.Eq(symbols['x'], current_x + u_sym * (x2 - current_x))
                    y_eq = sp.Eq(symbols['y'], current_y + u_sym * (y2 - current_y))
                    
                    # Add constraint that u is between 0 and 1
                    u_constraint = sp.And(u_sym >= 0, u_sym <= 1)
                    
                    # Create SymbolicExpression objects for each equation
                    for eq in [x_eq, y_eq]:
                        equations.append(SymbolicExpression(sympy_obj=eq, variables={
                            'x': symbols['x'], 
                            'y': symbols['y'], 
                            'u': u_sym
                        }))
                    
                    # Add the constraint as a separate equation
                    equations.append(SymbolicExpression(sympy_obj=u_constraint, variables={
                        'u': u_sym
                    }))
                    
                    # Update current point
                    current_x, current_y = x2, y2
            
            elif command == 'C':  # Cubic Bezier curve
                if len(points) >= 6:
                    # Control points and end point
                    c1x, c1y = points[0], points[1]
                    c2x, c2y = points[2], points[3]
                    x2, y2 = points[4], points[5]
                    
                    # Create parametric equations for the Bezier curve
                    # B(u) = (1-u)³P₀ + 3(1-u)²uP₁ + 3(1-u)u²P₂ + u³P₃
                    x_eq = sp.Eq(
                        symbols['x'], 
                        (1-u_sym)**3 * current_x + 
                        3*(1-u_sym)**2 * u_sym * c1x + 
                        3*(1-u_sym) * u_sym**2 * c2x + 
                        u_sym**3 * x2
                    )
                    
                    y_eq = sp.Eq(
                        symbols['y'], 
                        (1-u_sym)**3 * current_y + 
                        3*(1-u_sym)**2 * u_sym * c1y + 
                        3*(1-u_sym) * u_sym**2 * c2y + 
                        u_sym**3 * y2
                    )
                    
                    # Add constraint that u is between 0 and 1
                    u_constraint = sp.And(u_sym >= 0, u_sym <= 1)
                    
                    # Create SymbolicExpression objects for each equation
                    for eq in [x_eq, y_eq]:
                        equations.append(SymbolicExpression(sympy_obj=eq, variables={
                            'x': symbols['x'], 
                            'y': symbols['y'], 
                            'u': u_sym
                        }))
                    
                    # Add the constraint as a separate equation
                    equations.append(SymbolicExpression(sympy_obj=u_constraint, variables={
                        'u': u_sym
                    }))
                    
                    # Update current point
                    current_x, current_y = x2, y2
            
            elif command == 'Q':  # Quadratic Bezier curve
                if len(points) >= 4:
                    # Control point and end point
                    cx, cy = points[0], points[1]
                    x2, y2 = points[2], points[3]
                    
                    # Create parametric equations for the Bezier curve
                    # B(u) = (1-u)²P₀ + 2(1-u)uP₁ + u²P₂
                    x_eq = sp.Eq(
                        symbols['x'], 
                        (1-u_sym)**2 * current_x + 
                        2*(1-u_sym) * u_sym * cx + 
                        u_sym**2 * x2
                    )
                    
                    y_eq = sp.Eq(
                        symbols['y'], 
                        (1-u_sym)**2 * current_y + 
                        2*(1-u_sym) * u_sym * cy + 
                        u_sym**2 * y2
                    )
                    
                    # Add constraint that u is between 0 and 1
                    u_constraint = sp.And(u_sym >= 0, u_sym <= 1)
                    
                    # Create SymbolicExpression objects for each equation
                    for eq in [x_eq, y_eq]:
                        equations.append(SymbolicExpression(sympy_obj=eq, variables={
                            'x': symbols['x'], 
                            'y': symbols['y'], 
                            'u': u_sym
                        }))
                    
                    # Add the constraint as a separate equation
                    equations.append(SymbolicExpression(sympy_obj=u_constraint, variables={
                        'u': u_sym
                    }))
                    
                    # Update current point
                    current_x, current_y = x2, y2
            
            elif command == 'Z':  # Close path
                # We don't need to add any equations for this command
                pass
        
        return equations
    
    def analyze_geometric_properties(self, time_value: float) -> Dict[str, Any]:
        """
        Analyze geometric properties of the shape at a specific time.
        
        Args:
            time_value: Time value to evaluate animated parameters
            
        Returns:
            Dictionary of geometric properties (area, perimeter, center, etc.)
            
        Example:
            >>> analyzer = ShapeEquationMathAnalyzer(circle_component)
            >>> properties = analyzer.analyze_geometric_properties(0.5)
            >>> properties
            {'area': 78.54, 'perimeter': 31.42, 'center': (100, 150)}
        """
        shape_type = self.shape_type
        properties = {}
        
        if shape_type == 'circle':
            # Get parameters
            cx_expr = self.get_parameter_as_symbolic_expression('cx')
            cy_expr = self.get_parameter_as_symbolic_expression('cy')
            radius_expr = self.get_parameter_as_symbolic_expression('radius')
            
            if cx_expr and cy_expr and radius_expr:
                cx = cx_expr.evaluate({'t': time_value})
                cy = cy_expr.evaluate({'t': time_value})
                radius = radius_expr.evaluate({'t': time_value})
                
                # Calculate properties
                properties['area'] = math.pi * radius**2
                properties['perimeter'] = 2 * math.pi * radius
                properties['center'] = (cx, cy)
        
        elif shape_type == 'rectangle':
            # Get parameters
            x_expr = self.get_parameter_as_symbolic_expression('x')
            y_expr = self.get_parameter_as_symbolic_expression('y')
            width_expr = self.get_parameter_as_symbolic_expression('width')
            height_expr = self.get_parameter_as_symbolic_expression('height')
            
            if x_expr and y_expr and width_expr and height_expr:
                x = x_expr.evaluate({'t': time_value})
                y = y_expr.evaluate({'t': time_value})
                width = width_expr.evaluate({'t': time_value})
                height = height_expr.evaluate({'t': time_value})
                
                # Calculate properties
                properties['area'] = width * height
                properties['perimeter'] = 2 * (width + height)
                properties['center'] = (x + width/2, y + height/2)
        
        elif shape_type == 'ellipse':
            # Get parameters
            cx_expr = self.get_parameter_as_symbolic_expression('cx')
            cy_expr = self.get_parameter_as_symbolic_expression('cy')
            rx_expr = self.get_parameter_as_symbolic_expression('rx')
            ry_expr = self.get_parameter_as_symbolic_expression('ry')
            
            if cx_expr and cy_expr and rx_expr and ry_expr:
                cx = cx_expr.evaluate({'t': time_value})
                cy = cy_expr.evaluate({'t': time_value})
                rx = rx_expr.evaluate({'t': time_value})
                ry = ry_expr.evaluate({'t': time_value})
                
                # Calculate properties
                properties['area'] = math.pi * rx * ry
                # Approximation of ellipse perimeter
                h = ((rx - ry)**2) / ((rx + ry)**2)
                properties['perimeter'] = math.pi * (rx + ry) * (1 + (3*h) / (10 + math.sqrt(4 - 3*h)))
                properties['center'] = (cx, cy)
        
        elif shape_type == 'line':
            # Get parameters
            x1_expr = self.get_parameter_as_symbolic_expression('x1')
            y1_expr = self.get_parameter_as_symbolic_expression('y1')
            x2_expr = self.get_parameter_as_symbolic_expression('x2')
            y2_expr = self.get_parameter_as_symbolic_expression('y2')
            
            if x1_expr and y1_expr and x2_expr and y2_expr:
                x1 = x1_expr.evaluate({'t': time_value})
                y1 = y1_expr.evaluate({'t': time_value})
                x2 = x2_expr.evaluate({'t': time_value})
                y2 = y2_expr.evaluate({'t': time_value})
                
                # Calculate properties
                properties['length'] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                properties['midpoint'] = ((x1 + x2)/2, (y1 + y2)/2)
        
        # Add more shape types as needed
        
        return properties


# Utility functions

def parse_text_to_symbolic(text_equation: str, 
                          known_symbols: Optional[Dict[str, sp.Symbol]] = None) -> Optional[SymbolicExpression]:
    """
    Parse a text equation into a SymbolicExpression.
    
    Args:
        text_equation: String representation of the equation
        known_symbols: Optional dictionary of known symbols
        
    Returns:
        A SymbolicExpression object, or None if parsing fails
        
    Example:
        >>> expr = parse_text_to_symbolic("x**2 + 3*y")
        >>> print(expr.to_string())
        x**2 + 3*y
    """
    try:
        return SymbolicExpression(expression_str=text_equation, variables=known_symbols)
    except ValueError as e:
        print(f"Error parsing equation: {str(e)}")
        return None


def are_expressions_equivalent(expr1: SymbolicExpression, 
                              expr2: SymbolicExpression) -> bool:
    """
    Check if two symbolic expressions are mathematically equivalent.
    
    Args:
        expr1: First expression
        expr2: Second expression
        
    Returns:
        True if the expressions are equivalent, False otherwise
        
    Example:
        >>> expr1 = SymbolicExpression(expression_str="x**2 + 2*x + 1")
        >>> expr2 = SymbolicExpression(expression_str="(x + 1)**2")
        >>> are_expressions_equivalent(expr1, expr2)
        True
    """
    try:
        # Combine variables from both expressions
        variables = expr1.variables.copy()
        variables.update(expr2.variables)
        
        # Check if the difference simplifies to zero
        diff = expr1.sympy_expr - expr2.sympy_expr
        simplified_diff = sp.simplify(diff)
        
        return simplified_diff == 0
    except Exception as e:
        print(f"Error comparing expressions: {str(e)}")
        return False


def substitute_values(expression: SymbolicExpression, 
                     assignments: Dict[str, float]) -> SymbolicExpression:
    """
    Substitute values into a symbolic expression.
    
    Args:
        expression: The symbolic expression
        assignments: Dictionary mapping variable names to values
        
    Returns:
        A new SymbolicExpression with the substitutions applied
        
    Example:
        >>> expr = SymbolicExpression(expression_str="x**2 + y")
        >>> new_expr = substitute_values(expr, {"x": 3})
        >>> print(new_expr.to_string())
        9 + y
    """
    try:
        # Create substitution dictionary using the stored symbols
        subs_dict = {}
        for var_name, value in assignments.items():
            if var_name in expression.variables:
                subs_dict[expression.variables[var_name]] = value
        
        # Apply substitution
        new_expr = expression.sympy_expr.subs(subs_dict)
        
        # Create a new SymbolicExpression with the result
        # Filter out substituted variables from the variables dictionary
        new_variables = {name: symbol for name, symbol in expression.variables.items() 
                        if name not in assignments}
        
        return SymbolicExpression(sympy_obj=new_expr, variables=new_variables)
    except Exception as e:
        print(f"Error substituting values: {str(e)}")
        return expression  # Return the original expression on error


def create_symbolic_function(expression: SymbolicExpression, 
                           var_names: List[str]) -> Callable:
    """
    Create a callable function from a symbolic expression.
    
    Args:
        expression: The symbolic expression
        var_names: List of variable names to use as function arguments
        
    Returns:
        A callable function that evaluates the expression
        
    Example:
        >>> expr = SymbolicExpression(expression_str="x**2 + y")
        >>> func = create_symbolic_function(expr, ["x", "y"])
        >>> func(3, 4)
        13.0
    """
    # Create a lambda function that evaluates the expression
    def symbolic_function(*args):
        if len(args) != len(var_names):
            raise ValueError(f"Expected {len(var_names)} arguments, got {len(args)}")
        
        # Create assignments dictionary
        assignments = {var_names[i]: args[i] for i in range(len(args))}
        
        # Evaluate the expression
        return expression.evaluate(assignments)
    
    return symbolic_function


# Example usage
if __name__ == "__main__":
    # Create a symbolic expression
    expr = SymbolicExpression(expression_str="x**2 + 2*x*y + y**2")
    print(f"Expression: {expr.to_string()}")
    print(f"LaTeX: {expr.to_latex()}")
    
    # Evaluate the expression
    result = expr.evaluate({"x": 2, "y": 3})
    print(f"Evaluated at x=2, y=3: {result}")
    
    # Differentiate with respect to x
    derivative = expr.differentiate("x")
    print(f"Derivative with respect to x: {derivative.to_string()}")
    
    # Integrate with respect to x
    integral = expr.integrate("x")
    print(f"Integral with respect to x: {integral.to_string()}")
    
    # Simplify an expression
    complex_expr = SymbolicExpression(expression_str="(x + y)**2 - x**2 - 2*x*y")
    simplified = complex_expr.simplify()
    print(f"Original: {complex_expr.to_string()}")
    print(f"Simplified: {simplified.to_string()}")
    
    # Check if expressions are equivalent
    expr1 = SymbolicExpression(expression_str="(x + 1)**2")
    expr2 = SymbolicExpression(expression_str="x**2 + 2*x + 1")
    print(f"Are equivalent: {are_expressions_equivalent(expr1, expr2)}")
    
    # Create a callable function
    func = create_symbolic_function(expr, ["x", "y"])
    print(f"Function evaluated at x=1, y=2: {func(1, 2)}")
    
    # Example shape component (circle)
    circle_component = {
        "params": {
            "shape_type": "circle",
            "cx": 100,
            "cy": 150,
            "radius": 50
        },
        "animations": {
            "radius": [
                {"time": 0, "value": 50},
                {"time": 1, "value": 100}
            ]
        }
    }
    
    # Analyze the shape
    analyzer = ShapeEquationMathAnalyzer(circle_component)
    
    # Get radius as a symbolic expression
    radius_expr = analyzer.get_parameter_as_symbolic_expression("radius")
    print(f"Radius expression: {radius_expr.to_string()}")
    
    # Get boundary equations
    boundary_eqs = analyzer.get_shape_boundary_equations()
    if boundary_eqs:
        print(f"Boundary equation: {boundary_eqs[0].to_string()}")
    
    # Analyze geometric properties
    properties = analyzer.analyze_geometric_properties(0.5)
    print(f"Geometric properties at t=0.5: {properties}")
