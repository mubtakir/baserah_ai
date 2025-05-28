#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Foundation Module for Basira System

This module provides the core mathematical tools and functions for handling
the mathematical aspects of equations used in the Basira System. It enables
symbolic analysis, simplification, evaluation, and algebraic, differential,
and integral operations on equations.

The module works closely with shape equations parsed by AdvancedShapeEquationParser
to extract mathematical information, as well as with other mathematical/behavioral
equations that may be used by modules like EquationEvolutionService_EEE or
LogicalReasoningEngine.

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
    A wrapper around SymPy expressions to facilitate their handling and
    to store additional information about them.
    
    This class provides methods for evaluating, simplifying, differentiating,
    and integrating symbolic expressions, as well as for converting them to
    string or LaTeX representations.
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
        if expression_str is None and sympy_obj is None:
            raise ValueError("Either expression_str or sympy_obj must be provided")
        
        # Initialize variables dictionary
        self.variables = variables or {}
        
        # Parse expression string if provided
        if expression_str is not None:
            try:
                # Create symbols for any variables in the expression that aren't in variables
                local_dict = {name: symbol for name, symbol in self.variables.items()}
                
                # Parse the expression
                self.sympy_expr = sp.sympify(expression_str, locals=local_dict)
                
                # Update variables with any new symbols created during parsing
                for symbol in self.sympy_expr.free_symbols:
                    if symbol.name not in self.variables:
                        self.variables[symbol.name] = symbol
            except Exception as e:
                raise ValueError(f"Failed to parse expression: {expression_str}. Error: {str(e)}")
        else:
            # Use provided SymPy object
            self.sympy_expr = sympy_obj
            
            # Update variables with symbols from the expression
            for symbol in self.sympy_expr.free_symbols:
                if symbol.name not in self.variables:
                    self.variables[symbol.name] = symbol
    
    def evaluate(self, assignments: Dict[str, float]) -> Optional[float]:
        """
        Evaluate the expression with the given variable assignments.
        
        Args:
            assignments: Dictionary mapping variable names to values
            
        Returns:
            The numerical value of the expression, or None if evaluation fails
            
        Example:
            >>> expr = SymbolicExpression("x**2 + y")
            >>> expr.evaluate({"x": 2, "y": 3})
            7.0
        """
        try:
            # Create substitution dictionary
            subs_dict = {}
            for var_name, value in assignments.items():
                if var_name in self.variables:
                    subs_dict[self.variables[var_name]] = value
            
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
            >>> expr = SymbolicExpression("x**2 + 2*x**2")
            >>> simplified = expr.simplify()
            >>> print(simplified.to_string())
            3*x**2
        """
        simplified_expr = sp.simplify(self.sympy_expr)
        return SymbolicExpression(sympy_obj=simplified_expr, variables=self.variables.copy())
    
    def differentiate(self, var_name: str) -> 'SymbolicExpression':
        """
        Compute the derivative of the expression with respect to the given variable.
        
        Args:
            var_name: Name of the variable to differentiate with respect to
            
        Returns:
            A new SymbolicExpression representing the derivative
            
        Raises:
            ValueError: If the variable is not found in the expression
            
        Example:
            >>> expr = SymbolicExpression("x**2 + y*x")
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
        Compute the integral of the expression with respect to the given variable.
        
        Args:
            var_name: Name of the variable to integrate with respect to
            limits: Optional tuple (lower, upper) for definite integration
            
        Returns:
            A new SymbolicExpression representing the integral
            
        Raises:
            ValueError: If the variable is not found in the expression
            
        Example:
            >>> expr = SymbolicExpression("2*x")
            >>> indefinite = expr.integrate("x")
            >>> print(indefinite.to_string())
            x**2
            >>> definite = expr.integrate("x", (0, 1))
            >>> print(definite.to_string())
            1
        """
        if var_name not in self.variables:
            raise ValueError(f"Variable {var_name} not found in expression")
        
        var_symbol = self.variables[var_name]
        
        if limits is None:
            # Indefinite integration
            integral = sp.integrate(self.sympy_expr, var_symbol)
        else:
            # Definite integration
            lower, upper = limits
            # Convert limits to SymPy expressions if they're strings
            if isinstance(lower, str):
                lower = sp.sympify(lower, locals=self.variables)
            if isinstance(upper, str):
                upper = sp.sympify(upper, locals=self.variables)
            
            integral = sp.integrate(self.sympy_expr, (var_symbol, lower, upper))
        
        return SymbolicExpression(sympy_obj=integral, variables=self.variables.copy())
    
    def solve_for(self, var_name: str) -> List['SymbolicExpression']:
        """
        Solve the equation for the specified variable.
        
        This method assumes the expression is an equation (Eq) or can be set equal to zero.
        
        Args:
            var_name: Name of the variable to solve for
            
        Returns:
            A list of SymbolicExpression objects representing the solutions
            
        Raises:
            ValueError: If the variable is not found in the expression
            
        Example:
            >>> eq = SymbolicExpression("Eq(x**2 - 4, 0)")
            >>> solutions = eq.solve_for("x")
            >>> [sol.to_string() for sol in solutions]
            ['-2', '2']
        """
        if var_name not in self.variables:
            raise ValueError(f"Variable {var_name} not found in expression")
        
        var_symbol = self.variables[var_name]
        
        # Check if the expression is an equation
        if isinstance(self.sympy_expr, sp.Eq):
            equation = self.sympy_expr
        else:
            # If not an equation, set it equal to zero
            equation = sp.Eq(self.sympy_expr, 0)
        
        # Solve the equation
        solutions = sp.solve(equation, var_symbol)
        
        # Convert solutions to SymbolicExpression objects
        result = []
        for sol in solutions:
            result.append(SymbolicExpression(sympy_obj=sol, variables=self.variables.copy()))
        
        return result
    
    def get_free_symbols(self) -> Set[sp.Symbol]:
        """
        Get the set of free symbols (variables) in the expression.
        
        Returns:
            A set of SymPy symbols
            
        Example:
            >>> expr = SymbolicExpression("x**2 + y*z")
            >>> {symbol.name for symbol in expr.get_free_symbols()}
            {'x', 'y', 'z'}
        """
        return self.sympy_expr.free_symbols
    
    def to_string(self) -> str:
        """
        Get the string representation of the expression.
        
        Returns:
            String representation of the SymPy expression
            
        Example:
            >>> expr = SymbolicExpression("x**2 + y")
            >>> expr.to_string()
            'x**2 + y'
        """
        return str(self.sympy_expr)
    
    def to_latex(self) -> str:
        """
        Get the LaTeX representation of the expression.
        
        Returns:
            LaTeX representation of the SymPy expression
            
        Example:
            >>> expr = SymbolicExpression("x**2 + sqrt(y)")
            >>> expr.to_latex()
            'x^{2} + \\sqrt{y}'
        """
        return sp.latex(self.sympy_expr)
    
    def get_complexity_score(self) -> float:
        """
        Calculate a complexity score for the expression.
        
        The score is based on the number of operations, functions, and depth of the expression.
        
        Returns:
            A float representing the complexity (higher is more complex)
            
        Example:
            >>> expr1 = SymbolicExpression("x + y")
            >>> expr2 = SymbolicExpression("sin(x)**2 + exp(y*z)")
            >>> expr1.get_complexity_score() < expr2.get_complexity_score()
            True
        """
        # Count the number of operations
        count_ops = sp.count_ops(self.sympy_expr)
        
        # Count the number of function calls
        count_funcs = sum(1 for atom in self.sympy_expr.atoms(sp.Function))
        
        # Estimate the depth of the expression tree
        def get_depth(expr):
            if expr.is_Atom:
                return 0
            else:
                return 1 + max(get_depth(arg) for arg in expr.args)
        
        depth = get_depth(self.sympy_expr)
        
        # Combine the metrics
        complexity = count_ops + 2 * count_funcs + depth
        
        return float(complexity)
    
    def __str__(self) -> str:
        """String representation of the expression."""
        return self.to_string()
    
    def __repr__(self) -> str:
        """Detailed string representation of the expression."""
        return f"SymbolicExpression({self.to_string()})"


class ShapeEquationMathAnalyzer:
    """
    Analyzer for the mathematical aspects of shape equations.
    
    This class provides methods for extracting mathematical information from
    shape components parsed by AdvancedShapeEquationParser, such as parameter
    expressions, boundary equations, and geometric properties.
    """
    
    # Define parameter mappings for different shape types
    SHAPE_PARAM_MAPS = {
        'circle': ['cx', 'cy', 'radius'],
        'rectangle': ['x', 'y', 'width', 'height'],
        'ellipse': ['cx', 'cy', 'rx', 'ry'],
        'line': ['x1', 'y1', 'x2', 'y2'],
        'path': ['commands'],
        'polygon': ['points'],
        'text': ['x', 'y', 'text']
    }
    
    def __init__(self, parsed_shape_component: Dict[str, Any]):
        """
        Initialize a ShapeEquationMathAnalyzer.
        
        Args:
            parsed_shape_component: A dictionary representing a parsed shape component
            
        Raises:
            ValueError: If the shape component is invalid or missing required fields
        """
        self.component = parsed_shape_component
        
        # Validate the component
        if 'params' not in self.component:
            raise ValueError("Shape component must have 'params' field")
        
        # Extract shape type
        self.shape_type = self.component['params'].get('shape_type')
        if not self.shape_type:
            raise ValueError("Shape component must have 'shape_type' in params")
        
        # Define basic symbols
        self.t_sym = sp.Symbol('t')  # Time symbol for animation
        self.x_sym = sp.Symbol('x')  # X coordinate
        self.y_sym = sp.Symbol('y')  # Y coordinate
        self.z_sym = sp.Symbol('z')  # Z coordinate (for 3D shapes)
        
        # Create a dictionary of basic symbols
        self.basic_symbols = {
            't': self.t_sym,
            'x': self.x_sym,
            'y': self.y_sym,
            'z': self.z_sym
        }
    
    def get_parameter_as_symbolic_expression(self, param_name: str, 
                                           time_symbol: Optional[sp.Symbol] = None) -> Optional[SymbolicExpression]:
        """
        Get a symbolic expression for a parameter, which may be animated or static.
        
        Args:
            param_name: Name of the parameter
            time_symbol: Optional symbol to use for time (defaults to self.t_sym)
            
        Returns:
            A SymbolicExpression representing the parameter, or None if not found
            
        Example:
            >>> component = {'params': {'shape_type': 'circle', 'cx': 100, 'cy': 150, 'radius': 50},
            ...              'animations': {'radius': [{'time': 0, 'value': 50}, {'time': 1, 'value': 100}]}}
            >>> analyzer = ShapeEquationMathAnalyzer(component)
            >>> radius_expr = analyzer.get_parameter_as_symbolic_expression('radius')
            >>> radius_expr.to_string()
            '50 + 50*t'
        """
        time_symbol = time_symbol or self.t_sym
        
        # Check if the parameter is animated
        animations = self.component.get('animations', {})
        if param_name in animations:
            keyframes = animations[param_name]
            
            # If there are keyframes, create a symbolic expression for the interpolation
            if keyframes:
                return self._create_interpolation_expression(keyframes, time_symbol)
        
        # If not animated, check for static parameter
        params = self.component.get('params', {})
        if param_name in params:
            # Create a constant symbolic expression
            return SymbolicExpression(sympy_obj=sp.sympify(params[param_name]),
                                     variables={'t': time_symbol})
        
        # Check in style properties
        style = self.component.get('style', {})
        if param_name in style:
            # Handle numeric style properties
            value = style[param_name]
            if isinstance(value, (int, float)):
                return SymbolicExpression(sympy_obj=sp.sympify(value),
                                         variables={'t': time_symbol})
            elif isinstance(value, str) and value.startswith('#'):
                # Handle color values (as strings)
                # For now, just return None for colors
                return None
        
        # Parameter not found
        return None
    
    def _create_interpolation_expression(self, keyframes: List[Dict[str, Any]], 
                                       time_symbol: sp.Symbol) -> SymbolicExpression:
        """
        Create a symbolic expression for interpolating between keyframes.
        
        Args:
            keyframes: List of keyframe dictionaries with 'time' and 'value'
            time_symbol: Symbol to use for time
            
        Returns:
            A SymbolicExpression representing the interpolation
            
        Note:
            This implementation uses linear interpolation between keyframes.
            More complex interpolation methods could be added in the future.
        """
        # Sort keyframes by time
        sorted_keyframes = sorted(keyframes, key=lambda kf: kf['time'])
        
        # If there's only one keyframe, return a constant expression
        if len(sorted_keyframes) == 1:
            return SymbolicExpression(sympy_obj=sp.sympify(sorted_keyframes[0]['value']),
                                     variables={'t': time_symbol})
        
        # For multiple keyframes, create a piecewise function
        pieces = []
        
        for i in range(len(sorted_keyframes) - 1):
            t0 = sorted_keyframes[i]['time']
            t1 = sorted_keyframes[i + 1]['time']
            v0 = sorted_keyframes[i]['value']
            v1 = sorted_keyframes[i + 1]['value']
            
            # Skip if times are the same
            if t1 == t0:
                continue
            
            # Create linear interpolation expression for this segment
            slope = (v1 - v0) / (t1 - t0)
            expr = v0 + slope * (time_symbol - t0)
            
            # Add condition for this piece
            condition = sp.And(time_symbol >= t0, time_symbol <= t1)
            pieces.append((expr, condition))
        
        # If no valid pieces were created, return the first value
        if not pieces:
            return SymbolicExpression(sympy_obj=sp.sympify(sorted_keyframes[0]['value']),
                                     variables={'t': time_symbol})
        
        # Create piecewise function
        piecewise_expr = sp.Piecewise(*pieces)
        
        return SymbolicExpression(sympy_obj=piecewise_expr,
                                 variables={'t': time_symbol})
    
    def get_shape_boundary_equations(self, time_value: Optional[float] = None,
                                   x_sym: Optional[sp.Symbol] = None,
                                   y_sym: Optional[sp.Symbol] = None,
                                   z_sym: Optional[sp.Symbol] = None) -> List[SymbolicExpression]:
        """
        Get the symbolic equations that describe the boundary of the shape.
        
        Args:
            time_value: Optional time value to evaluate animated parameters
            x_sym: Optional symbol for x coordinate (defaults to self.x_sym)
            y_sym: Optional symbol for y coordinate (defaults to self.y_sym)
            z_sym: Optional symbol for z coordinate (defaults to self.z_sym)
            
        Returns:
            A list of SymbolicExpression objects representing the boundary equations
            
        Example:
            >>> component = {'params': {'shape_type': 'circle', 'cx': 100, 'cy': 150, 'radius': 50}}
            >>> analyzer = ShapeEquationMathAnalyzer(component)
            >>> equations = analyzer.get_shape_boundary_equations()
            >>> equations[0].to_string()
            'Eq((x - 100)**2 + (y - 150)**2, 2500)'
        """
        # Use default symbols if not provided
        x_sym = x_sym or self.x_sym
        y_sym = y_sym or self.y_sym
        z_sym = z_sym or self.z_sym
        
        # Create symbol dictionary
        symbols = {
            'x': x_sym,
            'y': y_sym,
            'z': z_sym,
            't': self.t_sym
        }
        
        # Get shape type
        shape_type = self.shape_type
        
        # Get parameters as symbolic expressions
        param_exprs = {}
        for param_name in self.SHAPE_PARAM_MAPS.get(shape_type, []):
            expr = self.get_parameter_as_symbolic_expression(param_name)
            if expr is not None:
                # If time_value is provided, evaluate the expression at that time
                if time_value is not None:
                    try:
                        value = expr.evaluate({'t': time_value})
                        param_exprs[param_name] = SymbolicExpression(sympy_obj=sp.sympify(value),
                                                                   variables=symbols)
                    except:
                        param_exprs[param_name] = expr
                else:
                    param_exprs[param_name] = expr
        
        # Generate boundary equations based on shape type
        if shape_type == 'circle':
            return self._get_circle_boundary_equations(param_exprs, symbols)
        elif shape_type == 'rectangle':
            return self._get_rectangle_boundary_equations(param_exprs, symbols)
        elif shape_type == 'ellipse':
            return self._get_ellipse_boundary_equations(param_exprs, symbols)
        elif shape_type == 'line':
            return self._get_line_boundary_equations(param_exprs, symbols)
        elif shape_type == 'polygon':
            return self._get_polygon_boundary_equations(param_exprs, symbols)
        elif shape_type == 'path':
            return self._get_path_boundary_equations(param_exprs, symbols)
        else:
            # Unknown shape type
            return []
    
    def _get_circle_boundary_equations(self, param_exprs: Dict[str, SymbolicExpression],
                                     symbols: Dict[str, sp.Symbol]) -> List[SymbolicExpression]:
        """
        Get boundary equations for a circle.
        
        Args:
            param_exprs: Dictionary of parameter expressions
            symbols: Dictionary of symbols
            
        Returns:
            List of boundary equations
        """
        # Check if we have all required parameters
        if 'cx' not in param_exprs or 'cy' not in param_exprs or 'radius' not in param_exprs:
            return []
        
        # Extract parameters
        cx_expr = param_exprs['cx'].sympy_expr
        cy_expr = param_exprs['cy'].sympy_expr
        radius_expr = param_exprs['radius'].sympy_expr
        
        # Create circle equation: (x - cx)^2 + (y - cy)^2 = radius^2
        x_sym = symbols['x']
        y_sym = symbols['y']
        
        circle_eq = sp.Eq((x_sym - cx_expr)**2 + (y_sym - cy_expr)**2, radius_expr**2)
        
        return [SymbolicExpression(sympy_obj=circle_eq, variables=symbols)]
    
    def _get_rectangle_boundary_equations(self, param_exprs: Dict[str, SymbolicExpression],
                                        symbols: Dict[str, sp.Symbol]) -> List[SymbolicExpression]:
        """
        Get boundary equations for a rectangle.
        
        Args:
            param_exprs: Dictionary of parameter expressions
            symbols: Dictionary of symbols
            
        Returns:
            List of boundary equations
        """
        # Check if we have all required parameters
        if 'x' not in param_exprs or 'y' not in param_exprs or \
           'width' not in param_exprs or 'height' not in param_exprs:
            return []
        
        # Extract parameters
        x_expr = param_exprs['x'].sympy_expr
        y_expr = param_exprs['y'].sympy_expr
        width_expr = param_exprs['width'].sympy_expr
        height_expr = param_exprs['height'].sympy_expr
        
        # Create rectangle equations (four lines)
        x_sym = symbols['x']
        y_sym = symbols['y']
        
        # Left edge: x = x_expr
        left_eq = sp.Eq(x_sym, x_expr)
        
        # Right edge: x = x_expr + width_expr
        right_eq = sp.Eq(x_sym, x_expr + width_expr)
        
        # Top edge: y = y_expr
        top_eq = sp.Eq(y_sym, y_expr)
        
        # Bottom edge: y = y_expr + height_expr
        bottom_eq = sp.Eq(y_sym, y_expr + height_expr)
        
        return [
            SymbolicExpression(sympy_obj=left_eq, variables=symbols),
            SymbolicExpression(sympy_obj=right_eq, variables=symbols),
            SymbolicExpression(sympy_obj=top_eq, variables=symbols),
            SymbolicExpression(sympy_obj=bottom_eq, variables=symbols)
        ]
    
    def _get_ellipse_boundary_equations(self, param_exprs: Dict[str, SymbolicExpression],
                                      symbols: Dict[str, sp.Symbol]) -> List[SymbolicExpression]:
        """
        Get boundary equations for an ellipse.
        
        Args:
            param_exprs: Dictionary of parameter expressions
            symbols: Dictionary of symbols
            
        Returns:
            List of boundary equations
        """
        # Check if we have all required parameters
        if 'cx' not in param_exprs or 'cy' not in param_exprs or \
           'rx' not in param_exprs or 'ry' not in param_exprs:
            return []
        
        # Extract parameters
        cx_expr = param_exprs['cx'].sympy_expr
        cy_expr = param_exprs['cy'].sympy_expr
        rx_expr = param_exprs['rx'].sympy_expr
        ry_expr = param_exprs['ry'].sympy_expr
        
        # Create ellipse equation: ((x - cx)/rx)^2 + ((y - cy)/ry)^2 = 1
        x_sym = symbols['x']
        y_sym = symbols['y']
        
        ellipse_eq = sp.Eq(((x_sym - cx_expr)/rx_expr)**2 + ((y_sym - cy_expr)/ry_expr)**2, 1)
        
        return [SymbolicExpression(sympy_obj=ellipse_eq, variables=symbols)]
    
    def _get_line_boundary_equations(self, param_exprs: Dict[str, SymbolicExpression],
                                   symbols: Dict[str, sp.Symbol]) -> List[SymbolicExpression]:
        """
        Get boundary equations for a line.
        
        Args:
            param_exprs: Dictionary of parameter expressions
            symbols: Dictionary of symbols
            
        Returns:
            List of boundary equations
        """
        # Check if we have all required parameters
        if 'x1' not in param_exprs or 'y1' not in param_exprs or \
           'x2' not in param_exprs or 'y2' not in param_exprs:
            return []
        
        # Extract parameters
        x1_expr = param_exprs['x1'].sympy_expr
        y1_expr = param_exprs['y1'].sympy_expr
        x2_expr = param_exprs['x2'].sympy_expr
        y2_expr = param_exprs['y2'].sympy_expr
        
        # Create line equation: (y - y1) = ((y2 - y1) / (x2 - x1)) * (x - x1)
        x_sym = symbols['x']
        y_sym = symbols['y']
        
        # Handle vertical lines (x1 = x2)
        if sp.simplify(x2_expr - x1_expr) == 0:
            line_eq = sp.Eq(x_sym, x1_expr)
        else:
            slope = (y2_expr - y1_expr) / (x2_expr - x1_expr)
            line_eq = sp.Eq(y_sym - y1_expr, slope * (x_sym - x1_expr))
        
        return [SymbolicExpression(sympy_obj=line_eq, variables=symbols)]
    
    def _get_polygon_boundary_equations(self, param_exprs: Dict[str, SymbolicExpression],
                                      symbols: Dict[str, sp.Symbol]) -> List[SymbolicExpression]:
        """
        Get boundary equations for a polygon.
        
        Args:
            param_exprs: Dictionary of parameter expressions
            symbols: Dictionary of symbols
            
        Returns:
            List of boundary equations
        """
        # For polygons, we need to extract the points from the component directly
        points = self.component['params'].get('points', [])
        if not points or len(points) < 3:
            return []
        
        # Create line equations for each edge
        equations = []
        x_sym = symbols['x']
        y_sym = symbols['y']
        
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            
            # Handle vertical lines
            if x2 == x1:
                line_eq = sp.Eq(x_sym, x1)
            else:
                slope = (y2 - y1) / (x2 - x1)
                line_eq = sp.Eq(y_sym - y1, slope * (x_sym - x1))
            
            equations.append(SymbolicExpression(sympy_obj=line_eq, variables=symbols))
        
        return equations
    
    def _get_path_boundary_equations(self, param_exprs: Dict[str, SymbolicExpression],
                                   symbols: Dict[str, sp.Symbol]) -> List[SymbolicExpression]:
        """
        Get boundary equations for a path.
        
        Args:
            param_exprs: Dictionary of parameter expressions
            symbols: Dictionary of symbols
            
        Returns:
            List of boundary equations
        """
        # For paths, we need to extract the commands from the component directly
        commands = self.component['params'].get('commands', [])
        if not commands:
            return []
        
        # Create equations for each segment
        equations = []
        x_sym = symbols['x']
        y_sym = symbols['y']
        
        current_point = None
        
        for cmd in commands:
            command_type = cmd.get('command')
            points = cmd.get('points', [])
            
            if command_type == 'M':  # Move to
                if len(points) >= 2:
                    current_point = (points[0], points[1])
            
            elif command_type == 'L':  # Line to
                if current_point and len(points) >= 2:
                    x1, y1 = current_point
                    x2, y2 = points[0], points[1]
                    
                    # Handle vertical lines
                    if x2 == x1:
                        line_eq = sp.Eq(x_sym, x1)
                    else:
                        slope = (y2 - y1) / (x2 - x1)
                        line_eq = sp.Eq(y_sym - y1, slope * (x_sym - x1))
                    
                    equations.append(SymbolicExpression(sympy_obj=line_eq, variables=symbols))
                    current_point = (x2, y2)
            
            elif command_type == 'C':  # Cubic Bezier curve
                # For Bezier curves, we can't easily represent them as equations
                # We could approximate with parametric equations, but for now we'll skip
                if current_point and len(points) >= 6:
                    current_point = (points[4], points[5])  # End point
            
            elif command_type == 'Z':  # Close path
                # No additional equation needed, as the path is already closed
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
            >>> component = {'params': {'shape_type': 'circle', 'cx': 100, 'cy': 150, 'radius': 50}}
            >>> analyzer = ShapeEquationMathAnalyzer(component)
            >>> properties = analyzer.analyze_geometric_properties(0.5)
            >>> properties['area']
            7853.981633974483
        """
        # Get shape type
        shape_type = self.shape_type
        
        # Get parameters evaluated at the given time
        param_values = {}
        for param_name in self.SHAPE_PARAM_MAPS.get(shape_type, []):
            expr = self.get_parameter_as_symbolic_expression(param_name)
            if expr is not None:
                try:
                    value = expr.evaluate({'t': time_value})
                    param_values[param_name] = value
                except:
                    pass
        
        # Calculate properties based on shape type
        properties = {}
        
        if shape_type == 'circle':
            if 'radius' in param_values:
                radius = param_values['radius']
                properties['area'] = math.pi * radius**2
                properties['perimeter'] = 2 * math.pi * radius
                properties['center'] = (
                    param_values.get('cx', 0),
                    param_values.get('cy', 0)
                )
        
        elif shape_type == 'rectangle':
            if 'width' in param_values and 'height' in param_values:
                width = param_values['width']
                height = param_values['height']
                properties['area'] = width * height
                properties['perimeter'] = 2 * (width + height)
                properties['center'] = (
                    param_values.get('x', 0) + width / 2,
                    param_values.get('y', 0) + height / 2
                )
        
        elif shape_type == 'ellipse':
            if 'rx' in param_values and 'ry' in param_values:
                rx = param_values['rx']
                ry = param_values['ry']
                properties['area'] = math.pi * rx * ry
                # Approximation of ellipse perimeter
                h = ((rx - ry) / (rx + ry))**2
                properties['perimeter'] = math.pi * (rx + ry) * (1 + 3*h / (10 + math.sqrt(4 - 3*h)))
                properties['center'] = (
                    param_values.get('cx', 0),
                    param_values.get('cy', 0)
                )
        
        elif shape_type == 'line':
            if all(p in param_values for p in ['x1', 'y1', 'x2', 'y2']):
                x1, y1 = param_values['x1'], param_values['y1']
                x2, y2 = param_values['x2'], param_values['y2']
                properties['length'] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                properties['midpoint'] = ((x1 + x2) / 2, (y1 + y2) / 2)
                if x2 != x1:
                    properties['slope'] = (y2 - y1) / (x2 - x1)
                else:
                    properties['slope'] = float('inf')  # Vertical line
        
        elif shape_type == 'polygon':
            points = self.component['params'].get('points', [])
            if points and len(points) >= 3:
                # Calculate area using Shoelace formula
                area = 0
                for i in range(len(points)):
                    x1, y1 = points[i]
                    x2, y2 = points[(i + 1) % len(points)]
                    area += x1 * y2 - x2 * y1
                properties['area'] = abs(area) / 2
                
                # Calculate perimeter
                perimeter = 0
                for i in range(len(points)):
                    x1, y1 = points[i]
                    x2, y2 = points[(i + 1) % len(points)]
                    perimeter += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                properties['perimeter'] = perimeter
                
                # Calculate centroid
                cx = cy = 0
                for x, y in points:
                    cx += x
                    cy += y
                properties['center'] = (cx / len(points), cy / len(points))
        
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
        >>> expr.to_string()
        'x**2 + 3*y'
    """
    try:
        return SymbolicExpression(expression_str=text_equation, variables=known_symbols)
    except Exception as e:
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
        >>> expr1 = SymbolicExpression("x**2 + 2*x + 1")
        >>> expr2 = SymbolicExpression("(x + 1)**2")
        >>> are_expressions_equivalent(expr1, expr2)
        True
    """
    try:
        # Combine variables from both expressions
        variables = expr1.variables.copy()
        variables.update(expr2.variables)
        
        # Check if the difference simplifies to zero
        diff = sp.simplify(expr1.sympy_expr - expr2.sympy_expr)
        return diff == 0
    except Exception as e:
        print(f"Error comparing expressions: {str(e)}")
        return False


def substitute_values(expression: SymbolicExpression, 
                     assignments: Dict[str, Any]) -> SymbolicExpression:
    """
    Substitute values into a symbolic expression.
    
    Args:
        expression: The expression to substitute into
        assignments: Dictionary mapping variable names to values
        
    Returns:
        A new SymbolicExpression with substituted values
        
    Example:
        >>> expr = SymbolicExpression("x**2 + y")
        >>> result = substitute_values(expr, {"x": 3})
        >>> result.to_string()
        '9 + y'
    """
    try:
        # Create substitution dictionary
        subs_dict = {}
        for var_name, value in assignments.items():
            if var_name in expression.variables:
                subs_dict[expression.variables[var_name]] = value
        
        # Substitute values
        result_expr = expression.sympy_expr.subs(subs_dict)
        
        # Create new SymbolicExpression with updated expression
        return SymbolicExpression(sympy_obj=result_expr, variables=expression.variables.copy())
    except Exception as e:
        print(f"Error substituting values: {str(e)}")
        return expression


def create_symbolic_interpolation(keyframes: List[Tuple[float, float]], 
                                time_symbol: Optional[sp.Symbol] = None) -> SymbolicExpression:
    """
    Create a symbolic expression for interpolating between keyframes.
    
    Args:
        keyframes: List of (time, value) tuples
        time_symbol: Optional symbol to use for time
        
    Returns:
        A SymbolicExpression representing the interpolation
        
    Example:
        >>> t_sym = sp.Symbol('t')
        >>> expr = create_symbolic_interpolation([(0, 10), (1, 20)], t_sym)
        >>> expr.to_string()
        'Piecewise((10 + 10*t, (t >= 0) & (t <= 1)))'
    """
    # Use default time symbol if not provided
    time_symbol = time_symbol or sp.Symbol('t')
    
    # Sort keyframes by time
    sorted_keyframes = sorted(keyframes)
    
    # If there's only one keyframe, return a constant expression
    if len(sorted_keyframes) == 1:
        return SymbolicExpression(sympy_obj=sp.sympify(sorted_keyframes[0][1]),
                                 variables={'t': time_symbol})
    
    # For multiple keyframes, create a piecewise function
    pieces = []
    
    for i in range(len(sorted_keyframes) - 1):
        t0, v0 = sorted_keyframes[i]
        t1, v1 = sorted_keyframes[i + 1]
        
        # Skip if times are the same
        if t1 == t0:
            continue
        
        # Create linear interpolation expression for this segment
        slope = (v1 - v0) / (t1 - t0)
        expr = v0 + slope * (time_symbol - t0)
        
        # Add condition for this piece
        condition = sp.And(time_symbol >= t0, time_symbol <= t1)
        pieces.append((expr, condition))
    
    # If no valid pieces were created, return the first value
    if not pieces:
        return SymbolicExpression(sympy_obj=sp.sympify(sorted_keyframes[0][1]),
                                 variables={'t': time_symbol})
    
    # Create piecewise function
    piecewise_expr = sp.Piecewise(*pieces)
    
    return SymbolicExpression(sympy_obj=piecewise_expr,
                             variables={'t': time_symbol})


def calculate_symbolic_derivative(expression: SymbolicExpression, 
                                var_name: str, 
                                order: int = 1) -> SymbolicExpression:
    """
    Calculate the symbolic derivative of an expression.
    
    Args:
        expression: The expression to differentiate
        var_name: Name of the variable to differentiate with respect to
        order: Order of the derivative (default: 1)
        
    Returns:
        A new SymbolicExpression representing the derivative
        
    Example:
        >>> expr = SymbolicExpression("x**3 + 2*x*y")
        >>> derivative = calculate_symbolic_derivative(expr, "x")
        >>> derivative.to_string()
        '3*x**2 + 2*y'
    """
    try:
        # Check if the variable exists
        if var_name not in expression.variables:
            raise ValueError(f"Variable {var_name} not found in expression")
        
        # Get the symbol
        var_symbol = expression.variables[var_name]
        
        # Calculate the derivative
        derivative = expression.sympy_expr
        for _ in range(order):
            derivative = sp.diff(derivative, var_symbol)
        
        # Create new SymbolicExpression with the derivative
        return SymbolicExpression(sympy_obj=derivative, variables=expression.variables.copy())
    except Exception as e:
        print(f"Error calculating derivative: {str(e)}")
        return expression


# Example usage
if __name__ == "__main__":
    # Create a symbolic expression
    expr = SymbolicExpression("x**2 + 3*x*y + y**2")
    print(f"Expression: {expr.to_string()}")
    
    # Simplify the expression
    simplified = expr.simplify()
    print(f"Simplified: {simplified.to_string()}")
    
    # Differentiate with respect to x
    derivative = expr.differentiate("x")
    print(f"Derivative w.r.t. x: {derivative.to_string()}")
    
    # Evaluate at x=2, y=3
    value = expr.evaluate({"x": 2, "y": 3})
    print(f"Value at x=2, y=3: {value}")
    
    # Create a shape component
    circle_component = {
        'params': {
            'shape_type': 'circle',
            'cx': 100,
            'cy': 150,
            'radius': 50
        },
        'animations': {
            'radius': [
                {'time': 0, 'value': 50},
                {'time': 1, 'value': 100}
            ]
        }
    }
    
    # Analyze the shape
    analyzer = ShapeEquationMathAnalyzer(circle_component)
    
    # Get radius as a symbolic expression
    radius_expr = analyzer.get_parameter_as_symbolic_expression('radius')
    print(f"Radius expression: {radius_expr.to_string()}")
    
    # Get boundary equations
    equations = analyzer.get_shape_boundary_equations()
    print(f"Boundary equation: {equations[0].to_string()}")
    
    # Analyze geometric properties
    properties = analyzer.analyze_geometric_properties(0.5)
    print(f"Area at t=0.5: {properties['area']}")
    print(f"Perimeter at t=0.5: {properties['perimeter']}")
