#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Integration Controller for Basira System

This module implements the System Integration Controller, which is responsible for
integrating all components of the Basira system into a unified system.

Author: Basira System Development Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum

# Configure logging
logger = logging.getLogger('integration_layer.system_integration_controller')


class IntegrationStrategy(str, Enum):
    """Integration strategies for the System Integration Controller."""
    HIERARCHICAL = "hierarchical"  # Hierarchical integration
    MODULAR = "modular"  # Modular integration
    HOLISTIC = "holistic"  # Holistic integration


class CoordinationMechanism(str, Enum):
    """Coordination mechanisms for the System Integration Controller."""
    CENTRALIZED = "centralized"  # Centralized coordination
    DISTRIBUTED = "distributed"  # Distributed coordination
    HYBRID = "hybrid"  # Hybrid coordination


class SystemIntegrationController:
    """
    System Integration Controller for the Basira System.
    
    This class is responsible for integrating all components of the Basira system
    into a unified system, and for coordinating the interactions between components.
    """
    
    def __init__(self, 
                integration_strategy: IntegrationStrategy = IntegrationStrategy.HOLISTIC,
                coordination_mechanism: CoordinationMechanism = CoordinationMechanism.HYBRID,
                system_monitoring: bool = True):
        """
        Initialize the System Integration Controller.
        
        Args:
            integration_strategy: Strategy for integrating components
            coordination_mechanism: Mechanism for coordinating components
            system_monitoring: Whether to enable system monitoring
        """
        self.logger = logging.getLogger('integration_layer.system_integration_controller.main')
        
        # Set integration strategy
        self.integration_strategy = integration_strategy if isinstance(integration_strategy, IntegrationStrategy) else IntegrationStrategy(integration_strategy)
        
        # Set coordination mechanism
        self.coordination_mechanism = coordination_mechanism if isinstance(coordination_mechanism, CoordinationMechanism) else CoordinationMechanism(coordination_mechanism)
        
        # Set system monitoring
        self.system_monitoring = system_monitoring
        
        # Initialize component registry
        self.component_registry = {}
        
        # Initialize interaction registry
        self.interaction_registry = {}
        
        # Initialize monitoring data
        self.monitoring_data = {}
        
        self.logger.info(f"System Integration Controller initialized with strategy: {self.integration_strategy.value}, mechanism: {self.coordination_mechanism.value}")
    
    def register_component(self, component_id: str, component: Any, component_type: str, dependencies: List[str] = None) -> None:
        """
        Register a component with the System Integration Controller.
        
        Args:
            component_id: Unique identifier for the component
            component: The component object
            component_type: Type of the component
            dependencies: List of component IDs that this component depends on
        """
        if component_id in self.component_registry:
            self.logger.warning(f"Component {component_id} already registered, will be replaced")
        
        self.component_registry[component_id] = {
            "component": component,
            "type": component_type,
            "dependencies": dependencies or [],
            "status": "registered"
        }
        
        self.logger.debug(f"Component {component_id} registered with type {component_type}")
    
    def register_interaction(self, source_id: str, target_id: str, interaction_type: str, parameters: Dict[str, Any] = None) -> None:
        """
        Register an interaction between components.
        
        Args:
            source_id: ID of the source component
            target_id: ID of the target component
            interaction_type: Type of interaction
            parameters: Parameters for the interaction
        """
        interaction_id = f"{source_id}_{target_id}_{interaction_type}"
        
        if interaction_id in self.interaction_registry:
            self.logger.warning(f"Interaction {interaction_id} already registered, will be replaced")
        
        self.interaction_registry[interaction_id] = {
            "source_id": source_id,
            "target_id": target_id,
            "type": interaction_type,
            "parameters": parameters or {},
            "status": "registered"
        }
        
        self.logger.debug(f"Interaction {interaction_id} registered")
    
    def initialize_components(self) -> None:
        """Initialize all registered components in the correct order."""
        # Get dependency order
        initialization_order = self._get_initialization_order()
        
        # Initialize components in order
        for component_id in initialization_order:
            try:
                component_info = self.component_registry[component_id]
                component = component_info["component"]
                
                # Check if component has initialize method
                if hasattr(component, "initialize") and callable(getattr(component, "initialize")):
                    component.initialize()
                
                component_info["status"] = "initialized"
                self.logger.info(f"Component {component_id} initialized")
            except Exception as e:
                component_info["status"] = "initialization_failed"
                self.logger.error(f"Failed to initialize component {component_id}: {e}")
    
    def _get_initialization_order(self) -> List[str]:
        """
        Get the order in which components should be initialized based on dependencies.
        
        Returns:
            List of component IDs in initialization order
        """
        # Build dependency graph
        graph = {component_id: set(info["dependencies"]) for component_id, info in self.component_registry.items()}
        
        # Topological sort
        result = []
        visited = set()
        temp_visited = set()
        
        def visit(node):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node}")
            if node in visited:
                return
            
            temp_visited.add(node)
            
            for dependency in graph.get(node, set()):
                if dependency in self.component_registry:
                    visit(dependency)
            
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        # Visit all nodes
        for component_id in self.component_registry:
            if component_id not in visited:
                visit(component_id)
        
        # Reverse to get correct order
        return list(reversed(result))
    
    def start_system(self) -> None:
        """Start the system by initializing components and enabling interactions."""
        self.logger.info("Starting system...")
        
        # Initialize components
        self.initialize_components()
        
        # Enable interactions
        self._enable_interactions()
        
        # Start monitoring if enabled
        if self.system_monitoring:
            self._start_monitoring()
        
        self.logger.info("System started successfully")
    
    def _enable_interactions(self) -> None:
        """Enable all registered interactions."""
        for interaction_id, interaction_info in self.interaction_registry.items():
            source_id = interaction_info["source_id"]
            target_id = interaction_info["target_id"]
            
            # Check if both source and target components are initialized
            if (source_id in self.component_registry and 
                self.component_registry[source_id]["status"] == "initialized" and
                target_id in self.component_registry and
                self.component_registry[target_id]["status"] == "initialized"):
                
                interaction_info["status"] = "enabled"
                self.logger.debug(f"Interaction {interaction_id} enabled")
            else:
                interaction_info["status"] = "disabled"
                self.logger.warning(f"Interaction {interaction_id} disabled due to component initialization issues")
    
    def _start_monitoring(self) -> None:
        """Start system monitoring."""
        self.logger.info("Starting system monitoring...")
        
        # Initialize monitoring data
        self.monitoring_data = {
            "components": {component_id: {"status": info["status"]} for component_id, info in self.component_registry.items()},
            "interactions": {interaction_id: {"status": info["status"]} for interaction_id, info in self.interaction_registry.items()},
            "system_status": "running"
        }
        
        self.logger.info("System monitoring started")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the system.
        
        Returns:
            Dictionary with system status information
        """
        if not self.system_monitoring:
            return {"system_status": "monitoring_disabled"}
        
        # Update monitoring data
        for component_id, info in self.component_registry.items():
            self.monitoring_data["components"][component_id]["status"] = info["status"]
        
        for interaction_id, info in self.interaction_registry.items():
            self.monitoring_data["interactions"][interaction_id]["status"] = info["status"]
        
        return self.monitoring_data
    
    def stop_system(self) -> None:
        """Stop the system by disabling interactions and shutting down components."""
        self.logger.info("Stopping system...")
        
        # Disable interactions
        for interaction_id, interaction_info in self.interaction_registry.items():
            interaction_info["status"] = "disabled"
        
        # Shutdown components in reverse initialization order
        shutdown_order = self._get_initialization_order()
        shutdown_order.reverse()
        
        for component_id in shutdown_order:
            try:
                component_info = self.component_registry[component_id]
                component = component_info["component"]
                
                # Check if component has shutdown method
                if hasattr(component, "shutdown") and callable(getattr(component, "shutdown")):
                    component.shutdown()
                
                component_info["status"] = "shutdown"
                self.logger.info(f"Component {component_id} shutdown")
            except Exception as e:
                component_info["status"] = "shutdown_failed"
                self.logger.error(f"Failed to shutdown component {component_id}: {e}")
        
        # Update system status
        if self.system_monitoring:
            self.monitoring_data["system_status"] = "stopped"
        
        self.logger.info("System stopped")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create System Integration Controller
    controller = SystemIntegrationController(
        integration_strategy=IntegrationStrategy.HOLISTIC,
        coordination_mechanism=CoordinationMechanism.HYBRID,
        system_monitoring=True
    )
    
    # Register some dummy components
    class DummyComponent:
        def initialize(self):
            print(f"Initializing {self.__class__.__name__}")
        
        def shutdown(self):
            print(f"Shutting down {self.__class__.__name__}")
    
    controller.register_component("component1", DummyComponent(), "dummy")
    controller.register_component("component2", DummyComponent(), "dummy", ["component1"])
    controller.register_component("component3", DummyComponent(), "dummy", ["component2"])
    
    # Register some interactions
    controller.register_interaction("component1", "component2", "data_flow")
    controller.register_interaction("component2", "component3", "data_flow")
    
    # Start the system
    controller.start_system()
    
    # Get system status
    status = controller.get_system_status()
    print(f"System status: {status['system_status']}")
    
    # Stop the system
    controller.stop_system()
