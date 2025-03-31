# -*- coding: utf-8 -*-
"""
Registry for dynamically discovering and loading components.
"""

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

T = TypeVar("T")


class Registry:
    """
    Registry for dynamically discovering and loading components.

    This class implements the registry pattern, allowing components to be registered and retrieved by name.
    This enables dynamic discovery and loading of components, facilitating extensibility and modularity.
    """

    _registries: Dict[str, Dict[str, Type[Any]]] = {}

    @classmethod
    def register(cls, registry_name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a class in the specified registry.

        Parameters
        ----------
        registry_name : str
            Name of the registry to register the class in.

        Returns
        -------
        Callable[[Type[T]], Type[T]]
            Decorator function that registers the class.
        """

        def decorator(registered_class: Type[T]) -> Type[T]:
            if registry_name not in cls._registries:
                cls._registries[registry_name] = {}

            name = registered_class.__name__
            cls._registries[registry_name][name] = registered_class
            return registered_class

        return decorator

    @classmethod
    def get(cls, registry_name: str, name: str) -> Type[Any]:
        """
        Get a class from the registry by name.

        Parameters
        ----------
        registry_name : str
            Name of the registry to get the class from.
        name : str
            Name of the class to get.

        Returns
        -------
        Type[Any]
            The registered class.

        Raises
        ------
        KeyError
            If the registry or class is not found.
        """
        if registry_name not in cls._registries:
            raise KeyError(f"Registry '{registry_name}' not found")
        if name not in cls._registries[registry_name]:
            raise KeyError(f"'{name}' not found in registry '{registry_name}'")

        return cls._registries[registry_name][name]

    @classmethod
    def list(cls, registry_name: str) -> Dict[str, Type[Any]]:
        """
        List all registered classes in a registry.

        Parameters
        ----------
        registry_name : str
            Name of the registry to list classes from.

        Returns
        -------
        Dict[str, Type[Any]]
            Dictionary mapping class names to registered classes.

        Raises
        ------
        KeyError
            If the registry is not found.
        """
        if registry_name not in cls._registries:
            raise KeyError(f"Registry '{registry_name}' not found")

        return cls._registries[registry_name].copy()

    @classmethod
    def register_all(cls, registry_name: str, classes: List[Type[Any]]) -> None:
        """
        Register multiple classes in the specified registry.

        Parameters
        ----------
        registry_name : str
            Name of the registry to register the classes in.
        classes : List[Type[Any]]
            List of classes to register.
        """
        if registry_name not in cls._registries:
            cls._registries[registry_name] = {}

        for registered_class in classes:
            name = registered_class.__name__
            cls._registries[registry_name][name] = registered_class

    @classmethod
    def get_registry_names(cls) -> List[str]:
        """
        Get the names of all registries.

        Returns
        -------
        List[str]
            List of registry names.
        """
        return list(cls._registries.keys())

    @classmethod
    def clear(cls, registry_name: Optional[str] = None) -> None:
        """
        Clear a registry or all registries.

        Parameters
        ----------
        registry_name : str, optional
            Name of the registry to clear, or None to clear all registries.
        """
        if registry_name is None:
            cls._registries.clear()
        elif registry_name in cls._registries:
            cls._registries[registry_name].clear()
