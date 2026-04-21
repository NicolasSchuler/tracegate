"""Vulture whitelist for framework false positives.

Frameworks often use string-based dispatch, lifecycle hooks, and decorators
that vulture cannot trace. Declare dynamically-used symbols here.
"""

# Example for a class with framework callbacks:
# from llmdebug.example import ExampleClass
# ExampleClass.on_mount
# ExampleClass.on_unmount
# ExampleClass.compose
