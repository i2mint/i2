"""
Demo of the ingress decorator for castgraph

This file demonstrates the various ways to use the ingress decorator
to automatically transform function arguments to the expected kinds.
"""

from i2.castgraph import TransformationGraph
import json


def demo_basic_usage():
    """Basic usage: transform arguments to expected kinds."""
    print("\n=== Basic Usage Demo ===\n")

    graph = TransformationGraph()
    graph.add_node('text', isa=lambda x: isinstance(x, str))
    graph.add_node(int)

    # Register transformation: int -> text
    @graph.register_edge(int, 'text')
    def int_to_text(i, ctx):
        return str(i)

    # Decorate function to auto-transform first arg to 'text'
    @graph.ingress('text')
    def process(x):
        return x + " processed"

    # Can pass int, will be transformed to text automatically
    result = process(42)
    print(f"process(42) = {result!r}")
    assert result == "42 processed"


def demo_explicit_arg_name():
    """Specify which argument to transform."""
    print("\n=== Explicit Argument Name Demo ===\n")

    graph = TransformationGraph()

    @graph.register_edge(str, int)
    def str_to_int(s, ctx):
        return int(s)

    # Transform specific argument 'x' to int
    @graph.ingress(int, 'x')
    def add(x, y):
        return x + y

    # First arg gets transformed, second doesn't
    result = add("10", 32)
    print(f"add('10', 32) = {result}")
    assert result == 42


def demo_attribute_syntax():
    """Use attribute syntax for cleaner code."""
    print("\n=== Attribute Syntax Demo ===\n")

    graph = TransformationGraph()
    # Register int as a node so it can be used as an attribute
    graph.add_node(int)

    @graph.register_edge(str, int)
    def str_to_int(s, ctx):
        return int(s)

    # Use @graph.ingress.int instead of @graph.ingress(int)
    @graph.ingress.int
    def square(n):
        return n * n

    result = square("7")
    print(f"square('7') = {result}")
    assert result == 49


def demo_attribute_with_arg_name():
    """Attribute syntax with explicit argument name."""
    print("\n=== Attribute Syntax + Arg Name Demo ===\n")

    graph = TransformationGraph()
    graph.add_node('data', isa=lambda x: isinstance(x, dict))

    @graph.register_edge(str, 'data')
    def parse_json(s, ctx):
        return json.loads(s)

    # Use attribute syntax with argument name
    @graph.ingress.data('obj')
    def get_name(obj):
        return obj.get('name', 'unknown')

    result = get_name('{"name": "Alice"}')
    print(f"get_name(json_str) = {result!r}")
    assert result == "Alice"


def demo_multi_hop():
    """Multi-hop transformations through the graph."""
    print("\n=== Multi-hop Transformation Demo ===\n")

    graph = TransformationGraph()

    # Set up chain: str -> float -> int
    @graph.register_edge(str, float)
    def str_to_float(s, ctx):
        print(f"  Converting {s!r} (str) -> {float(s)} (float)")
        return float(s)

    @graph.register_edge(float, int)
    def float_to_int(f, ctx):
        print(f"  Converting {f} (float) -> {int(f)} (int)")
        return int(f)

    @graph.ingress(int)
    def double(n):
        print(f"  Doubling {n} (int)")
        return n * 2

    print("Calling double('21.9'):")
    result = double("21.9")
    print(f"Result: {result}")
    assert result == 42


def demo_with_context():
    """Pass context through transformations."""
    print("\n=== Context Propagation Demo ===\n")

    graph = TransformationGraph()

    @graph.register_edge(str, int)
    def str_to_int_with_base(s, ctx):
        base = (ctx or {}).get("base", 10)
        print(f"  Parsing {s!r} with base={base}")
        return int(s, base)

    # Bake context into decorator
    @graph.ingress(int, context={"base": 16})
    def add_one(n):
        return n + 1

    result = add_one("FF")
    print(f"add_one('FF') with hex context = {result}")
    assert result == 256  # FF = 255, + 1 = 256


def demo_real_world_pipeline():
    """Real-world example: processing configuration data."""
    print("\n=== Real-world Pipeline Demo ===\n")

    graph = TransformationGraph()

    # Define kinds for config data pipeline
    graph.add_node('json_str', isa=lambda x: isinstance(x, str) and x.startswith('{'))
    graph.add_node('config_dict', isa=lambda x: isinstance(x, dict))
    graph.add_node(
        'validated_config', isa=lambda x: isinstance(x, dict) and 'version' in x
    )

    @graph.register_edge('json_str', 'config_dict')
    def parse_config(s, ctx):
        print(f"  Parsing JSON config")
        return json.loads(s)

    @graph.register_edge('config_dict', 'validated_config')
    def validate_config(d, ctx):
        print(f"  Validating config")
        if 'version' not in d:
            d['version'] = '1.0'
        return d

    # Function that requires validated config
    @graph.ingress.validated_config('config')
    def deploy_service(config):
        print(f"  Deploying service with config version {config['version']}")
        return f"Service deployed (v{config['version']})"

    # Can pass raw JSON string, will be transformed through the pipeline
    config_json = '{"service": "api", "port": 8080}'
    print(f"Deploying with JSON: {config_json}")
    result = deploy_service(config_json)
    print(f"Result: {result}")
    assert result == "Service deployed (v1.0)"


if __name__ == '__main__':
    demo_basic_usage()
    demo_explicit_arg_name()
    demo_attribute_syntax()
    demo_attribute_with_arg_name()
    demo_multi_hop()
    demo_with_context()
    demo_real_world_pipeline()

    print("\n" + "=" * 50)
    print("All demos passed! ✓")
    print("=" * 50)
