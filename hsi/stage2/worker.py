"""Fresh-process entry: only import the selected stage's numerical runtime."""
from pathlib import Path
import argparse
import json


def execute(operation, arguments):
    values = dict(arguments)
    if operation == 'views':
        from .views import select_views
        return select_views(**values)
    if operation == 'sdf':
        from .sdf import prepare
        return prepare(**values)
    if operation == 'h3':
        from .h3 import generate
        return generate(**values)
    if operation == 'recovery':
        from .recovery import recover
        return recover(**values)
    if operation == 'placement':
        from .placement import run
        return run(**values)
    if operation == 'render':
        from .render_evidence import run
        return run(**{key: Path(value) if key.endswith('path') or key == 'output' else value
                      for key, value in values.items()})
    raise ValueError('Unknown Stage2 worker operation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', choices=('views', 'sdf', 'h3', 'recovery', 'placement', 'render'), required=True)
    parser.add_argument('--arguments', type=Path, required=True)
    args = parser.parse_args()
    from hsi.common.artifacts import read_sealed
    execute(args.operation, read_sealed(args.arguments)['arguments'])


if __name__ == '__main__':
    main()
