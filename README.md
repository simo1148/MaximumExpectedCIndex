# ECindex

<One-line description of what the library does.>

## Installation

```bash
pip install ecindex
```

From a local checkout:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from ecindex import ec_index

ec_index([1.0, 2.0, 3.0])
```

## Development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest            # tests
ruff check .      # lint
ruff format .     # format
mypy              # type check
```

## License

MIT — see [LICENSE](LICENSE).
