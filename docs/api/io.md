
# File I/O API

USEL provides eight public functions for reading and writing CSV, JSON, NumPy binary (`.npy`), and plain-text files. All functions are exported from `usel.io`.


## Reading files

### `read_csv(filepath, delimiter=",", skip_header=0)`

Read a CSV file into a NumPy array.

- `filepath` (`str | Path`): Input file path.
- `delimiter` (`str`): Field separator; defaults to `","`.
- `skip_header` (`int`): Number of header lines to skip; defaults to `0`.
- **Returns:** `numpy.ndarray` containing floating-point data.
- **Raises:** `IOError_` if the file is missing or reading fails.

### `read_json(filepath)`

Read and parse a JSON file.

- `filepath` (`str | Path`): Input file path.
- **Returns:** `Any`, depending on the JSON content.
- **Raises:** `IOError_` if the file is missing or reading/parsing fails.

### `read_npy(filepath)`

Load a NumPy `.npy` binary file with `allow_pickle=False`.

- `filepath` (`str | Path`): Input file path.
- **Returns:** `numpy.ndarray`.
- **Raises:** `IOError_` if the file is missing or loading fails.

### `read_text(filepath, encoding="utf-8")`

Read a text file into a string.

- `filepath` (`str | Path`): Input file path.
- `encoding` (`str`): Text encoding; defaults to `"utf-8"`.
- **Returns:** `str`.
- **Raises:** `IOError_` if the file is missing or reading fails.

## Writing files

### `write_csv(data, filepath, delimiter=",")`

Write a NumPy array to a CSV file.

- `data` (`numpy.ndarray`): Data converted using `numpy.asarray`.
- `filepath` (`str | Path`): Destination path.
- `delimiter` (`str`): Field separator; defaults to `","`.
- **Returns:** `None`.
- **Raises:** `IOError_` if writing fails.

### `write_json(data, filepath, indent=2)`

Serialize a JSON-compatible object and write it to a file.

- `data` (`Any`): Object accepted by Python's JSON encoder.
- `filepath` (`str | Path`): Destination path.
- `indent` (`int`): JSON indentation level; defaults to `2`.
- **Returns:** `None`.
- **Raises:** `IOError_` if serialization or writing fails.

### `write_npy(data, filepath)`

Write data to a NumPy `.npy` binary file.

- `data` (`numpy.ndarray`): Data converted using `numpy.asarray`.
- `filepath` (`str | Path`): Destination path.
- **Returns:** `None`.
- **Raises:** `IOError_` if writing fails.

### `write_text(content, filepath, encoding="utf-8")`

Write a string to a text file.

- `content` (`str`): Text to write.
- `filepath` (`str | Path`): Destination path.
- `encoding` (`str`): Text encoding; defaults to `"utf-8"`.
- **Returns:** `None`.
- **Raises:** `IOError_` if writing fails.

## Notes and limitations

- All functions accept file paths as strings or `pathlib.Path` objects.
- Read and write failures are wrapped in USEL's `IOError_` exception.
- Write functions do not explicitly create missing parent directories.
- JSON values must be serializable by Python's JSON encoder.
- NumPy `.npy` loading disables pickle support.
