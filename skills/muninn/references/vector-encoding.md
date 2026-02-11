# Vector Encoding Reference

Muninn vectors are raw float32 blobs (little-endian, `sizeof(float) * dim` bytes).
This is the single biggest source of errors when AI tools generate muninn code.

## Python

```python
import struct

dim = 384
values = [0.1, 0.2, 0.3]  # len must equal dim

# Encode
blob = struct.pack(f'{dim}f', *values)

# Decode
values = list(struct.unpack(f'{dim}f', blob))
```

## Node.js

```javascript
const dim = 384;
const values = new Float32Array([0.1, 0.2, 0.3]);

// Encode
const blob = Buffer.from(values.buffer);

// Decode
const decoded = new Float32Array(
  blob.buffer, blob.byteOffset, blob.byteLength / 4
);
```

## C

```c
float vec[384] = {0.1f, 0.2f, 0.3f};

// Bind as blob
sqlite3_bind_blob(stmt, col, vec, dim * sizeof(float), SQLITE_TRANSIENT);

// Read from column
const float *result = (const float *)sqlite3_column_blob(stmt, col);
int n = sqlite3_column_bytes(stmt, col) / sizeof(float);
```

## Common Errors

| Error | Fix |
|-------|-----|
| Passing JSON array `[0.1, 0.2]` | Use binary blob: `struct.pack()` / `Buffer.from()` |
| Wrong byte order | Muninn expects little-endian (native on x86/ARM) |
| Dimension mismatch | Insert vector length must match `FLOAT32[N]` in CREATE TABLE |
| Using `float64` / `double` | Must be `float32` — use `f` format in struct.pack, Float32Array in JS |
