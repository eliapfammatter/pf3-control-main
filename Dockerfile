# Linux x86-64 image so the FMU's linux64/*.so can be dlopen'd natively
# (under QEMU emulation on Apple Silicon). Pin the platform here so it's
# correct regardless of how you invoke `docker build`.
FROM --platform=linux/amd64 python:3.13-slim

# FMUs that link against the C/math runtime need these at load time.
# Add more (e.g. libgomp1 for OpenMP) if the FMU fails to load with a
# missing-symbol / missing-library error.
# zip/unzip: the SIMSEN RFMI client shells out to `zip` to package the FMU's
# local resources and send them to the remote SimsenRFMIServer (the Linux
# equivalent of the Windows RFMI_7Z / 7z.exe requirement in the docs).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 \
        libgomp1 \
        zip \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Install deps first so this layer caches across code edits.
# Use the compiled lockfile if you have it; otherwise see the note below.
COPY requirements.in ./
RUN pip install --no-cache-dir pip-tools && \
    pip-compile requirements.in -o requirements-linux.txt && \
    pip install --no-cache-dir -r requirements-linux.txt

# Source + FMU get mounted at runtime via -v, so nothing else to COPY.
# (If you'd rather bake the code in, uncomment:)
# COPY . .

CMD ["python", "compare_simulation_models.py", "--help"]
