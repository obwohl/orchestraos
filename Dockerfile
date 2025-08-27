# --- STAGE 1: Die "Builder"-Umgebung ---
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# --- 1. System-Vorbereitung (inkl. Zlib für eine saubere Konfiguration) ---
RUN apt-get update && apt-get install -y \
    build-essential \
    ccache \
    cmake \
    git \
    lld \
    ninja-build \
    python3 \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# --- 2. Klonen der Repositories (mit Fortschrittsanzeige) ---
WORKDIR /work
RUN git clone --progress --depth=1 --single-branch https://github.com/openxla/stablehlo.git && \
    cd stablehlo && \
    git clone --progress --depth=1 --single-branch https://github.com/llvm/llvm-project.git

WORKDIR /work/stablehlo

# --- 3. Korrekten LLVM-Commit auschecken (mit Fortschrittsanzeige) ---
# ACHTUNG: KANN SEHR LANGE DAUERN (10-30+ Minuten).
RUN hash="$(cat ./build_tools/llvm_version.txt)" && \
    cd llvm-project && \
    git fetch --progress origin "$hash" && \
    git checkout "$hash"

# --- 4. LLVM/MLIR bauen (mit expliziter Job-Limitierung - KORRIGIERTE SYNTAX) ---
# ACHTUNG: DER ZEITAUFWENDIGSTE SCHRITT (jetzt langsamer, aber stabiler).
# KORREKTUR: Die Modifikation des Build-Skripts ist fehleranfällig. Wir rufen CMake direkt auf
# mit der korrekten Syntax für das Job-Limit.
RUN ./build_tools/build_mlir.sh "${PWD}/llvm-project" "${PWD}/llvm-build"
# Wir überschreiben den Build-Befehl aus dem Skript mit der korrekten Variante.
RUN cmake --build /work/stablehlo/llvm-build -- -j2

# --- 5. StableHLO bauen und installieren (mit expliziter Job-Limitierung) ---
RUN mkdir build && \
    cd build && \
    cmake .. -GNinja \
      -DCMAKE_BUILD_TYPE='Release' \
      -DLLVM_ENABLE_ASSERTIONS='ON' \
      -DSTABLEHLO_ENABLE_BINDINGS_PYTHON='OFF' \
      -DMLIR_DIR=/work/stablehlo/llvm-build/lib/cmake/mlir \
      -DCMAKE_INSTALL_PREFIX=/opt/stablehlo_install && \
    cmake --build . -- -j2 && \
    cmake --install .

# --- 6. Tests ausführen ---
RUN cd build && ninja check-stablehlo-tests


# --- STAGE 2: Der finale "Artifact Packaging"-Stage ---
FROM ubuntu:24.04

WORKDIR /artifacts
RUN mkdir stablehlo-only stablehlo-sdk

# Kopiert die vollständige, "installierte" StableHLO-Distribution.
COPY --from=builder /opt/stablehlo_install ./stablehlo-only/

# Kopiert die Artefakte für die "stablehlo-sdk" Version.
COPY --from=builder /opt/stablehlo_install ./stablehlo-sdk/stablehlo/
COPY --from=builder /work/stablehlo/llvm-build ./stablehlo-sdk/llvm/

# Erstellt die beiden .tar.gz-Dateien.
CMD ["sh", "-c", "echo 'Creating archives...'; \
    tar -czf /artifacts/stablehlo-only.tar.gz -C /artifacts/stablehlo-only . && \
    tar -czf /artifacts/stablehlo-sdk.tar.gz -C /artifacts/stablehlo-sdk . && \
    echo 'Archives created successfully in /artifacts/. Container is running.' && \
    sleep infinity"]