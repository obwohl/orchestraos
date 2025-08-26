# --- STAGE 1: Die "Builder"-Umgebung ---
# Dieser Stage dient als gemeinsame Basis und wird am Ende verworfen.
# Hier werden alle Werkzeuge installiert und der gesamte Code kompiliert.
FROM ubuntu:24.04 AS builder

# Umgebungsvariablen für einen nicht-interaktiven Build
ENV DEBIAN_FRONTEND=noninteractive

# --- 1. System-Vorbereitung (Gemäß README) ---
RUN apt-get update && apt-get install -y \
    build-essential \
    ccache \
    cmake \
    git \
    lld \
    ninja-build \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# --- 2. Klonen der Repositories (mit Fortschrittsanzeige) ---
WORKDIR /work
RUN git clone --progress --depth=1 --single-branch https://github.com/openxla/stablehlo.git && \
    cd stablehlo && \
    git clone --progress --depth=1 --single-branch https://github.com/llvm/llvm-project.git

# Arbeitsverzeichnis auf das geklonte StableHLO-Repo setzen
WORKDIR /work/stablehlo

# --- 3. Korrekten LLVM-Commit auschecken (mit Fortschrittsanzeige) ---
# ACHTUNG: DIESER SCHRITT KANN SEHR LANGE DAUERN (10-30+ Minuten).
RUN hash="$(cat ./build_tools/llvm_version.txt)" && \
    cd llvm-project && \
    git fetch --progress origin "$hash" && \
    git checkout "$hash"

# --- 4. LLVM/MLIR bauen (Gemäß README) ---
# ACHTUNG: DIES IST DER ZEITAUFWENDIGSTE SCHRITT (30-90+ Minuten).
RUN ./build_tools/build_mlir.sh "${PWD}/llvm-project" "${PWD}/llvm-build"

# --- 5. StableHLO bauen und installieren (Gemäß README) ---
# Dies ist der entscheidende Schritt, der die "missing library files" behebt.
# `cmake --install .` sammelt alle Binaries, Header UND die wichtigen .so-Bibliotheken
# und legt sie in einer sauberen Struktur unter /opt/stablehlo_install ab.
RUN mkdir build && \
    cd build && \
    cmake .. -GNinja \
      -DCMAKE_BUILD_TYPE='Release' \
      -DLLVM_ENABLE_ASSERTIONS='ON' \
      -DSTABLEHLO_ENABLE_BINDINGS_PYTHON='OFF' \
      -DMLIR_DIR=/work/stablehlo/llvm-build/lib/cmake/mlir \
      -DCMAKE_INSTALL_PREFIX=/opt/stablehlo_install && \
    cmake --build . && \
    cmake --install .


# ==============================================================================
# --- STAGE 2: Der finale "Artifact Packaging"-Stage ---
# Dieser Stage ist das Standardziel. Er startet mit einem sauberen Image,
# kopiert die Ergebnisse aus dem "builder" und verpackt sie in .tar.gz-Archive.
# ==============================================================================
FROM ubuntu:24.04

# --- 8. Finale Artefakte aus dem Builder kopieren ---
# Erstellt das Zielverzeichnis für die gepackten Archive.
WORKDIR /artifacts

# Kopiert die vollständige, "installierte" StableHLO-Distribution.
COPY --from=builder /opt/stablehlo_install ./stablehlo_install/

# Kopiert die vollständige, kompilierte LLVM-Toolchain.
COPY --from=builder /work/stablehlo/llvm-build ./llvm_build/

# --- 9. Artefakte verpacken ---
# Dieser Befehl erstellt die beiden .tar.gz-Dateien.
# Der `-C` Befehl sorgt dafür, dass die Archive keine übergeordneten Pfade enthalten,
# sodass sie sauber in jedem Zielverzeichnis entpackt werden können.
CMD ["sh", "-c", "echo 'Creating archives...'; \
    tar -czf /artifacts/stablehlo-only.tar.gz -C /artifacts/stablehlo_install . && \
    mkdir -p sdk_temp/stablehlo && cp -r stablehlo_install/* sdk_temp/stablehlo/ && cp -r llvm_build sdk_temp/llvm && \
    tar -czf /artifacts/stablehlo-sdk.tar.gz -C /artifacts/sdk_temp . && \
    echo 'Archives created successfully in /artifacts/. Container is running.' && \
    sleep infinity"]