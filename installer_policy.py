"""Lab-pinned, user-local source builds. Root access is limited to prerequisites."""
import shlex
def build_script(mode, repository="", package="", toolkit=""):
    if mode not in ("cpu","gpu"):
        raise ValueError("Unknown build mode")
    preamble = ""
    if mode == "gpu":
        preamble = f"""
wget -q -O "$work/keyring.deb" {shlex.quote(repository + "/cuda-keyring_1.1-1_all.deb")}
sudo dpkg -i "$work/keyring.deb"
sudo apt-get update
sudo apt-get install -y {shlex.quote(package)}
export PATH="/usr/local/cuda-{toolkit}/bin:$PATH"
command -v nvcc >/dev/null || fail "CUDA compiler missing"
printf '__global__ void k(){{}}\nint main(){{return 0;}}\n' > "$work/check.cu"
nvcc -ccbin "$CXX" "$work/check.cu" -o "$work/check" || fail "CUDA rejects the selected host compiler; set CC/CXX to supported compilers."
"""
    return """set -euo pipefail
fail() { echo "Preflight failed: $*" >&2; exit 1; }
[ "$(id -u)" -ne 0 ] || fail "Run setup as your normal user, not root."
command -v sudo >/dev/null || fail "sudo is required for build prerequisites."
available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
[ "$available" -ge 2097152 ] || fail "At least 2 GiB available RAM required."
free_kb=$(df -Pk "$HOME" | awk 'END {print $4}')
[ "$free_kb" -ge 6291456 ] || fail "At least 6 GiB free disk space required."
jobs=$((available / 2097152))
cores=$(nproc)
[ "$jobs" -le "$cores" ] || jobs=$cores
[ "$jobs" -le 4 ] || jobs=4
sudo apt-get update
sudo apt-get install -y ca-certificates wget build-essential cmake
cmake_version=$(cmake --version | head -1 | awk '{print $3}')
[ "$(printf '%s\n' 3.28 "$cmake_version" | sort -V | head -1)" = 3.28 ] || fail "CMake 3.28+ required; upgrade CMake."
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
command -v "$CC" >/dev/null || fail "C compiler missing"
command -v "$CXX" >/dev/null || fail "C++ compiler missing"
work=$(mktemp -d "$HOME/.md-build.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
""" + preamble + f"""
prefix="$HOME/software/gromacs-2026.2-{mode}"
[ ! -e "$prefix" ] || fail "Installation directory exists; choose the existing executable or move it manually."
echo "Building lab default 2026.2 ({mode}), jobs=$jobs, destination=$prefix"
cd "$work"
wget -q -O source.tar.gz https://ftp.gromacs.org/gromacs/gromacs-2026.2.tar.gz
echo "74c859cb7f53919f9c7760c287b43323  source.tar.gz" | md5sum --check -
tar -xzf source.tar.gz
cmake -S gromacs-2026.2 -B build -DGMX_GPU={'CUDA' if mode == 'gpu' else 'OFF'} -DGMX_BUILD_OWN_FFTW=ON -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
cmake --build build --parallel "$jobs"
cmake --install build
"$prefix/bin/gmx" --version > "$prefix/build-info.txt" 2>&1
cat "$prefix/build-info.txt"
"""
