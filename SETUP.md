# Setting up the Build Environment on a Non-Root Workstation

Free disk space around 20GB is expected.

## Install CMake

Run in terminal:

```bash
wget https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2-linux-x86_64.sh
chmod +x cmake-3.23.2-linux-x86_64.sh
./cmake-3.23.2-linux-x86_64.sh
```
Append in `~/.bashrc`:
```bash
export PATH=<PATH_TO_CMAKE>/bin:$PATH
```
Check:
```bash
source ~/.bashrc
which cmake # Should point to the newly installed CMake binary.
```

## Install Clang

Reference: https://clang.llvm.org/get_started.html.

Run in terminal:

```bash
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.1/llvm-project-12.0.1.src.tar.xz
tar -xf llvm-project-12.0.1.src.tar.xz
cd llvm-project-12.0.1.src
mkdir build && cd build
cmake -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../llvm
make -j 24 # Change the # of jobs as you like.
# Choose an install path you like.
# For example, LLVM_PATH could be /storage/tejwanir/social-mdp-core/llvm,
cmake -DCMAKE_INSTALL_PREFIX=<LLVM_PATH> -P cmake_install.cmake
```
Append in `.bashrc`:
```bash
export PATH=<LLVM_PATH>/bin:$PATH
```
Check:
```bash
source .bashrc
which clang++
clang++ -v
```

## Install CUDA Toolkit

Because we are non-root we here just install the compiler but not the drivers. So run `nvidia-smi` to check you have a CUDA-capable GPU and a compatible driver!

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run
chmod +x cuda_11.6.2_510.47.03_linux.run
./cuda_11.6.2_510.47.03_linux.run
```

In the CLI menu,
* uncheck "Driver";
* uncheck "Samples", "Demo Suite", and "Documentation" if you like, as these are not strictly required;
* in "Options > Toolkit Options",
  * in "Change Toolkit Install Path," change the toolkit install path to a path you have full control of. Let it be `<CUDA_HOME>`. For example: `/storage/tejwanir/cuda-11.6`
  * uncheck "Create symbolic links...," "Create desktop...," and "Install manpage documents," because we can't;
* in "Options > Library install path," enter the library path. We suggest `<CUDA_HOME>/lib64`. 

Finally, click "Install."

After the installation finished, append these lines to `.bashrc`:
```bash
export CUDA_HOME=<CUDA_HOME> # Replace with the installation directory.
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Check:
```bash
source .bashrc
which nvcc
nvcc --version
echo $LD_LIBRARY_PATH
```
## Install Vcpkg and Dependencies of this project

In a comfortable path, run
```bash
git clone https://github.com/microsoft/vcpkg
./vcpkg/bootstrap-vcpkg.sh
./vcpkg/vcpkg install fmt nlohmann-json sfml ftxui
```

## Building
In the root directory of the repo:
```shell
mkdir build && cd build
# For example, LLVM_PATH could be /storage/tejwanir/social-mdp-core/llvm,
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=<PATH_TO_VCPKG>/vcpkg/scripts/buildsystems/vcpkg.cmake \
	-DCMAKE_CUDA_HOST_COMPILE=<LLVM_PATH>/bin/clang++ \  
	-DCMAKE_CXX_COMPILER=<LLVM_PATH>/bin/clang++ \
	-DCMAKE_CUDA_ARCHITECTURES=80 \
	-DOpenGL_GL_PREFERENCE=GLVND
make
cd ..
```