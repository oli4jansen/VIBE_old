# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /content/openpose

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /content/openpose/build

# Include any dependencies generated for this target.
include src/openpose/pose/CMakeFiles/openpose_pose.dir/depend.make

# Include the progress variables for this target.
include src/openpose/pose/CMakeFiles/openpose_pose.dir/progress.make

# Include the compile flags for this target's objects.
include src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make

src/openpose/pose/CMakeFiles/openpose_pose.dir/openpose_pose_generated_renderPose.cu.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/openpose_pose_generated_renderPose.cu.o.depend
src/openpose/pose/CMakeFiles/openpose_pose.dir/openpose_pose_generated_renderPose.cu.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/openpose_pose_generated_renderPose.cu.o.Release.cmake
src/openpose/pose/CMakeFiles/openpose_pose.dir/openpose_pose_generated_renderPose.cu.o: ../src/openpose/pose/renderPose.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object src/openpose/pose/CMakeFiles/openpose_pose.dir/openpose_pose_generated_renderPose.cu.o"
	cd /content/openpose/build/src/openpose/pose/CMakeFiles/openpose_pose.dir && /usr/local/bin/cmake -E make_directory /content/openpose/build/src/openpose/pose/CMakeFiles/openpose_pose.dir//.
	cd /content/openpose/build/src/openpose/pose/CMakeFiles/openpose_pose.dir && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/content/openpose/build/src/openpose/pose/CMakeFiles/openpose_pose.dir//./openpose_pose_generated_renderPose.cu.o -D generated_cubin_file:STRING=/content/openpose/build/src/openpose/pose/CMakeFiles/openpose_pose.dir//./openpose_pose_generated_renderPose.cu.o.cubin.txt -P /content/openpose/build/src/openpose/pose/CMakeFiles/openpose_pose.dir//openpose_pose_generated_renderPose.cu.o.Release.cmake

src/openpose/pose/CMakeFiles/openpose_pose.dir/defineTemplates.cpp.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make
src/openpose/pose/CMakeFiles/openpose_pose.dir/defineTemplates.cpp.o: ../src/openpose/pose/defineTemplates.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/openpose/pose/CMakeFiles/openpose_pose.dir/defineTemplates.cpp.o"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openpose_pose.dir/defineTemplates.cpp.o -c /content/openpose/src/openpose/pose/defineTemplates.cpp

src/openpose/pose/CMakeFiles/openpose_pose.dir/defineTemplates.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openpose_pose.dir/defineTemplates.cpp.i"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/openpose/src/openpose/pose/defineTemplates.cpp > CMakeFiles/openpose_pose.dir/defineTemplates.cpp.i

src/openpose/pose/CMakeFiles/openpose_pose.dir/defineTemplates.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openpose_pose.dir/defineTemplates.cpp.s"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/openpose/src/openpose/pose/defineTemplates.cpp -o CMakeFiles/openpose_pose.dir/defineTemplates.cpp.s

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseCpuRenderer.cpp.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make
src/openpose/pose/CMakeFiles/openpose_pose.dir/poseCpuRenderer.cpp.o: ../src/openpose/pose/poseCpuRenderer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/openpose/pose/CMakeFiles/openpose_pose.dir/poseCpuRenderer.cpp.o"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openpose_pose.dir/poseCpuRenderer.cpp.o -c /content/openpose/src/openpose/pose/poseCpuRenderer.cpp

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseCpuRenderer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openpose_pose.dir/poseCpuRenderer.cpp.i"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/openpose/src/openpose/pose/poseCpuRenderer.cpp > CMakeFiles/openpose_pose.dir/poseCpuRenderer.cpp.i

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseCpuRenderer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openpose_pose.dir/poseCpuRenderer.cpp.s"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/openpose/src/openpose/pose/poseCpuRenderer.cpp -o CMakeFiles/openpose_pose.dir/poseCpuRenderer.cpp.s

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractor.cpp.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make
src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractor.cpp.o: ../src/openpose/pose/poseExtractor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractor.cpp.o"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openpose_pose.dir/poseExtractor.cpp.o -c /content/openpose/src/openpose/pose/poseExtractor.cpp

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openpose_pose.dir/poseExtractor.cpp.i"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/openpose/src/openpose/pose/poseExtractor.cpp > CMakeFiles/openpose_pose.dir/poseExtractor.cpp.i

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openpose_pose.dir/poseExtractor.cpp.s"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/openpose/src/openpose/pose/poseExtractor.cpp -o CMakeFiles/openpose_pose.dir/poseExtractor.cpp.s

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffe.cpp.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make
src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffe.cpp.o: ../src/openpose/pose/poseExtractorCaffe.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffe.cpp.o"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openpose_pose.dir/poseExtractorCaffe.cpp.o -c /content/openpose/src/openpose/pose/poseExtractorCaffe.cpp

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffe.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openpose_pose.dir/poseExtractorCaffe.cpp.i"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/openpose/src/openpose/pose/poseExtractorCaffe.cpp > CMakeFiles/openpose_pose.dir/poseExtractorCaffe.cpp.i

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffe.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openpose_pose.dir/poseExtractorCaffe.cpp.s"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/openpose/src/openpose/pose/poseExtractorCaffe.cpp -o CMakeFiles/openpose_pose.dir/poseExtractorCaffe.cpp.s

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffeStaf.cpp.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make
src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffeStaf.cpp.o: ../src/openpose/pose/poseExtractorCaffeStaf.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffeStaf.cpp.o"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openpose_pose.dir/poseExtractorCaffeStaf.cpp.o -c /content/openpose/src/openpose/pose/poseExtractorCaffeStaf.cpp

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffeStaf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openpose_pose.dir/poseExtractorCaffeStaf.cpp.i"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/openpose/src/openpose/pose/poseExtractorCaffeStaf.cpp > CMakeFiles/openpose_pose.dir/poseExtractorCaffeStaf.cpp.i

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffeStaf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openpose_pose.dir/poseExtractorCaffeStaf.cpp.s"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/openpose/src/openpose/pose/poseExtractorCaffeStaf.cpp -o CMakeFiles/openpose_pose.dir/poseExtractorCaffeStaf.cpp.s

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffeTaf.cpp.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make
src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffeTaf.cpp.o: ../src/openpose/pose/poseExtractorCaffeTaf.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffeTaf.cpp.o"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openpose_pose.dir/poseExtractorCaffeTaf.cpp.o -c /content/openpose/src/openpose/pose/poseExtractorCaffeTaf.cpp

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffeTaf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openpose_pose.dir/poseExtractorCaffeTaf.cpp.i"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/openpose/src/openpose/pose/poseExtractorCaffeTaf.cpp > CMakeFiles/openpose_pose.dir/poseExtractorCaffeTaf.cpp.i

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffeTaf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openpose_pose.dir/poseExtractorCaffeTaf.cpp.s"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/openpose/src/openpose/pose/poseExtractorCaffeTaf.cpp -o CMakeFiles/openpose_pose.dir/poseExtractorCaffeTaf.cpp.s

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorNet.cpp.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make
src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorNet.cpp.o: ../src/openpose/pose/poseExtractorNet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorNet.cpp.o"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openpose_pose.dir/poseExtractorNet.cpp.o -c /content/openpose/src/openpose/pose/poseExtractorNet.cpp

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorNet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openpose_pose.dir/poseExtractorNet.cpp.i"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/openpose/src/openpose/pose/poseExtractorNet.cpp > CMakeFiles/openpose_pose.dir/poseExtractorNet.cpp.i

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorNet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openpose_pose.dir/poseExtractorNet.cpp.s"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/openpose/src/openpose/pose/poseExtractorNet.cpp -o CMakeFiles/openpose_pose.dir/poseExtractorNet.cpp.s

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseGpuRenderer.cpp.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make
src/openpose/pose/CMakeFiles/openpose_pose.dir/poseGpuRenderer.cpp.o: ../src/openpose/pose/poseGpuRenderer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/openpose/pose/CMakeFiles/openpose_pose.dir/poseGpuRenderer.cpp.o"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openpose_pose.dir/poseGpuRenderer.cpp.o -c /content/openpose/src/openpose/pose/poseGpuRenderer.cpp

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseGpuRenderer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openpose_pose.dir/poseGpuRenderer.cpp.i"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/openpose/src/openpose/pose/poseGpuRenderer.cpp > CMakeFiles/openpose_pose.dir/poseGpuRenderer.cpp.i

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseGpuRenderer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openpose_pose.dir/poseGpuRenderer.cpp.s"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/openpose/src/openpose/pose/poseGpuRenderer.cpp -o CMakeFiles/openpose_pose.dir/poseGpuRenderer.cpp.s

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseParameters.cpp.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make
src/openpose/pose/CMakeFiles/openpose_pose.dir/poseParameters.cpp.o: ../src/openpose/pose/poseParameters.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object src/openpose/pose/CMakeFiles/openpose_pose.dir/poseParameters.cpp.o"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openpose_pose.dir/poseParameters.cpp.o -c /content/openpose/src/openpose/pose/poseParameters.cpp

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseParameters.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openpose_pose.dir/poseParameters.cpp.i"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/openpose/src/openpose/pose/poseParameters.cpp > CMakeFiles/openpose_pose.dir/poseParameters.cpp.i

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseParameters.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openpose_pose.dir/poseParameters.cpp.s"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/openpose/src/openpose/pose/poseParameters.cpp -o CMakeFiles/openpose_pose.dir/poseParameters.cpp.s

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseParametersRender.cpp.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make
src/openpose/pose/CMakeFiles/openpose_pose.dir/poseParametersRender.cpp.o: ../src/openpose/pose/poseParametersRender.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object src/openpose/pose/CMakeFiles/openpose_pose.dir/poseParametersRender.cpp.o"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openpose_pose.dir/poseParametersRender.cpp.o -c /content/openpose/src/openpose/pose/poseParametersRender.cpp

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseParametersRender.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openpose_pose.dir/poseParametersRender.cpp.i"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/openpose/src/openpose/pose/poseParametersRender.cpp > CMakeFiles/openpose_pose.dir/poseParametersRender.cpp.i

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseParametersRender.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openpose_pose.dir/poseParametersRender.cpp.s"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/openpose/src/openpose/pose/poseParametersRender.cpp -o CMakeFiles/openpose_pose.dir/poseParametersRender.cpp.s

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseTracker.cpp.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make
src/openpose/pose/CMakeFiles/openpose_pose.dir/poseTracker.cpp.o: ../src/openpose/pose/poseTracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object src/openpose/pose/CMakeFiles/openpose_pose.dir/poseTracker.cpp.o"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openpose_pose.dir/poseTracker.cpp.o -c /content/openpose/src/openpose/pose/poseTracker.cpp

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseTracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openpose_pose.dir/poseTracker.cpp.i"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/openpose/src/openpose/pose/poseTracker.cpp > CMakeFiles/openpose_pose.dir/poseTracker.cpp.i

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseTracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openpose_pose.dir/poseTracker.cpp.s"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/openpose/src/openpose/pose/poseTracker.cpp -o CMakeFiles/openpose_pose.dir/poseTracker.cpp.s

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseRenderer.cpp.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make
src/openpose/pose/CMakeFiles/openpose_pose.dir/poseRenderer.cpp.o: ../src/openpose/pose/poseRenderer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object src/openpose/pose/CMakeFiles/openpose_pose.dir/poseRenderer.cpp.o"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openpose_pose.dir/poseRenderer.cpp.o -c /content/openpose/src/openpose/pose/poseRenderer.cpp

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseRenderer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openpose_pose.dir/poseRenderer.cpp.i"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/openpose/src/openpose/pose/poseRenderer.cpp > CMakeFiles/openpose_pose.dir/poseRenderer.cpp.i

src/openpose/pose/CMakeFiles/openpose_pose.dir/poseRenderer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openpose_pose.dir/poseRenderer.cpp.s"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/openpose/src/openpose/pose/poseRenderer.cpp -o CMakeFiles/openpose_pose.dir/poseRenderer.cpp.s

src/openpose/pose/CMakeFiles/openpose_pose.dir/renderPose.cpp.o: src/openpose/pose/CMakeFiles/openpose_pose.dir/flags.make
src/openpose/pose/CMakeFiles/openpose_pose.dir/renderPose.cpp.o: ../src/openpose/pose/renderPose.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object src/openpose/pose/CMakeFiles/openpose_pose.dir/renderPose.cpp.o"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openpose_pose.dir/renderPose.cpp.o -c /content/openpose/src/openpose/pose/renderPose.cpp

src/openpose/pose/CMakeFiles/openpose_pose.dir/renderPose.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openpose_pose.dir/renderPose.cpp.i"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /content/openpose/src/openpose/pose/renderPose.cpp > CMakeFiles/openpose_pose.dir/renderPose.cpp.i

src/openpose/pose/CMakeFiles/openpose_pose.dir/renderPose.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openpose_pose.dir/renderPose.cpp.s"
	cd /content/openpose/build/src/openpose/pose && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /content/openpose/src/openpose/pose/renderPose.cpp -o CMakeFiles/openpose_pose.dir/renderPose.cpp.s

# Object files for target openpose_pose
openpose_pose_OBJECTS = \
"CMakeFiles/openpose_pose.dir/defineTemplates.cpp.o" \
"CMakeFiles/openpose_pose.dir/poseCpuRenderer.cpp.o" \
"CMakeFiles/openpose_pose.dir/poseExtractor.cpp.o" \
"CMakeFiles/openpose_pose.dir/poseExtractorCaffe.cpp.o" \
"CMakeFiles/openpose_pose.dir/poseExtractorCaffeStaf.cpp.o" \
"CMakeFiles/openpose_pose.dir/poseExtractorCaffeTaf.cpp.o" \
"CMakeFiles/openpose_pose.dir/poseExtractorNet.cpp.o" \
"CMakeFiles/openpose_pose.dir/poseGpuRenderer.cpp.o" \
"CMakeFiles/openpose_pose.dir/poseParameters.cpp.o" \
"CMakeFiles/openpose_pose.dir/poseParametersRender.cpp.o" \
"CMakeFiles/openpose_pose.dir/poseTracker.cpp.o" \
"CMakeFiles/openpose_pose.dir/poseRenderer.cpp.o" \
"CMakeFiles/openpose_pose.dir/renderPose.cpp.o"

# External object files for target openpose_pose
openpose_pose_EXTERNAL_OBJECTS = \
"/content/openpose/build/src/openpose/pose/CMakeFiles/openpose_pose.dir/openpose_pose_generated_renderPose.cu.o"

src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/defineTemplates.cpp.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/poseCpuRenderer.cpp.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractor.cpp.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffe.cpp.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffeStaf.cpp.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorCaffeTaf.cpp.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/poseExtractorNet.cpp.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/poseGpuRenderer.cpp.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/poseParameters.cpp.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/poseParametersRender.cpp.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/poseTracker.cpp.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/poseRenderer.cpp.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/renderPose.cpp.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/openpose_pose_generated_renderPose.cu.o
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/build.make
src/openpose/pose/libopenpose_pose.so: /usr/local/cuda/lib64/libcudart_static.a
src/openpose/pose/libopenpose_pose.so: /usr/lib/x86_64-linux-gnu/librt.so
src/openpose/pose/libopenpose_pose.so: src/openpose/core/libopenpose_core.so
src/openpose/pose/libopenpose_pose.so: /usr/local/cuda/lib64/libcudart_static.a
src/openpose/pose/libopenpose_pose.so: /usr/lib/x86_64-linux-gnu/librt.so
src/openpose/pose/libopenpose_pose.so: src/openpose/pose/CMakeFiles/openpose_pose.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/content/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Linking CXX shared library libopenpose_pose.so"
	cd /content/openpose/build/src/openpose/pose && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/openpose_pose.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/openpose/pose/CMakeFiles/openpose_pose.dir/build: src/openpose/pose/libopenpose_pose.so

.PHONY : src/openpose/pose/CMakeFiles/openpose_pose.dir/build

src/openpose/pose/CMakeFiles/openpose_pose.dir/clean:
	cd /content/openpose/build/src/openpose/pose && $(CMAKE_COMMAND) -P CMakeFiles/openpose_pose.dir/cmake_clean.cmake
.PHONY : src/openpose/pose/CMakeFiles/openpose_pose.dir/clean

src/openpose/pose/CMakeFiles/openpose_pose.dir/depend: src/openpose/pose/CMakeFiles/openpose_pose.dir/openpose_pose_generated_renderPose.cu.o
	cd /content/openpose/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /content/openpose /content/openpose/src/openpose/pose /content/openpose/build /content/openpose/build/src/openpose/pose /content/openpose/build/src/openpose/pose/CMakeFiles/openpose_pose.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/openpose/pose/CMakeFiles/openpose_pose.dir/depend
