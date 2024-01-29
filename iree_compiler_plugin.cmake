# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(yaml-cpp 0.7.0 REQUIRED)
include_directories(${yaml-cpp_INCLUDE_DIRS})

# find_package (Python3 3.10 REQUIRED)
# find_package(flexml_metadata CONFIG REQUIRED HINTS "${Python3_SITELIB}")
# message("${Python3_SITELIB}")
# include_directories("flexml_metadata-0.0.1/libraries/abstraction/include/")
# include_directories(/proj/rdi/staff/jornt/miniconda3/envs/vaiml/lib/python3.10/site-packages/)
# include_directories(/proj/rdi/staff/jornt/miniconda3/envs/vaiml/lib/python3.10/site-packages/flexml_metadata/include/)


set(IREE_AMD_AIE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
include_directories("${IREE_AMD_AIE_SOURCE_DIR}/flexml_metadata-0.0.1/libraries/yamlUtils/include")
include_directories("${IREE_AMD_AIE_SOURCE_DIR}/flexml_metadata-0.0.1/libraries/logging/include")

external_cc_library(
  PACKAGE
    flexml_metadata
  NAME
    flexml_metadata
  ROOT
    "${IREE_AMD_AIE_SOURCE_DIR}/flexml_metadata-0.0.1/"
  INCLUDES
    "${IREE_AMD_AIE_SOURCE_DIR}/flexml_metadata-0.0.1/libraries/yamlUtils/include"
    "${IREE_AMD_AIE_SOURCE_DIR}/flexml_metadata-0.0.1/libraries/logging/include"
    "${IREE_AMD_AIE_SOURCE_DIR}/flexml_metadata-0.0.1/libraries/utils/include"
    "${IREE_AMD_AIE_SOURCE_DIR}/flexml_metadata-0.0.1/libraries/abstraction/include/"
    "${yaml-cpp_INCLUDE_DIRS}"
  SRCS
    "libraries/yamlUtils/src/FileLister.cpp"
    "libraries/yamlUtils/src/FileLoader.cpp"
    "libraries/yamlUtils/src/YamlLoader.cpp"
    "libraries/abstraction/src/AcceptedValues.cpp"
    "libraries/abstraction/src/AttributeHelpers.cpp"
    "libraries/abstraction/src/BaseAbstraction.cpp"
    "libraries/abstraction/src/BaseAbstractionUtils.cpp"
    "libraries/abstraction/src/DataTypes.cpp"
    "libraries/abstraction/src/GenericParameter.cpp"
    "libraries/abstraction/src/Kernel.cpp"
    "libraries/abstraction/src/TemplatedGraph.cpp"
    "libraries/abstraction/src/YamlHelpers.cpp"
    "libraries/abstraction/src/YamlTranslationHelpers.cpp"
    "libraries/abstraction/src/YamlTranslationHelpers.h"
    "libraries/abstraction/src/expressions/AbstractionEvaluator.cpp"
    "libraries/abstraction/src/expressions/Ast.cpp"
    "libraries/abstraction/src/expressions/Parser.cpp"
  HDRS
    "libraries/logging/include/flexmlMetadata/logging/Logging.h"
    "libraries/yamlUtils/include/flexmlMetadata/yamlUtils/FileLister.h"
    "libraries/yamlUtils/include/flexmlMetadata/yamlUtils/FileLoader.h"
    "libraries/yamlUtils/include/flexmlMetadata/yamlUtils/YamlLoader.h"
    "libraries/utils/include/flexmlMetadata/utils/DatabaseManagement.h"
    "libraries/abstraction/include/flexmlMetadata/abstraction/AcceptedValues.h"
    "libraries/abstraction/include/flexmlMetadata/abstraction/AttributeCollectors.h"
    "libraries/abstraction/include/flexmlMetadata/abstraction/AttributeHelpers.h"
    "libraries/abstraction/include/flexmlMetadata/abstraction/BaseAbstraction.h"
    "libraries/abstraction/include/flexmlMetadata/abstraction/BaseAbstractionUtils.h"
    "libraries/abstraction/include/flexmlMetadata/abstraction/DataTypes.h"
    "libraries/abstraction/include/flexmlMetadata/abstraction/GenericParameter.h"
    "libraries/abstraction/include/flexmlMetadata/abstraction/Kernel.h"
    "libraries/abstraction/include/flexmlMetadata/abstraction/Macros.h"
    "libraries/abstraction/include/flexmlMetadata/abstraction/NestedUnorderedMap.h"
    "libraries/abstraction/include/flexmlMetadata/abstraction/TemplatedGraph.h"
    "libraries/abstraction/include/flexmlMetadata/abstraction/TypeHelpers.h"
    "libraries/abstraction/include/flexmlMetadata/abstraction/YamlHelpers.h"
    "libraries/abstraction/include/flexmlMetadata/expressions/AbstractionEvaluator.h"
    "libraries/abstraction/include/flexmlMetadata/expressions/Ast.h"
    "libraries/abstraction/include/flexmlMetadata/expressions/Concepts.h"
    "libraries/abstraction/include/flexmlMetadata/expressions/Evaluator.h"
    "libraries/abstraction/include/flexmlMetadata/expressions/KernelInputCollector.h"
    "libraries/abstraction/include/flexmlMetadata/expressions/Parser.h"
  COPTS
    "-frtti"
    "-std=c++23"
    "-fexceptions"
  DEPS
    yaml-cpp
  PUBLIC
)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/compiler/plugins/target/AMD-AIE target/AMD-AIE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tools/plugins AMD-AIE/tools)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests/samples AMD-AIE/tests/samples)
