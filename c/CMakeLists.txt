cmake_minimum_required(VERSION 3.5)

# Definitions
set(PROJECT_NAME tensor_engine)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")


project(${PROJECT_NAME})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

add_executable(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC 
    .
    ..
)

target_sources(${PROJECT_NAME} PUBLIC 
    main.c
    engine.c
    op_addmm.c
    op_permute.c
)