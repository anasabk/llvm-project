#ifndef MBGA_H
#define MBGA_H


#include <inttypes.h>
#include <stdlib.h>
#include <stdbool.h>

#include "stdgraph.h"


typedef struct {
    block_t *color_mat;
    int color_num;
    int fitness;
    int uncolored;
} mbga_individual_t;

typedef struct {
    const block_t *edge_mat;
    const uint8_t *weights;
    int size;
} mbga_graph_t;

struct crossover_param_s {
    const mbga_graph_t *graph;
    int *target_color_count;
    int *child_count;
    pthread_mutex_t* mut;
    int population_size;
    mbga_individual_t *population;
    mbga_individual_t *child_buffer;
    block_t *used_parents;
};



/**
 * @brief Color the graph using the MBGA algorithm.
 * 
 * @param graph A graph instance.
 * @param population_size Size of the population.
 * @param target_color_num The maximum number of colors allowed.
 * @param max_gen_num The maximum number of iterations.
 * @param thread_num The number of parallel threads.
 * @param best_solution_time Output pointer to the time it took to find the last solution.
 * @param best_iteration Output pointer to the number of iterations it took to find the last solution.
 * @param solution Output pointer to the resulting solution instance.
 */
void graph_color_MBGA (
    const mbga_graph_t *graph,
    int population_size,
    int target_color_num, 
    int max_gen_num, 
    int thread_num,
    mbga_individual_t *solution
);


/**
 * @brief Get a random color that was not used previously in the 
 * used_color_list. When a color is returned, it is added to the 
 * used_color_list.
 * 
 * @param size Max number of colors.
 * @param colors_used Number of colors used.
 * @param used_color_list List of used colors.
 * @return If an unused color is found, return it. If all colors are 
 * used, return -1.
 */
int get_rand_color(int max_color_num, int colors_used, block_t used_color_list[]);


/**
 * @brief Do a crossover operation between two parent colors to produce a 
 * new child color. Vertices are checked if they were used previously 
 * through used_vertex_list before being added to the child color.
 * 
 * @param graph A graph instance.
 * @param parent_color Array of pointers to two parents.
 * @param child_color Pointer to the child color.
 * @param pool The pool containing unallocated vertices.
 * @param pool_count Total number of vertices in the pool.
 * @param used_vertex_list List of used vertices.
 * @param used_vertex_count Pointer to number of used vertices.
 */
void crossover(
    const mbga_graph_t *graph,
    const block_t *parent_color[2],
    block_t child_color[],
    block_t pool[],
    block_t used_vertex_list[],
    int *pool_count,
    int *used_count
);


/**
 * @brief Fix the conflicts inside a color.
 * 
 * @param graph A graph instance.
 * @param conflict_count An array of conflict counts for each vertex.
 * @param color Pointer to the color.
 * @param pool The pool containing unallocated vertices.
 * @param pool_count Total number of vertices in the pool.
 */
int fix_conflicts(
    const mbga_graph_t *graph,
    int conflict_count[],
    block_t color[],
    block_t pool[]
);


/**
 * @brief Revisit previous colors of child until max_color and
 * try to place vertices from the pool to those colors.
 * 
 * @param graph A graph instance.
 * @param child A solution instance.
 * @param max_color A the maximum number of colors to scan starting from 0.
 * @param simple_flag When true, swaps are done one with 1 conflicts.
 * @param pool The pool containing unallocated vertices.
 * @param pool_count Total number of vertices in the pool.
 */
int revisit(
    const mbga_graph_t *graph,
    mbga_individual_t *child,
    int max_color,
    bool simple_flag,
    block_t pool[]
);


/**
 * @brief Generates a new child solution by combining parent1 and parent2.
 * 
 * @param graph A graph instance.
 * @param parent1 A solution instance as the first parent.
 * @param parent2 A solution instance as the second parent.
 * @param target_color_count Target number of colors.
 * @param child Pointer to the output of the child solution.
 */
void generate_child (
    const mbga_graph_t *graph,
    const mbga_individual_t *parent1,
    const mbga_individual_t *parent2, 
    block_t *used_color_list,
    block_t *used_vertex_list,
    block_t *pool,
    int target_color_count,
    mbga_individual_t *child
);


bool validate_colors(
    mbga_graph_t *graph,
    mbga_individual_t *indiv
);


void* generator_thread(void *param);


#endif