#define _GNU_SOURCE
#pragma GCC target ("sse4")

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <fcntl.h>
#include <unistd.h> 
#include <string.h>
#include <float.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <errno.h>
#include <pthread.h>
#include <math.h>
#include <inttypes.h>

#include "mtBitEA.h"


void graph_color_random (
    int graph_size, 
    block_t *colors, 
    int max_color
) {
    block_t (*color_mat)[][TOTAL_BLOCK_NUM(graph_size)] = 
        (block_t (*)[][TOTAL_BLOCK_NUM(graph_size)])colors;

    int index;
    for(int i = 0; i < graph_size; i++) {
        index = rand()%max_color;
        SET_BIT((*color_mat)[index], i);
    }
}


bool read_graph (
    const char* filename, 
    int graph_size, 
    block_t *edges, 
    int offset_i
) {
    block_t (*edge_mat)[][TOTAL_BLOCK_NUM(graph_size)] = 
        (block_t (*)[][TOTAL_BLOCK_NUM(graph_size)])edges;

    FILE *fp = fopen(filename, "r");
    if(fp == NULL)
        return false;

    memset((*edge_mat), 0, graph_size*TOTAL_BLOCK_NUM(graph_size)*sizeof(block_t));

    char buffer[128];
    char *token, *saveptr;
    int row, column;
    while(fgets(buffer, 128, fp) != NULL) {
        buffer[strcspn(buffer, "\n")] = 0;

        token = strtok_r (buffer, " ", &saveptr);
        if(saveptr[0] == 0) 
            break;
        row = atoi(token) + offset_i;
        token = strtok_r (NULL, " ", &saveptr);
        column = atoi(token) + offset_i;

        SET_BIT((*edge_mat)[row], column);
        SET_BIT((*edge_mat)[column], row);
    }
    
    fclose(fp);
    return true;
}


bool write_graph (
    const char* filename, 
    int graph_size, 
    const block_t *edges, 
    int offset_i
) {
    block_t (*edge_mat)[][TOTAL_BLOCK_NUM(graph_size)] = 
        (block_t (*)[][TOTAL_BLOCK_NUM(graph_size)])edges;

    FILE *fp = fopen(filename, "w");
    if(fp == NULL)
        return false;

    for(int i = 0; i < graph_size; i++)
        for(int j = 0; j < graph_size; j++)
            if (CHECK_BIT((*edge_mat)[i], j))
                fprintf(fp, "%d %d\n", i, j);
    
    fclose(fp);
    return true;
}


bool read_weights(const char* filename, int graph_size, uint8_t weights[]) {
    FILE *fp = fopen(filename, "r");
    
    if(fp == NULL)
        return false;

    memset(weights, 0, graph_size * sizeof(uint8_t));

    char buffer[64];
    int vertex = 0;
    while(fgets(buffer, 64, fp) != NULL && vertex < graph_size) {
        buffer[strcspn(buffer, "\n")] = 0;
        weights[vertex] = atoi(buffer);
        vertex++;
    }
    
    fclose(fp);
    return true;
}


bool write_weights(const char* filename, int graph_size, const uint8_t weights[]) {
    FILE *fp = fopen(filename, "w");
    if(fp == NULL)
        return false;

    for (int i = 0; i < graph_size; i++)
        fprintf(fp, "%d\n", weights[i]);
    
    fclose(fp);
    return true;
}


bool is_valid(
    int graph_size, 
    block_t *edges,
    int color_num, 
    block_t *colors
) {
    block_t (*edge_mat)[][TOTAL_BLOCK_NUM(graph_size)] = 
        (block_t (*)[][TOTAL_BLOCK_NUM(graph_size)])edges;
    
    block_t (*color_mat)[][TOTAL_BLOCK_NUM(graph_size)] = 
        (block_t (*)[][TOTAL_BLOCK_NUM(graph_size)])colors;

    // Iterate through vertices.
    int i, j, k, i_block;
    block_t i_mask;
    bool vertex_is_colored, error_flag = false, over_colored;
    for(i = 0; i < graph_size; i++) {
        vertex_is_colored = false;
        over_colored = false;
        i_block = BLOCK_INDEX(i);
        i_mask = MASK(i);

        // Iterate through colors and look for the vertex.
        for(j = 0; j < color_num; j++){
            if(((*color_mat)[j][i_block] & i_mask)) {
                if(!vertex_is_colored) {
                    vertex_is_colored = true;

                } else {
                    over_colored = true;
                }

                for(k = i + 1; k < graph_size; k++) { // Through every vertex after i in color j.
                    if(CHECK_BIT((*color_mat)[j], k) && ((*edge_mat)[k][i_block] & i_mask)) {
                        // The two vertices have the same color.
                        printf("The vertices %d and %d are connected and have the same color %d.\n", i, k, j);
                        error_flag = true;
                    }
                }
            }
        }

        // Check if the vertex had more then one color.
        if(!vertex_is_colored) {
            printf("The vertex %d has no color.\n", i);
            error_flag = true;
        }

        // Check if the vertex had more then one color.
        if(over_colored) {
            printf("The vertex %d has more than one color.\n", i);
            error_flag = true;
        }
    }

    return !error_flag;
}


int count_edges(
    int graph_size, 
    const block_t *edges, 
    int *edge_counts
) {
    const block_t (*edge_mat)[][TOTAL_BLOCK_NUM(graph_size)] = 
        (const block_t (*)[][TOTAL_BLOCK_NUM(graph_size)])edges;

    memset(edge_counts, 0, graph_size*sizeof(int));

    int i, j, total = 0;
    for(i = 0; i < graph_size; i++) {
        for(j = 0; j < TOTAL_BLOCK_NUM(graph_size); j++)
            edge_counts[i] += __builtin_popcountl((*edge_mat)[i][j]);
        total += edge_counts[i];
    }

    return total;
}


void print_colors(
    const char *filename, 
    const char *header, 
    int color_num, 
    int graph_size, 
    const block_t *colors
) {
    const block_t (*color_mat)[][TOTAL_BLOCK_NUM(graph_size)] = 
        (const block_t (*)[][TOTAL_BLOCK_NUM(graph_size)])colors;

    FILE* fresults;
    fresults = fopen(filename, "w");

    if(!fresults) {
        printf("%s\ncould not print results, aborting ...\n", strerror(errno));
        return;
    }

    fprintf(fresults, "%s\n\n", header);

    for(int i = 0; i < color_num; i++)
        for(int j = 0; j < graph_size; j++)
            if(CHECK_BIT((*color_mat)[i], j)) 
                fprintf(fresults, "%d %d\n", i, j);

    fclose(fresults);
}


bool exists(int* arr, int len, int target) {
    for(int i = 0; i < len; i++)
        if(target == arr[i]) return true;

    return false;
}


int graph_color_greedy(
    int graph_size, 
    const block_t *edges, 
    block_t *colors, 
    int max_color_possible
) {
    const block_t (*edge_mat)[][TOTAL_BLOCK_NUM(graph_size)] = 
        (const block_t (*)[][TOTAL_BLOCK_NUM(graph_size)])edges;
    
    block_t (*color_mat)[][TOTAL_BLOCK_NUM(graph_size)] = 
        (block_t (*)[][TOTAL_BLOCK_NUM(graph_size)])colors;

    // Go through the queue and color each vertex.
    int prob_queue[graph_size];
    block_t adjacent_colors[TOTAL_BLOCK_NUM(max_color_possible)];
    int max_color = 0, current_vert;
    int i, j, k;
    for(i = 0; i < graph_size; i++) {
        // Get a new random vertex.
        do { prob_queue[i] = rand()%graph_size; } while(exists(prob_queue, i, prob_queue[i]));
        current_vert = prob_queue[i];

        // Initialize the temporary data.
        memset(adjacent_colors, 0, (TOTAL_BLOCK_NUM(max_color_possible))*sizeof(block_t));
        for(j = 0; j < TOTAL_BLOCK_NUM(graph_size); j++)
            for(k = 0; k < max_color_possible; k++)
                if(((*edge_mat)[current_vert][j] & (*color_mat)[k][j]))
                    SET_BIT(adjacent_colors, k);

        // Find the first unused color (starting from 0) and assign this vertex to it.
        for(j = 0; j < max_color_possible; j++) {
            if(!CHECK_BIT(adjacent_colors, j)) {
                SET_BIT((*color_mat)[j], current_vert);
                if(max_color < j) 
                    max_color = j;
                break;
            }
        }
    }

    return max_color + 1;
}


void count_conflicts(
    int graph_size, 
    const block_t color[], 
    const block_t *edges, 
    int conflict_count[]
) {
    const block_t (*edge_mat)[][TOTAL_BLOCK_NUM(graph_size)] = 
        (const block_t (*)[][TOTAL_BLOCK_NUM(graph_size)])edges;

    int i, j;
    for(i = 0; i < graph_size; i++) {
        conflict_count[i] = 0;
        if(CHECK_BIT(color, i)) {
            for(j = 0; j < TOTAL_BLOCK_NUM(graph_size); j++)
                conflict_count[i] += __builtin_popcountl(color[j] & (*edge_mat)[i][j]);
        }
    }
}


void graph_color_MBGA(
    const mbga_graph_t *graph,
    int population_size,
    int base_color_count, 
    int max_gen_num, 
    int thread_num,
    mbga_individual_t *solution
) {
    mbga_individual_t *population = malloc(population_size * sizeof(mbga_individual_t)); 
    for (int i = 0; i < population_size; i++) {
        population[i].color_mat = calloc(base_color_count * TOTAL_BLOCK_NUM((size_t)graph->size), sizeof(block_t));
        population[i].uncolored = base_color_count;
        population[i].color_num = base_color_count;
        population[i].fitness = __INT_MAX__;

        graph_color_random(graph->size, population[i].color_mat, base_color_count);
    }

    // Create and initialize the list of used parents.
    block_t used_parents[(TOTAL_BLOCK_NUM(population_size))];
    memset(used_parents, 0, (TOTAL_BLOCK_NUM(population_size))*sizeof(block_t));

    // Initialize thread parameters.
    int target_color = base_color_count;
    int gen_num = max_gen_num;

    pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
    struct crossover_param_s temp_param[thread_num]; 
    for (int i = 0; i < thread_num; i++) {
        temp_param[i] = (struct crossover_param_s){
            graph,
            &target_color,
            &gen_num,
            &mut,
            population_size,
            population,
            malloc(sizeof(mbga_individual_t)),
            used_parents
        };

        temp_param[i].child_buffer->fitness = -1;
        temp_param[i].child_buffer->color_mat = malloc(target_color * TOTAL_BLOCK_NUM(graph->size) * sizeof(block_t));
    }
    
    // Launch the generator threads.
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_t thread_id[thread_num];
    for(int i = 0; i < thread_num; i++)
        pthread_create(&thread_id[i], &attr, generator_thread, &temp_param[i]);

    // Wait for all threads to return. 
    for(int i = 0; i < thread_num; i++) {
        pthread_join(thread_id[i], NULL);
        free(temp_param[i].child_buffer->color_mat);
        free(temp_param[i].child_buffer);
    }

    // Find the best solution in the population.
    int best_i = 0;
    for(int i = 0; i < population_size; i++)
        if (population[i].fitness < population[best_i].fitness || 
            (population[i].fitness == population[best_i].fitness && population[i].color_num <= population[best_i].color_num))
            best_i = i;

    // Return the best solution
    *solution = population[best_i];
    population[best_i].color_mat = malloc(1);

    // Free allocated space.
    pthread_mutex_destroy(&mut);
    for(int i = 0; i < population_size; i++)
        free(population[i].color_mat);
    free(population);
}

int get_rand_color(int max_color_num, int colors_used, block_t used_color_list[]) {
    // There are no available colors.
    if(colors_used >= max_color_num) {
        return -1;

    // There are only 2 colors available, search for them linearly.
    } else if(colors_used > max_color_num*0.9) {
        for(int i = 0; i < max_color_num; i++) {
            if(!CHECK_BIT(used_color_list, i)) {
                SET_BIT(used_color_list, i);
                return i;
            }
        }
    }

    // Randomly try to select an available color.
    int temp;
    while(1) {
        temp = rand()%max_color_num;
        if(!CHECK_BIT(used_color_list, temp)) {
            SET_BIT(used_color_list, temp);
            return temp;
        }
    }
}

int fix_conflicts(
    const mbga_graph_t *graph,
    int conflict_count[],
    block_t color[],
    block_t pool[]
) {
    block_t (*edge_matrix)[][TOTAL_BLOCK_NUM(graph->size)] = 
        (block_t (*)[][TOTAL_BLOCK_NUM(graph->size)])graph->edge_mat;

    block_t conflict_array[TOTAL_BLOCK_NUM(graph->size)];

    // Keep removing problematic vertices until all conflicts are gone.
    int i, worst_vert = 0, vert_block, pool_count = 0;
    block_t vert_mask;
    while(1) {
        // Find the vertex with the most conflicts.
        for(i = 0; i < graph->size; i++) {
            if (conflict_count[worst_vert] < conflict_count[i] ||
                 (conflict_count[worst_vert] == conflict_count[i] && graph->weights[worst_vert] >= graph->weights[i])
            ) {
                worst_vert = i;
            }
        }

        for(i = 0; i < TOTAL_BLOCK_NUM(graph->size); i++)
            conflict_array[i] = (*edge_matrix)[worst_vert][i] & color[i];

        if(conflict_count[worst_vert] <= 0)
            return pool_count;

        // Update other conflict counters.
        vert_mask = MASK(worst_vert);
        vert_block = BLOCK_INDEX(worst_vert);
        for(i = 0; i < graph->size; i++)
            if(CHECK_BIT(conflict_array, i) && conflict_count[i] > 0)
                conflict_count[i]--;

        // Remove the vertex.
        color[vert_block] &= ~vert_mask;
        pool[vert_block] |= vert_mask;

        conflict_count[worst_vert] = 0;

        pool_count++;
    }
}

void crossover(
    const mbga_graph_t *graph,
    const block_t *parent_color[2],
    block_t child_color[],
    block_t pool[],
    block_t used_vertex_list[],
    int *pool_count,
    int *used_count
) {
    // Merge the two colors
    for(int i = 0; i < (TOTAL_BLOCK_NUM(graph->size)); i++) {
        child_color[i] = ((parent_color[0][i] | parent_color[1][i]) & ~(used_vertex_list[i])) | pool[i];
        used_vertex_list[i] |= child_color[i];
        *used_count += __builtin_popcountl(child_color[i]);
    }

    memset(pool, 0, (TOTAL_BLOCK_NUM(graph->size))*sizeof(block_t));

    // Count conflicts.
    int conflict_counts[graph->size];
    count_conflicts(
        graph->size,
        child_color,
        graph->edge_mat,
        conflict_counts
    );

    // Fix the conflicts.
    *pool_count = fix_conflicts(
        graph,
        conflict_counts,
        child_color,
        pool
    );
}

int revisit(
    const mbga_graph_t *graph,
    mbga_individual_t *child,
    int max_color,
    bool simple_flag,
    block_t pool[]
) {
    block_t (*edge_matrix)[][TOTAL_BLOCK_NUM(graph->size)] = 
        (block_t (*)[][TOTAL_BLOCK_NUM(graph->size)])graph->edge_mat;
        
    block_t (*child_color_mat)[][TOTAL_BLOCK_NUM(graph->size)] = 
        (block_t (*)[][TOTAL_BLOCK_NUM(graph->size)])child->color_mat;

    int i, j, k, i_block;
    block_t i_mask;
    block_t conflict_array[TOTAL_BLOCK_NUM(graph->size)];
    int add_to_pool = 0;
    int conflict_id, conflict_block, competition, conflict_count;
    bool place_flag;

    // Search back and try placing vertices from the pool in the colors.
    for(i = 0; i < graph->size; i++) {
        i_block = BLOCK_INDEX(i);
        i_mask = MASK(i);

        // Check if the vertex is in the pool.
        if(pool[i_block] & i_mask) {
            // Loop through every color.
            for(j = 0; j < max_color; j++) {  
                // Count conflicts and calculate competition
                conflict_count = 0;
                for(k = 0; k < TOTAL_BLOCK_NUM(graph->size); k++) {
                    conflict_array[k] = (*edge_matrix)[i][k] & (*child_color_mat)[j][k];
                    conflict_count += __builtin_popcountl(conflict_array[k]);
                }

                // /**
                //  * If the total competition is smaller than the weight
                //  * of the vertex in question, move all the conflicts to the 
                //  * pool, and place the vertex in the color.
                // */
                place_flag = true;
                if (simple_flag && conflict_count == 1) {
                    conflict_id = 0;
                    conflict_block = 0;
                    for(k = 0; k < TOTAL_BLOCK_NUM(graph->size); k++) {
                        if(conflict_array[k]) {
                            conflict_id = sizeof(block_t)*8*(k + 1) - 1 - __builtin_clzl(conflict_array[k]);
                            conflict_block = k;
                            break;
                        }
                    }

                    if(graph->weights[i] < graph->weights[conflict_id]) {
                        place_flag = false;

                    } else {
                        (*child_color_mat)[j][conflict_block] &= ~conflict_array[conflict_block];
                        pool[conflict_block] |= conflict_array[conflict_block];
                        add_to_pool++;
                    }

                } else if (!simple_flag && conflict_count > 0) {
                    competition = 0;
                    for(k = 0; k < graph->size; k++)
                        if(CHECK_BIT(conflict_array, k))
                            competition += graph->weights[k];

                    if (competition > graph->weights[i]) {
                        place_flag = false;

                    } else {
                        for(k = 0; k < TOTAL_BLOCK_NUM(graph->size); k++) {
                            (*child_color_mat)[j][k] &= ~conflict_array[k];
                            pool[k] |= conflict_array[k];
                        }

                        add_to_pool += conflict_count;
                    }

                } else if (conflict_count != 0) {
                    place_flag = false;
                }

                if (place_flag) {
                    (*child_color_mat)[j][i_block] |= i_mask;
                    pool[i_block] &= ~i_mask;
                    add_to_pool--;

                    break;
                }
            }
        }
    }

    return add_to_pool;
}

void generate_child (
    const mbga_graph_t *graph,
    const mbga_individual_t *parent1,
    const mbga_individual_t *parent2,
    block_t *used_color_list,
    block_t *used_vertex_list,
    block_t *pool,
    int target_color_count,
    mbga_individual_t *child
) {
    block_t (*child_color_mat)[][TOTAL_BLOCK_NUM(graph->size)] = 
        (block_t (*)[][TOTAL_BLOCK_NUM(graph->size)])child->color_mat;

    block_t (*used_color_list_p)[][TOTAL_BLOCK_NUM(target_color_count)] = 
        (block_t (*)[][TOTAL_BLOCK_NUM(target_color_count)])used_color_list;

    int color1, color2;
    const block_t *chosen_parent_colors[2];
    int pool_count = 0, used_count = 0;
    for(
        child->color_num = 0; 
        child->color_num < target_color_count; 
        child->color_num++
    ) {
        color1 = get_rand_color(parent1->color_num, child->color_num, (*used_color_list_p)[0]);
        color2 = get_rand_color(parent2->color_num, child->color_num, (*used_color_list_p)[1]);
        chosen_parent_colors[0] = &parent1->color_mat[color1*TOTAL_BLOCK_NUM(graph->size)];
        chosen_parent_colors[1] = &parent2->color_mat[color2*TOTAL_BLOCK_NUM(graph->size)];

        crossover(
            graph,
            chosen_parent_colors,
            (*child_color_mat)[child->color_num],
            pool,
            used_vertex_list,
            &pool_count,
            &used_count
        );

        pool_count += revisit(
            graph,
            child,
            child->color_num,
            true,
            pool
        );

        if(pool_count == 0 && used_count == graph->size)
            break;
    }

    // If there are unseen vertices, drop them in the pool.
    for(int i = 0; i < (TOTAL_BLOCK_NUM(graph->size)); i++)
        pool[i] |= ~used_vertex_list[i];
    pool[TOTAL_BLOCK_NUM(graph->size)-1] &= 
        ((~((block_t)0)) >> (TOTAL_BLOCK_NUM(graph->size)*sizeof(block_t)*8 - graph->size));

    revisit(
        graph,
        child,
        target_color_count,
        false,
        pool
    );

    // Calculate the fitness while randomly allocating the remaining vertices in the colors.
    int temp_block, color_num;
    block_t temp_mask;

    child->uncolored = 0;
    child->fitness = 0;
    
    for(int i = 0; i < graph->size; i++) {
        temp_block = BLOCK_INDEX(i);
        temp_mask = MASK(i);
        if(pool[temp_block] & temp_mask) {
            color_num = rand()%target_color_count;
            (*child_color_mat)[color_num][temp_block] |= temp_mask;

            if(color_num + 1 > child->color_num)
                child->color_num = color_num + 1;

            child->fitness += graph->weights[i];
            child->uncolored++;
        }
    }
}

void* generator_thread(void *param) {
    const mbga_graph_t *graph       = ((struct crossover_param_s*)param)->graph;
    int *target_color_count         = ((struct crossover_param_s*)param)->target_color_count;
    int *gen_num                    = ((struct crossover_param_s*)param)->child_count;
    pthread_mutex_t* mut            = ((struct crossover_param_s*)param)->mut;
    int population_size             = ((struct crossover_param_s*)param)->population_size;
    mbga_individual_t *population   = ((struct crossover_param_s*)param)->population;
    mbga_individual_t *child   = ((struct crossover_param_s*)param)->child_buffer;
    block_t *used_parents       = ((struct crossover_param_s*)param)->used_parents;


    // list of used colors in the parents.
    block_t *used_color_list = 
        calloc(2, (TOTAL_BLOCK_NUM(*target_color_count))*sizeof(block_t));

    // list of used vertices in the parents.
    block_t *used_vertex_list = 
        calloc(1, (TOTAL_BLOCK_NUM(graph->size))*sizeof(block_t));

    // Pool.
    block_t *pool = 
        calloc(1, (TOTAL_BLOCK_NUM(graph->size))*sizeof(block_t));


    int temp_target_color;
    int parent1, parent2, bad_parent = -1;
    block_t *temp_ptr;
    while(1) {

        pthread_mutex_lock(mut);

        if ((*gen_num) <= 0) {
            pthread_mutex_unlock(mut);
            break;
        } else {
            (*gen_num)--;
        }

        // Pick 2 random parents
        do { parent1 = rand()%population_size; } while (CHECK_BIT(used_parents, parent1));
        SET_BIT(used_parents, parent1);
        do { parent2 = rand()%population_size; } while (CHECK_BIT(used_parents, parent2));
        SET_BIT(used_parents, parent2);

        // Make the target harder if it was found.
        if(child->fitness == 0)
            (*target_color_count) = child->color_num-1;
        temp_target_color = *target_color_count;

        if (bad_parent != -1) {
            // Replace a bad parent.
            if(child->color_num <= population[bad_parent].color_num && child->fitness <= population[bad_parent].fitness) {
                temp_ptr = population[bad_parent].color_mat;
                memcpy(&population[bad_parent], child, sizeof(mbga_individual_t));
                child->color_mat = temp_ptr;
            }

            RESET_BIT(used_parents, parent1);
            RESET_BIT(used_parents, parent2);
        }

        pthread_mutex_unlock(mut);


        memset(used_color_list, 0, 2*(TOTAL_BLOCK_NUM(temp_target_color))*sizeof(block_t));
        memset(used_vertex_list,0,   (TOTAL_BLOCK_NUM(graph->size))*sizeof(block_t));
        memset(pool,            0,   (TOTAL_BLOCK_NUM(graph->size))*sizeof(block_t));


        // Generate a child
        generate_child (
            graph,
            &population[parent1], 
            &population[parent2], 
            used_color_list,
            used_vertex_list,
            pool,
            temp_target_color,
            child
        );


        if(population[parent1].fitness <= population[parent2].fitness && population[parent1].color_num <= population[parent2].color_num)
            bad_parent = parent2;
        else
            bad_parent = parent1;
    }

    free(used_color_list);
    free(used_vertex_list);
    free(pool);

    return NULL;
}


bool validate_colors(
    mbga_graph_t *graph,
    mbga_individual_t *indiv
) {
    block_t (*edge_mat)[][TOTAL_BLOCK_NUM(graph->size)] = 
        (block_t (*)[][TOTAL_BLOCK_NUM(graph->size)])graph->edge_mat;

    block_t (*color_mat)[][TOTAL_BLOCK_NUM(graph->size)] = 
        (block_t (*)[][TOTAL_BLOCK_NUM(indiv->color_num)])indiv->color_mat;

    block_t pool[TOTAL_BLOCK_NUM(graph->size)];
    memset(pool, 0, TOTAL_BLOCK_NUM(graph->size)*sizeof(block_t));

    int confict_count[graph->size];
    int calc_fitness = 0;

    for (int i = 0; i < indiv->color_num; i++) {
        count_conflicts(
            graph->size, 
            (*color_mat)[i], 
            graph->edge_mat,
            confict_count
        );

        fix_conflicts(
            graph,
            confict_count,
            (*color_mat)[i],
            pool
        );
    }

    for (int i = 0; i < graph->size; i++) {
        if(CHECK_BIT(pool, i)) {
            calc_fitness += graph->weights[i];
        }
    }

    if(calc_fitness != indiv->fitness) {
        printf("Fitness does not match, given %d vs actual %d.\n", indiv->fitness, calc_fitness);
    }

    int vertex_counts[graph->size];
    memset(vertex_counts, 0, graph->size*sizeof(int));

    block_t conflict_array[TOTAL_BLOCK_NUM(graph->size)];
    memset(conflict_array, 0, TOTAL_BLOCK_NUM(graph->size)*sizeof(block_t));

    // Iterate through vertices.
    int i, j, k;
    bool error_flag = false;
    for(i = 0; i < indiv->color_num; i++){
        for(j = 0; j < graph->size; j++) {
            if(CHECK_BIT((*color_mat)[i], j)) {
                for (k = 0; k < TOTAL_BLOCK_NUM(graph->size); k++)
                    conflict_array[k] = (*color_mat)[i][k] & (*edge_mat)[j][k];

                for(k = i + 1; k < graph->size; k++) { // Through every vertex after i in color j.
                    if(CHECK_BIT(conflict_array, k)) {
                        // The two vertices have the same color.
                        printf("The vertices %d and %d are connected and have the same color %d.\n", j, k, i);
                        error_flag = true;
                    }
                }

                vertex_counts[j]++;
            }
        }
    }

    for(i = 0; i < graph->size; i++) {
        if(vertex_counts[i] > 1) {
            printf("The vertex %d was repeated %d times.\n", i, vertex_counts[i]);
            error_flag = true;
        } else if (CHECK_BIT(pool, i) && (vertex_counts[i] == 1)) {
            printf("The vertex %d exists both in the pool and in the colors.\n", i);
            error_flag = true;
        } else if (!CHECK_BIT(pool, i) && (vertex_counts[i] == 0)) {
            printf("The vertex %d was not found.\n", i);
            error_flag = true;
        }
    }

    return !error_flag;
}
