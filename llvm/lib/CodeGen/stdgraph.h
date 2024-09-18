#ifndef STDGRAPH_H
#define STDGRAPH_H

#ifdef __cplusplus
extern "C"{
#endif 


#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>


#define block_t uint64_t

#define BLOCK_INDEX(bit_index)              ((bit_index)/(sizeof(block_t)*8))
#define MASK_INDEX(bit_index)               ((bit_index)%(sizeof(block_t)*8))
#define MASK(bit_index)                     ((block_t)1 << MASK_INDEX(bit_index))
#define TOTAL_BLOCK_NUM(vertex_num)         (int)(BLOCK_INDEX(vertex_num-1)+1)

#define CHECK_BIT(array, index)      (array[BLOCK_INDEX(index)] & MASK(index))
#define SET_BIT(array, index)        array[BLOCK_INDEX(index)] |=  MASK(index)
#define RESET_BIT(array, index)      array[BLOCK_INDEX(index)] &= ~MASK(index)


bool read_graph(
    const char* filename, 
    int graph_size, 
    block_t *edges, 
    int offset_i
);

bool write_graph(
    const char* filename, 
    int graph_size, 
    const block_t *edges, 
    int offset_i
);

bool read_weights(const char* filename, int size, uint8_t weights[]);

bool write_weights(const char* filename, int size, const uint8_t weights[]);

bool is_valid(
    int graph_size, 
    block_t *edges,
    int color_num, 
    block_t *colors
);

int count_edges(
    int graph_size, 
    const block_t *edges, 
    int *edge_counts
);

void print_colors(
    const char *filename, 
    const char *header, 
    int color_num, 
    int graph_size, 
    const block_t *colors
);

int graph_color_greedy(
    int graph_size, 
    const block_t *edges, 
    block_t *colors, 
    int max_color_possible
);

/**
 * @brief randomly color the graph with max_color being the
 * upper bound of colors used.
 * 
 * @param size Size of the graph.
 * @param edges The edge matrix of the graph.
 * @param colors The result color matrix of the graph.
 * @param max_color The upper bound of colors to be used.
 */
void graph_color_random(
    int graph_size, 
    block_t *colors, 
    int max_color
);

void count_conflicts(
    int graph_size, 
    const block_t color[], 
    const block_t *edges, 
    int conflict_count[]
);


#ifdef __cplusplus
}
#endif

#endif
