#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <fcntl.h>
#include <errno.h>
#include <inttypes.h>

#include "stdgraph.h"


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
        for(int j = i+1; j < graph_size; j++)
            if (CHECK_BIT((*edge_mat)[i], j))
                fprintf(fp, "%d %d\n", i, j);
    
    fclose(fp);
    return true;
}


bool read_weights(const char* filename, int graph_size, float weights[]) {
    FILE *fp = fopen(filename, "r");
    
    if(fp == NULL)
        return false;

    memset(weights, 0, graph_size * sizeof(float));

    char buffer[64];
    int vertex = 0;
    while(fgets(buffer, 64, fp) != NULL && vertex < graph_size) {
        buffer[strcspn(buffer, "\n")] = 0;
        weights[vertex] = atof(buffer);
        vertex++;
    }
    
    fclose(fp);
    return true;
}


bool write_weights(const char* filename, int graph_size, const float weights[]) {
    FILE *fp = fopen(filename, "w");
    if(fp == NULL)
        return false;

    for (int i = 0; i < graph_size; i++)
        fprintf(fp, "%f\n", weights[i]);
    
    fclose(fp);
    return true;
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
