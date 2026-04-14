/*
 * nqueens_seq.c
 * ─────────────────────────────────────────────────────────────────────────────
 * Problema das N-Rainhas — Brute Force Sequencial
 *
 * Conta TODAS as soluções válidas para o tabuleiro NxN (não para na primeira).
 * Mede o tempo total de execução para referência de speed-up com OpenMP.
 *
 * Compilação:
 *   gcc -O2 -o nqueens_seq nqueens_seq.c
 *
 * Uso:
 *   ./nqueens_seq <N>
 *   ./nqueens_seq 13
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_MAX 20


int board[N_MAX];

long long total_solucoes = 0;

//verifica se da pra por rainha no local
int place(int row, int col) {
    for (int i = 0; i < row; i++) {
        /* mesmo coluna */
        if (board[i] == col)
            return 0;
        /* mesma diagonal */
        if (abs(board[i] - col) == abs(i - row))
            return 0;
    }
    return 1;
}

/* ─────────────────────────────────────────────────────────────────────── */
/*  Backtracking — percorre todas as combinações sem parar na primeira     */
/*                                                                         */
/*  PONTO DE PARALELIZAÇÃO (OpenMP — próxima etapa):                       */
/*  O loop da linha 0 (primeira rainha) será dividido entre threads:       */
/*                                                                         */
/*    #pragma omp parallel for schedule(dynamic, 1)                        */
/*    reduction(+ : total_solucoes)                                        */
/*    for (col = 0; col < n; col++) { ... }                                */
/*                                                                         */
/*  Cada thread recebe uma coluna inicial e explora sua subárvore.         */
/*  schedule(dynamic) é essencial pois subárvores têm tamanhos diferentes. */
/* ─────────────────────────────────────────────────────────────────────── */
void queen(int row, int n) {
    // se todas as rainhas foram postas, soma mais um
    if (row == n) {
        total_solucoes++;
        return;
    }

    /* tenta colocar a rainha em cada coluna da linha atual */
    for (int col = 0; col < n; col++) {
        if (place(row, col)) {
            board[row] = col;
            queen(row + 1, n); /* recursão — próxima linha */
            board[row] = -1;   /* backtrack */
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────── */
/*  Ponto de entrada                                                       */
/* ─────────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <N>\n", argv[0]);
        fprintf(stderr, "Ex : %s 13\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n < 1 || n > N_MAX) {
        fprintf(stderr, "ERRO: N deve estar entre 1 e %d\n", N_MAX);
        return 1;
    }

    /* inicializa o tabuleiro */
    memset(board, -1, sizeof(board));

    printf("================================================\n");
    printf("  N-Rainhas Brute Force — Versao Sequencial\n");
    printf("================================================\n");
    printf("  N = %d  (tabuleiro %dx%d)\n\n", n, n, n);

    /* medição de tempo */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    queen(0, n);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double tempo = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("+-------------------------------+\n");
    printf("| N               : %4d        |\n", n);
    printf("| Solucoes        : %10lld |\n", total_solucoes);
    printf("| Tempo (seq)     : %9.4f s |\n", tempo);
    printf("+-------------------------------+\n");

    return 0;
}