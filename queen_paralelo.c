/*
 * nqueens_omp.c
 * ─────────────────────────────────────────────────────────────────────────────
 * Problema das N-Rainhas — Brute Force Paralelo com OpenMP
 *
 * Conta TODAS as soluções válidas para o tabuleiro NxN.
 * Mede o tempo total de execução para cálculo de speed-up.
 *
 * Correções em relação à versão com bug:
 *   1. board[] deixou de ser global — cada thread tem seu próprio board local
 *      (evita race condition: threads sobrescrevendo o estado umas das outras)
 *   2. #pragma omp aplicado APENAS nas duas primeiras linhas da recursão,
 *      gerando N² tarefas independentes — melhor granularidade e balanceamento
 *
 * Compilação:
 *   gcc -O2 -fopenmp -o nqueens_omp nqueens_omp.c
 *
 * Uso:
 *   ./nqueens_omp <N> <num_threads>
 *   ./nqueens_omp 15 4
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define N_MAX 20

/* ─────────────────────────────────────────────────────────────────────── */
/*  Verifica conflitos usando o board LOCAL da thread (thread-safe)        */
/* ─────────────────────────────────────────────────────────────────────── */
int place(int board_local[], int row, int col) {
    for (int i = 0; i < row; i++) {
        if (board_local[i] == col)                       /* mesma coluna   */
            return 0;
        if (abs(board_local[i] - col) == abs(i - row))  /* mesma diagonal */
            return 0;
    }
    return 1;
}

/* ─────────────────────────────────────────────────────────────────────── */
/*  Backtracking recursivo — opera apenas no board LOCAL da thread         */
/*  Sem nenhuma diretiva OpenMP aqui: totalmente thread-safe               */
/* ─────────────────────────────────────────────────────────────────────── */
void queen(int board_local[], int row, int n, long long *count) {
    if (row == n) {          /* caso base: solução completa encontrada */
        (*count)++;
        return;
    }
    for (int col = 0; col < n; col++) {
        if (place(board_local, row, col)) {
            board_local[row] = col;
            queen(board_local, row + 1, n, count);
            board_local[row] = -1;   /* backtrack */
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────── */
/*  Ponto de entrada                                                       */
/* ─────────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <N> <num_threads>\n", argv[0]);
        fprintf(stderr, "Ex : %s 15 4\n", argv[0]);
        return 1;
    }

    int n           = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    if (n < 1 || n > N_MAX) {
        fprintf(stderr, "ERRO: N deve estar entre 1 e %d\n", N_MAX);
        return 1;
    }
    if (num_threads < 1) {
        fprintf(stderr, "ERRO: numero de threads deve ser >= 1\n");
        return 1;
    }

    omp_set_num_threads(num_threads);

    printf("================================================\n");
    printf("  N-Rainhas Brute Force — Versao Paralela OMP\n");
    printf("================================================\n");
    printf("  N           = %d  (tabuleiro %dx%d)\n", n, n, n);
    printf("  Threads     = %d\n\n", num_threads);

    long long total_solucoes = 0;

    double t0 = omp_get_wtime();

    /*
     * PARALELIZAÇÃO — loop colapsado das duas primeiras linhas
     * ──────────────────────────────────────────────────────────
     * Paralelizar só a linha 0 gera N tarefas (ex: 15 para N=15).
     * Paralelizar linhas 0 e 1 juntas gera até N² tarefas (ex: 225),
     * o que distribui melhor a carga entre muitos cores.
     *
     * collapse(2): o OpenMP trata os dois loops como um só,
     *   criando N² iterações independentes para distribuir.
     *
     * schedule(dynamic, 1): essencial porque subárvores têm
     *   tamanhos muito diferentes — distribui dinamicamente,
     *   evitando thread ociosa esperando outra terminar uma
     *   subárvore grande (balanceamento de carga).
     *
     * reduction(+:total_solucoes): cada thread acumula localmente
     *   e o OpenMP soma no final — sem race condition.
     *
     * board_local[]: declarado DENTRO do loop — privado por thread
     *   automaticamente (stack local), sem variável compartilhada.
     */
    #pragma omp parallel for collapse(2) schedule(dynamic, 1) reduction(+:total_solucoes)
    for (int col0 = 0; col0 < n; col0++) {
        for (int col1 = 0; col1 < n; col1++) {

            /* board privado por thread */
            int board_local[N_MAX];
            memset(board_local, -1, sizeof(board_local));

            long long count = 0;

            /* fixa rainha da linha 0 */
            board_local[0] = col0;

            /* verifica se rainha da linha 1 é válida antes de explorar */
            if (place(board_local, 1, col1)) {
                board_local[1] = col1;
                /* explora subárvore a partir da linha 2 */
                queen(board_local, 2, n, &count);
            }

            total_solucoes += count;
        }
    }

    double tempo = omp_get_wtime() - t0;

    printf("+-------------------------------+\n");
    printf("| N               : %4d        |\n", n);
    printf("| Threads         : %4d        |\n", num_threads);
    printf("| Solucoes        : %10lld |\n", total_solucoes);
    printf("| Tempo (paralelo): %9.4f s |\n", tempo);
    printf("+-------------------------------+\n");

    return 0;
}