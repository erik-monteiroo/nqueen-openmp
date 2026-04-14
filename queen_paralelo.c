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

//função que verifica se pode por rainha
int place(int board_local[], int row, int col) {
    //pra cada linha anterior onde já tem rainhas
    for (int i = 0; i < row; i++) {
        //verifica se ta na mesma coluna
        if (board_local[i] == col)                       
            return 0;
        // verifica se ta na mesma diagonal
        if (abs(board_local[i] - col) == abs(i - row))  
            return 0;
    }
    return 1;
}


void queen(int board_local[], int row, int n, long long *count) {
    //caso base -> se cheguei no fim do tabuleiro, conto como solução pois todas as rainhas foram postas neste cenario
    if (row == n) {          
        (*count)++;
        return;
    }

    //se ainda não chegamos,
    //tenta-se colocar mais uma rainha na linha atual
    //pra cada coluna do tabuleiro
    for (int col = 0; col < n; col++) {
        //se é possível colocar uma rainha
        if (place(board_local, row, col)) {
            //poe a rainha
            board_local[row] = col;
            //e continua na linha de baixo (row + 1) recursivamente
            queen(board_local, row + 1, n, count);
            //tanto faz se achou solucao ou nao, esta rainha é apagada desta posição
            //assim no proximo ciclo do for é possivel testar a rainha na proxima posicao
            //gerando o proximo cenario possivel
            board_local[row] = -1;  
        }
    }
}

//função principal
int main(int argc, char *argv[]) {
    //print de como deve ser o input
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <N> <num_threads>\n", argv[0]);
        fprintf(stderr, "Ex : %s 15 4\n", argv[0]);
        return 1;
    }
    //pega o input
    int n           = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    //verificacoes
    if (n < 1 || n > N_MAX) {
        fprintf(stderr, "ERRO: N deve estar entre 1 e %d\n", N_MAX);
        return 1;
    }
    if (num_threads < 1) {
        fprintf(stderr, "ERRO: numero de threads deve ser >= 1\n");
        return 1;
    }

    //starta o numero de threads 
    omp_set_num_threads(num_threads);

    printf("================================================\n");
    printf("  N-Rainhas — Versao Paralela OMP\n");
    printf("================================================\n");
    printf("  N           = %d  (tabuleiro %dx%d)\n", n, n, n);
    printf("  Threads     = %d\n\n", num_threads);

    //variavel onde se guarda o total de solucoes
    long long total_solucoes = 0;

    //relogio do omp para pegar o tempo de execução geral
    //esse é o tempo de inicio
    double t0 = omp_get_wtime();

    //   #pragma omp parallel for \  - manda o proximo for ser paralelo   
    //          collapse(2) \ -  combina os dois for (cada tarefa é uma combinação possivel dos dois NxN)
    //          schedule(dynamic, 1)\ - distribui conmforme for ficando livre de 1 em 1 tarefa
    //          reduction(+:total_solucoes) - soma todas as variaveis internas com a externa geral
   
    #pragma omp parallel for collapse(2) schedule(dynamic, 1) reduction(+:total_solucoes)

    //Esses dois for fazem o que a funcao queen faz, mas manualmente para as duas primeiras colunas,
    //dessa forma é possivel dividir NxN cenarios iniciais para atribuir as threads
    //cada cenário se torna uma tarefa
    
    //percorre procurando lugar para a primeira rainha
    for (int col0 = 0; col0 < n; col0++) {
        //percorre procurando lugar para a segunda rainha
        for (int col1 = 0; col1 < n; col1++) {

            //A PARTIR DAQUI É UMA TAREFA ()
            //cria um board local para esta thread
            int board_local[N_MAX];

            memset(board_local, -1, sizeof(board_local));

            //variavel local que conta numero de solucoes
            long long count = 0;

            /* fixa rainha da linha 0 */
            board_local[0] = col0;

            //faz uma verificação rápida se este cenário da tarefa é válido para continuar
            if (place(board_local, 1, col1)) {
                //coloca a rainha la
                board_local[1] = col1;
                //continua na subarvore deste
                queen(board_local, 2, n, &count);
            }
            //soma count no final
            total_solucoes += count;
        }
    }

    //calcula o tempo de exeucuao
    double tempo = omp_get_wtime() - t0;

    //print final
    printf("+-------------------------------+\n");
    printf("| N               : %4d        |\n", n);
    printf("| Threads         : %4d        |\n", num_threads);
    printf("| Solucoes        : %10lld |\n", total_solucoes);
    printf("| Tempo (paralelo): %9.4f s |\n", tempo);
    printf("+-------------------------------+\n");

    return 0;
}