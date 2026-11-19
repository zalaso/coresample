# Diario dei costi

Una riga per sessione di lavoro con un agente. Serve a sapere quanto costa
davvero ogni tappa e quando un modello più costoso vale la differenza.

| Data | Tappa | Modello | Obiettivo | Token (in / out) | Costo ($) | Test passati? | Dove si è bloccato |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-13 | Fase 1, CPU | Opus 5.5 | Struttura del repo, core a ciclo singolo, runner dei test | | | Non ancora eseguiti | Mancano Verilator, GCC RISC-V e Spike |
| 2026-11-13 | Fase 1, CPU | Opus 5.5 | Prima compilazione, riscv-tests, confronto con Spike | | | 49/49, identici a Spike | Solo errori di build: percorso relativo nel Makefile, sottomodulo di Spike mancante. Il core non ha richiesto correzioni funzionali |
| 2026-11-19 | Fase 1, CPU | Opus 5.5 | riscv-arch-test (ACT4) con Spike come riferimento | | | 52/52, identici a Spike | binutils 2.42 riempiva di zeri gli allineamenti nei test di salto: risolto con `-Wa,-mno-relax`. Il core non ha richiesto correzioni |
| 2026-11-19 | Fase 1, CPU | Opus 5.5 | Visualizzatore della traccia: programma C, VCD, traccia Perfetto | | | 17 test unitari e tra livelli, regressione completa | Un test tra livelli ha trovato le chiamate in coda (`tail check`) non riconosciute: corretto. Il browser integrato blocca localhost dai siti pubblici: Perfetto verificato con un estratto |
