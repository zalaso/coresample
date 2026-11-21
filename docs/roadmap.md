# Roadmap

Durate indicative, lavorando part-time con l'aiuto degli agenti.

## Settimana 0: decisioni

| Decisione | Scelta | Stato |
| --- | --- | --- |
| Linguaggio del circuito | SystemVerilog, simulato con Verilator | Deciso |
| Linguaggio del simulatore di reti | Rust: compila anche in WebAssembly, così il simulatore può girare nel browser accanto alle tracce | Deciso |
| Budget mensile per le API | | Da decidere |
| Modelli | Opus 5.5 di default, Sonnet 5.5 per la routine, Fable 5.2 solo dove Opus fallisce | Deciso |

## Fase 1, settimane 1-6: le due basi in parallelo

- **CPU:** core RV32IM a ciclo singolo, riscv-tests, confronto passo per passo con Spike, poi visualizzatore della traccia.
- **Rete (dalla settimana 3):** simulatore con autograd: MLP, rete convoluzionale, blocco transformer, ogni operazione testata contro PyTorch.

**G1, settimana 6:** il core passa tutti i test senza divergenze da Spike. Primo rilascio pubblico.

## Fase 2, settimane 7-13: unire i pezzi

- **CPU:** pipeline a 5 stadi, traccia interna in formato Kanata con la causa di ogni stallo (vista in Konata).
- **Rete:** quantizzazione int8 con riferimento PyTorch.
- **Integrazione:** compilatore con mappa dei collegamenti e marcatori, che punta prima al core a ciclo singolo.

**G2, settimana 13:** una piccola rete gira sul core con output identico al riferimento int8.

## Fase 3, settimane 14-18: lo strumento

Indice della traccia, flame graph, pannelli collegati, confronto tra due esecuzioni, tracce precalcolate nel browser.

**G3, settimana 18:** prima versione completa.

## Dopo

Cache, reti convoluzionali per TinyML, domande tra livelli componibili, simulazione dal vivo nel browser, FPGA facoltativa.

## Regole

- Per ogni tappa i test si scrivono prima del codice.
- Nessuna fase inizia finché il traguardo precedente non è verificato.

## Stato della fase 1, CPU

- [x] Struttura del repository e script di installazione
- [x] Ambiente minimo per riscv-tests e runner con confronto Spike
- [x] Core RV32IM a ciclo singolo, prima stesura
- [x] Prima compilazione con Verilator, senza avvisi con `-Wall`
- [x] rv32ui e rv32um passano sul core: 49 test su 49 (escluso `ma_data`)
- [x] Controllo che i test sappiano fallire: un core con SUB sbagliata fa fallire `rv32ui-sub`
- [x] Nessuna divergenza da Spike: 49 test, ogni istruzione identica in pc, codifica, registro scritto e accesso in memoria
- [x] Controllo che il confronto sappia fallire: un valore scritto alterato a mano viene segnalato all'istruzione esatta
- [x] riscv-arch-test (ACT4): 52 test su 52 (I, M, Zmmul, Zifencei), ogni istruzione identica a Spike. I risultati attesi li calcola Spike invece di Sail, con la stessa catena di ACT
- [x] Controllo che riscv-arch-test sappia fallire: un valore atteso alterato fa fallire il test, e un core con MULHSU sbagliata fa fallire solo i due test di MULHSU
- [x] Programma C sul core: `dense_int8`, uno strato denso int8 con ReLU, risultato confrontato con un riferimento Python e identico a Spike istruzione per istruzione
- [x] Forme d'onda VCD dal simulatore, da aprire in Surfer
- [x] Traccia Perfetto: flame graph delle chiamate e una slice per istruzione, sullo stesso asse dei tempi del VCD
- [x] Test tra livelli: istruzioni nella traccia = istruzioni ritirate, frame annidati, ogni istruzione nella funzione giusta, pc del VCD = pc del log ciclo per ciclo
- [x] Rilascio pubblico (G1): v0.1.0 su GitHub, licenza Apache-2.0

## Stato della fase 1, rete

- [x] Rust 1.83.0 fissato in `sim/nn/rust-toolchain.toml`, PyTorch 2.5.1 (CPU) in `.venv` per i riferimenti
- [x] Riferimenti PyTorch in un formato binario minimo, f32 esatti, riproducibili bit per bit
- [x] Tensori, grafo con operazioni esplicite, autograd: matmul, add, add_bias, relu, cross_entropy
- [x] Ogni operazione, avanti e indietro, entro 1e-5 + 1e-4·|b| da PyTorch; gradienti controllati anche con le differenze finite
- [x] MLP 2-16-3 addestrato con SGD per 100 passi: ogni loss e ogni peso finale entro 4e-7 da PyTorch
- [x] Controllo che i test sappiano fallire: ReLU con gradiente 1 in zero, cross-entropy senza media sul batch
- [x] Rete convoluzionale: conv2d (stride, padding, kernel rettangolari), max pool con i pareggi come PyTorch, flatten; CNN addestrata per 60 passi entro 3e-7 da PyTorch
- [x] Controllo che i test sappiano fallire: pareggi del max pool, kernel ribaltato, stride ignorato. Il kernel ribaltato passa le differenze finite e lo trova solo il confronto con PyTorch
- [ ] Blocco transformer (layer norm, softmax, attenzione)
