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
- [x] Blocco transformer: GPT a un blocco (pre-norm, attenzione causale multi-testa, MLP con GELU) in `models.rs`, fatto di operazioni elementari: embedding, layer norm, permute, batched matmul, scale, maschera causale, softmax, GELU. Addestrato per 80 passi entro 8e-7 da PyTorch; differenze finite su tutto il modello
- [x] Controllo che i test sappiano fallire: varianza della layer norm, gradiente della softmax, maschera che lascia vedere il futuro, derivata della GELU. La maschera sbagliata passa le differenze finite e la trovano solo il test unitario e PyTorch
- [x] Lezione: un riferimento di addestramento deve essere stabile. Con lr 0,3 PyTorch stesso ha un picco di loss al passo 70 e lì gli arrotondamenti esplodono; il generatore ora rifiuta le corse in cui la loss risale

## Stato della fase 2

Ordine: prima quantizzazione e compilatore (il percorso verso G2), poi la pipeline.

- [x] Quantizzazione int8 a soli interi (`quant.rs`): attivazioni uint8 con zero point, pesi int8 simmetrici, bias e accumulo int32, riquantizzazione in virgola fissa con un solo arrotondamento
- [x] Confronto con i kernel quantizzati di PyTorch (qnnpack) sull'MLP: tutti i valori interi identici (1536 nascosti, 288 in uscita), stessa accuratezza
- [x] Controllo che i test sappiano fallire: riquantizzazione troncata (la trova solo la soglia del 99% di identici), zero point ignorato, arrotondamento lontano da zero (lo trova solo il test unitario)
- [x] Quantizzazione della CNN, la rete per G2: convoluzione int8 (il padding vale lo zero point), max pool esatto su uint8, `QNetwork` come descrizione della rete per il compilatore. Ogni intero identico a PyTorch (12288 della convoluzione, 3072 del pooling, 144 in uscita), 46 immagini su 48 classificate giuste in entrambi
- [x] Controllo che i test sappiano fallire: kernel trasposto, stride del pooling ignorato, padding perso negli indici
- [x] Compilatore (`compiler/`): QNetwork → assembly RV32IM, una funzione per strato, regioni operazione (`bias`, `mac`, `requant`, `store`, `max`, `loop`…) nella sezione `.cs_map` dell'ELF
- [x] **G2 raggiunto:** la CNN gira sul core su tutte le 48 immagini, output identico al riferimento int8 byte per byte; passa anche su Spike; su un'immagine 75 805 istruzioni identiche a Spike. 3,6 milioni di cicli, 89% nella convoluzione
- [x] Test del compilatore: strati lineari su tutti i percorsi della riquantizzazione, convoluzioni di varie geometrie, pooling, auto-controllo, mappa che copre ogni istruzione eseguita
- [x] Controllo che i test sappiano fallire: riquantizzazione senza arrotondamento, `mulhu` al posto di `mulh`, pesi non trasposti, controllo del padding mancante. Il confronto con Spike da solo non li troverebbe: core e Spike eseguono lo stesso programma sbagliato. Li trova l'auto-controllo contro Rust int8
- [x] Pipeline a 5 stadi (`cs_pipe.sv`): forwarding da MEM e WB, bypass dal WB in lettura, stallo load-use, salti risolti in EX, FENCE.I come refetch. Tutte le suite girano anche sulla pipeline (`CORE=pipe`), identiche a Spike
- [x] Traccia Kanata (`pipeline_to_kanata.py`) con la causa di ogni stallo e di ogni istruzione scartata; conto dei cicli in cui ogni bolla ha esattamente una causa
- [x] CNN sulla pipeline: 88 030 cicli per immagine, CPI 1,161; i branch presi costano più degli stalli load-use (8,4% contro 5,1%)
- [x] Controllo che i test sappiano fallire: forwarding da MEM, forwarding da WB, bypass dal WB, stallo load-use, flush dello stadio ID, FENCE.I. Il bug di FENCE.I sfuggiva a riscv-tests e arch-test: aggiunto `sw/tests/fence_i_next.S`, che lo trova

**Fase 2 completa.** Il livello "operazione" è nell'ELF (`.cs_map`) ma non ancora nelle viste: è il primo lavoro della fase 3.

## Stato della fase 3

Decisione: G3 con la CNN; il transformer sul core è la prima estensione dopo G3. Il README ora dice "un input" invece di "un token".

- [x] Livello "operazione" nelle viste: lettore di `.cs_map` (`csmap.py`), tratti di operazione in Perfetto tra strato e istruzione, tempi reali della pipeline (uno stallo allunga l'istruzione che aspetta)
- [x] Report per strato e operazione (`op_report.py`): ogni ciclo addebitato a una sola riga (load-use all'istruzione che aspetta, svuotamenti al salto), le righe sommano al totale. Il `mac` della convoluzione è l'80% dei cicli
- [x] Controlli tra livelli: in ogni strato la somma delle operazioni è uguale alla durata dello strato, su entrambi i core; sulla pipeline il frame radice copre esattamente tutti i cicli
- [x] Indice della traccia (`trace_index.py`): blocchi grezzi da 4096 cicli, piramide di contenitori, albero esecuzione → immagine → strato. 48 immagini sulla pipeline: 4,2 milioni di cicli, `index.json` 214 kB, costruito a flusso con 21 MB di RAM; domande esatte su qualunque intervallo in decine di ms
- [x] Controlli: ogni intervallo confrontato col conto ciclo per ciclo (anche con le bolle), totali per operazione e per strato uguali a `op_report`
- [ ] Flame graph su tutti i livelli e pannelli collegati, in un'interfaccia nostra
- [ ] Confronto tra due esecuzioni
- [ ] Tracce precalcolate nel browser (sito statico)
