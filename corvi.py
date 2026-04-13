# ============================================================
# RILEVATORE DI CORVI - Per tablet Android
# ============================================================
# Questo programma fa 3 cose:
#   1. Guarda continuamente cosa riprende la fotocamera
#   2. Usa l'intelligenza artificiale per cercare uccelli/corvi
#   3. Registra un video quando ne trova uno, e lo salva
#      solo dopo 30 secondi di assenza dell'uccello
#
# NOTA: usa cv2.dnn (motore AI integrato in OpenCV)
#       NON serve installare onnxruntime
# ============================================================

import cv2      # gestisce fotocamera, immagini, video E il motore AI
import numpy as np  # calcoli matematici rapidi sulle immagini
import time     # funzioni per misurare il tempo
import os       # funzioni per gestire file e cartelle
import requests    # invia file e messaggi via internet (Telegram)
import subprocess  # esegue programmi esterni (ffmpeg per comprimere il video)

# ============================================================
# IMPOSTAZIONI - Modifica questi valori se necessario
# ============================================================

# Secondi di assenza dell'uccello prima di fermare la registrazione
SECONDI_SENZA_CORVO = 30

# Cartella dove verranno salvati i video sul tablet
CARTELLA_VIDEO = "/sdcard/rilevatore_corvi/"

# Percorso del file del modello AI
MODELLO_AI = "/sdcard/rilevatore_corvi/yolov8n.onnx"

# Soglia di confidenza minima per accettare un rilevamento
# 0.4 = almeno 40% di certezza
SOGLIA_CONFIDENZA = 0.4

# Numero della classe "uccello" nel modello YOLOv8
# Il modello conosce 80 oggetti: gli uccelli sono il numero 14
CLASSE_UCCELLO = 14

# Dimensione immagine richiesta dal modello (sempre 640x640 per YOLOv8)
DIMENSIONE_MODELLO = 640

# Analizza con l'AI solo ogni N fotogrammi per risparmiare CPU
# 15 = analisi ogni 15 fotogrammi — l'AI blocca meno il loop, più frame scritti
ANALIZZA_OGNI_N_FRAME = 15

# Secondi minimi di presenza del corvo per salvare il video
# Se il corvo è stato visibile meno di 10 secondi in totale, il file viene eliminato
SECONDI_MINIMI_CORVO = 5

# Risoluzione del video salvato sul tablet (larghezza, altezza)
RISOLUZIONE_SALVATAGGIO = (1280, 720)

# Fotogrammi al secondo del video salvato sul tablet
FPS_SALVATAGGIO = 60

# Risoluzione del video compresso inviato su Telegram
RISOLUZIONE_TELEGRAM = (854, 480)
# FPS del video compresso per Telegram
FPS_TELEGRAM = 60

# --- IMPOSTAZIONI TELEGRAM ---
# Token del bot creato con @BotFather
TELEGRAM_TOKEN  = "8742079341:AAF7HN8ZVcxwQp-wH0oc75pX6vC5yNNBkjs"
# Il tuo Chat ID (ricevuto con getUpdates)
TELEGRAM_CHAT_ID = "155938019"


# ============================================================
# FUNZIONE: crea la cartella di output se non esiste
# ============================================================
def crea_cartella_output():
    if not os.path.exists(CARTELLA_VIDEO):
        os.makedirs(CARTELLA_VIDEO)
        print(f"Cartella creata: {CARTELLA_VIDEO}")
    else:
        print(f"Cartella output: {CARTELLA_VIDEO}")


# ============================================================
# FUNZIONE: esegue l'analisi AI e restituisce gli uccelli trovati
# Usa cv2.dnn, il motore AI già incluso in OpenCV (no onnxruntime)
# ============================================================
def trova_uccelli(rete_ai, frame, larghezza_frame, altezza_frame):
    # cv2.dnn.blobFromImage prepara il fotogramma per il modello:
    # - lo ridimensiona a 640x640
    # - divide i valori per 255 (normalizzazione)
    # - inverte i canali colore da BGR a RGB (swapRB=True)
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0 / 255.0,        # normalizza i pixel da 0-255 a 0.0-1.0
        size=(DIMENSIONE_MODELLO, DIMENSIONE_MODELLO),  # ridimensiona a 640x640
        mean=(0, 0, 0),                  # nessuna sottrazione della media
        swapRB=True,                     # converte BGR -> RGB
        crop=False                       # non ritaglia, ridimensiona e basta
    )

    # Passiamo il blob preparato come ingresso alla rete neurale
    rete_ai.setInput(blob)

    # Eseguiamo l'analisi: la rete elabora l'immagine e produce le predizioni
    # output ha forma (1, 84, 8400):
    #   84 = 4 coordinate + 80 classi di oggetti
    #   8400 = numero di candidati analizzati
    output = rete_ai.forward()

    # Rimuoviamo la prima dimensione inutile: da (1, 84, 8400) a (84, 8400)
    predizioni = np.squeeze(output)

    # Trasponiamo per avere (8400, 84): più comodo da scorrere riga per riga
    predizioni = predizioni.T

    # Fattori di scala per riportare le coordinate a quelle reali del frame
    scala_x = larghezza_frame / DIMENSIONE_MODELLO
    scala_y = altezza_frame / DIMENSIONE_MODELLO

    uccelli_trovati = []

    for rilevamento in predizioni:
        # Colonne 4-83 = punteggi delle 80 classi di oggetti
        punteggi_classi = rilevamento[4:]

        # Troviamo la classe con il punteggio più alto
        classe_migliore = int(np.argmax(punteggi_classi))
        confidenza = float(punteggi_classi[classe_migliore])

        # Accettiamo solo se è un uccello con confidenza sufficiente
        if classe_migliore == CLASSE_UCCELLO and confidenza >= SOGLIA_CONFIDENZA:

            # Colonne 0-3 = coordinate del rettangolo (centro_x, centro_y, w, h)
            cx = rilevamento[0] * scala_x
            cy = rilevamento[1] * scala_y
            w  = rilevamento[2] * scala_x
            h  = rilevamento[3] * scala_y

            # Convertiamo da centro+dimensioni ad angoli del rettangolo
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            uccelli_trovati.append({
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'confidenza': confidenza
            })

    return uccelli_trovati


# ============================================================
# FUNZIONE: disegna rettangoli verdi attorno agli uccelli trovati
# ============================================================
def disegna_rilevamenti(frame, uccelli):
    for uccello in uccelli:
        # Rettangolo verde attorno all'uccello
        cv2.rectangle(
            frame,
            (uccello['x1'], uccello['y1']),
            (uccello['x2'], uccello['y2']),
            (0, 255, 0),  # verde
            2             # spessore 2 pixel
        )

        # Testo con percentuale di certezza sopra il rettangolo
        testo = f"Uccello {uccello['confidenza'] * 100:.0f}%"
        cv2.putText(
            frame, testo,
            (uccello['x1'], uccello['y1'] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

    return frame


# ============================================================
# FUNZIONE: crea una versione compressa del video per Telegram
# Usa ffmpeg con codec H.264, molto più efficiente di mp4v
# ============================================================
def comprimi_video(percorso_originale):
    # Creiamo il nome del file compresso aggiungendo "_tg" prima dell'estensione
    # es: corvo_20260412_143022.mp4 → corvo_20260412_143022_tg.mp4
    base, estensione = os.path.splitext(percorso_originale)
    percorso_compresso = base + "_tg" + estensione

    larghezza, altezza = RISOLUZIONE_TELEGRAM

    print(f"[FFMPEG] Compressione video per Telegram...")

    # Costruiamo il comando ffmpeg:
    # -i          = file di input
    # -vf scale   = ridimensiona al nuovo formato
    # -c:v libx264= codec H.264 (molto più efficiente di mp4v)
    # -crf 28     = qualità: 0=perfetta, 51=pessima. 28 = buon compromesso
    # -preset fast= velocità di codifica (fast = veloce, minor qualità di compressione)
    # -an         = nessuna traccia audio (la fotocamera non registra audio)
    # -y          = sovrascrive senza chiedere se il file esiste già
    comando = [
        "ffmpeg",
        "-i", percorso_originale,
        "-vf", f"scale={larghezza}:{altezza}",
        "-c:v", "libx264",
        "-crf", "28",
        "-preset", "fast",
        "-r", str(FPS_TELEGRAM),
        "-an",
        "-y",
        percorso_compresso
    ]

    try:
        # Eseguiamo ffmpeg e aspettiamo che finisca
        # stdout/stderr DEVNULL = non mostriamo l'output di ffmpeg nella console
        subprocess.run(
            comando,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300  # massimo 5 minuti per la compressione
        )

        if os.path.exists(percorso_compresso):
            dim_originale  = os.path.getsize(percorso_originale) / (1024 * 1024)
            dim_compressa  = os.path.getsize(percorso_compresso) / (1024 * 1024)
            print(f"[FFMPEG] Originale: {dim_originale:.1f} MB → Telegram: {dim_compressa:.1f} MB")
            return percorso_compresso
        else:
            print("[FFMPEG] Errore: file compresso non creato")
            return percorso_originale  # fallback: mandiamo l'originale

    except subprocess.TimeoutExpired:
        print("[FFMPEG] Timeout compressione — invio originale")
        return percorso_originale
    except FileNotFoundError:
        print("[FFMPEG] ffmpeg non trovato — esegui: pkg install ffmpeg")
        return percorso_originale


# ============================================================
# FUNZIONE: invia il video su Telegram
# ============================================================
def invia_video_telegram(percorso_video, secondi_visibile):
    # Prima comprimiamo il video per Telegram
    percorso_da_inviare = comprimi_video(percorso_video)

    print(f"[TELEGRAM] Invio video in corso...")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo"

    dim_mb = os.path.getsize(percorso_da_inviare) / (1024 * 1024)
    didascalia = (
        f"Corvo rilevato!\n"
        f"Tempo visibile: {secondi_visibile:.0f} secondi\n"
        f"Dimensione: {dim_mb:.1f} MB"
    )

    try:
        with open(percorso_da_inviare, 'rb') as file_video:
            risposta = requests.post(
                url,
                data={
                    'chat_id': TELEGRAM_CHAT_ID,
                    'caption': didascalia,
                },
                files={
                    'video': file_video
                },
                timeout=180  # 3 minuti per l'upload
            )

        if risposta.status_code == 200:
            print("[TELEGRAM] Video inviato con successo!")
        else:
            print(f"[TELEGRAM] Errore invio: codice {risposta.status_code}")
            print(f"[TELEGRAM] Dettaglio: {risposta.text}")

    except requests.exceptions.Timeout:
        print("[TELEGRAM] Timeout: connessione lenta o video troppo grande")
    except requests.exceptions.ConnectionError:
        print("[TELEGRAM] Errore: nessuna connessione internet")
    except Exception as e:
        print(f"[TELEGRAM] Errore imprevisto: {e}")

    finally:
        # Cancelliamo il file compresso temporaneo (_tg) dopo l'invio
        # Il file originale ad alta qualità rimane sul tablet
        if percorso_da_inviare != percorso_video and os.path.exists(percorso_da_inviare):
            os.remove(percorso_da_inviare)
            print("[TELEGRAM] File temporaneo eliminato")


# ============================================================
# FUNZIONE PRINCIPALE
# ============================================================
def main():
    print("=" * 55)
    print("   RILEVATORE DI CORVI - Avvio in corso...")
    print("=" * 55)

    # Creiamo la cartella dove salvare i video
    crea_cartella_output()

    # Controlliamo che il file del modello AI esista
    if not os.path.exists(MODELLO_AI):
        print(f"\nERRORE: File modello non trovato!")
        print(f"Percorso cercato: {MODELLO_AI}")
        print("Copia yolov8n.onnx in /sdcard/rilevatore_corvi/")
        return

    # Carichiamo il modello AI usando cv2.dnn (motore integrato in OpenCV)
    # readNetFromONNX legge direttamente il file .onnx senza librerie extra
    print("\nCaricamento modello AI...")
    rete_ai = cv2.dnn.readNetFromONNX(MODELLO_AI)
    print("Modello AI caricato!")

    # --- APERTURA FOTOCAMERA VIA IP WEBCAM ---

    print("\nConnessione alla fotocamera via IP Webcam...")
    # L'app IP Webcam crea uno stream video accessibile via HTTP
    # 127.0.0.1 = questo stesso tablet, 8080 = porta di default
    URL_STREAM = "http://127.0.0.1:8080/video"
    fotocamera = cv2.VideoCapture(URL_STREAM)

    if not fotocamera.isOpened():
        print("\nERRORE: Impossibile connettersi allo stream!")
        print("Controlla che:")
        print("  1. L'app IP Webcam sia aperta")
        print("  2. Hai toccato 'Avvia server' in fondo all'app")
        print(f"  3. URL usato: {URL_STREAM}")
        return

    # Leggiamo le dimensioni effettive dello stream
    larghezza = int(fotocamera.get(cv2.CAP_PROP_FRAME_WIDTH))
    altezza   = int(fotocamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps       = int(fotocamera.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25  # valore di sicurezza se non rilevato

    print(f"Fotocamera: {larghezza}x{altezza} @ {fps}fps")

    # --- VARIABILI DI STATO ---

    sta_registrando        = False  # True = stiamo registrando
    scrittore_video        = None   # oggetto che scrive il file video
    tempo_ultimo_corvo     = None   # quando abbiamo visto l'ultimo uccello
    nome_file_video        = None   # nome del file video corrente
    contatore_frame        = 0      # conta i frame per sapere quando analizzare
    uccelli_correnti       = []     # risultato dell'ultima analisi AI
    secondi_corvo_totali   = 0.0    # quanti secondi il corvo è stato visibile in questa registrazione
    ultimo_tick_corvo      = None   # timestamp dell'ultimo frame in cui il corvo era visibile
                                    # serve per calcolare quanto tempo è rimasto in inquadratura

    print("\n" + "=" * 55)
    print("  Monitoraggio ATTIVO — Premi CTRL+C per uscire")
    print(f"  Timer di stop: {SECONDI_SENZA_CORVO} secondi")
    print("=" * 55 + "\n")

    # Calcoliamo quanti secondi aspettare tra un frame e l'altro
    # 1 diviso 30fps = 0.033 secondi = 33 millisecondi
    pausa_per_frame = 1.0 / FPS_SALVATAGGIO

    # --- CICLO PRINCIPALE ---

    try:
        while True:

            # Aspettiamo il tempo giusto prima di leggere il prossimo frame
            # Questo evita di leggere sempre lo stesso frame in cache dallo stream HTTP
            time.sleep(pausa_per_frame)

            # Leggiamo un fotogramma dalla fotocamera
            successo, frame = fotocamera.read()

            # Se il frame non è valido, saltiamo
            if not successo or frame is None:
                continue

            contatore_frame += 1

            # Analizziamo con l'AI solo ogni ANALIZZA_OGNI_N_FRAME frame
            if contatore_frame % ANALIZZA_OGNI_N_FRAME == 0:
                uccelli_correnti = trova_uccelli(
                    rete_ai, frame, larghezza, altezza
                )

            # C'è almeno un uccello visibile?
            corvo_visibile = len(uccelli_correnti) > 0

            momento_attuale = time.time()

            if corvo_visibile:
                # Uccello visibile: resettiamo il timer dei 30 secondi
                tempo_ultimo_corvo = momento_attuale

                # Accumuliamo il tempo di visibilità del corvo
                # Se ultimo_tick_corvo non è None, il corvo era visibile anche al frame precedente:
                # aggiungiamo il tempo trascorso dall'ultima volta che lo abbiamo visto
                if ultimo_tick_corvo is not None:
                    secondi_corvo_totali += momento_attuale - ultimo_tick_corvo
                # Aggiorniamo il timestamp dell'ultimo avvistamento
                ultimo_tick_corvo = momento_attuale

                if not sta_registrando:
                    # Iniziamo a registrare
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    nome_file_video = os.path.join(
                        CARTELLA_VIDEO, f"corvo_{timestamp}.mp4"
                    )
                    codec = cv2.VideoWriter_fourcc(*'mp4v')
                    scrittore_video = cv2.VideoWriter(
                        nome_file_video, codec, FPS_SALVATAGGIO, RISOLUZIONE_SALVATAGGIO
                    )
                    sta_registrando = True
                    print(f"\n[REC] Registrazione iniziata -> {nome_file_video}")

            else:
                # Corvo non visibile: interrompiamo l'accumulo del tempo di visibilità
                # (riprenderà automaticamente quando tornerà)
                ultimo_tick_corvo = None

                # Controlliamo il timer dei 30 secondi
                if sta_registrando and tempo_ultimo_corvo is not None:
                    secondi_assenza = momento_attuale - tempo_ultimo_corvo
                    secondi_rimasti = SECONDI_SENZA_CORVO - secondi_assenza

                    if secondi_rimasti > 0:
                        print(
                            f"[TIMER] Corvo assente — stop tra {int(secondi_rimasti)}s "
                            f"| Visibile finora: {secondi_corvo_totali:.0f}s   ",
                            end='\r'
                        )

                    # Timer scaduto: decidiamo se salvare o eliminare il video
                    if secondi_assenza >= SECONDI_SENZA_CORVO:
                        scrittore_video.release()
                        scrittore_video = None
                        sta_registrando = False
                        tempo_ultimo_corvo = None
                        uccelli_correnti   = []
                        ultimo_tick_corvo  = None

                        if secondi_corvo_totali >= SECONDI_MINIMI_CORVO:
                            # Il corvo è stato visibile abbastanza: salviamo e inviamo
                            print(f"\n[SALVATO] Corvo visibile {secondi_corvo_totali:.0f}s → {nome_file_video}")
                            invia_video_telegram(nome_file_video, secondi_corvo_totali)
                        else:
                            # Il corvo è apparso troppo poco: eliminiamo il file
                            os.remove(nome_file_video)
                            print(f"\n[ELIMINATO] Corvo visibile solo {secondi_corvo_totali:.0f}s "
                                  f"(minimo {SECONDI_MINIMI_CORVO}s) — video scartato")

                        nome_file_video      = None
                        secondi_corvo_totali = 0.0  # resettiamo il contatore per la prossima sessione

            # Scriviamo il fotogramma nel video (se stiamo registrando)
            if sta_registrando and scrittore_video is not None:
                # Ridimensioniamo il frame alla risoluzione di salvataggio (es. 640x360)
                # Nessun riquadro verde: salviamo il video pulito così com'è
                frame_piccolo = cv2.resize(frame, RISOLUZIONE_SALVATAGGIO)
                scrittore_video.write(frame_piccolo)

            # Stato nella console
            if corvo_visibile and sta_registrando:
                print(
                    f"[REC] {len(uccelli_correnti)} uccello/i | Timer resettato   ",
                    end='\r'
                )
            elif not corvo_visibile and not sta_registrando:
                print("[IN ASCOLTO] Nessun uccello...   ", end='\r')

    except KeyboardInterrupt:
        print("\n\nInterruzione ricevuta, chiusura...")

    finally:
        # Se la registrazione era in corso al momento dell'interruzione,
        # applichiamo anche qui la regola dei 10 secondi minimi
        if scrittore_video is not None:
            scrittore_video.release()
            if secondi_corvo_totali >= SECONDI_MINIMI_CORVO:
                print(f"[SALVATO] Corvo visibile {secondi_corvo_totali:.0f}s → {nome_file_video}")
                invia_video_telegram(nome_file_video, secondi_corvo_totali)
            else:
                os.remove(nome_file_video)
                print(f"[ELIMINATO] Corvo visibile solo {secondi_corvo_totali:.0f}s — video scartato")

        fotocamera.release()
        print("Programma terminato.")


if __name__ == "__main__":
    main()
