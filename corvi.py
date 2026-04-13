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
import threading   # permette di eseguire l'AI in parallelo alla registrazione
from config import TELEGRAM_TOKEN  # token segreto, non nel codice pubblico

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

# Quante volte al secondo il thread AI analizza un frame
# 2 = due analisi al secondo — abbastanza per rilevare un corvo senza pesare sulla CPU
ANALISI_AL_SECONDO = 2

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
# File dove vengono salvati i Chat ID di tutti gli utenti che hanno avviato il bot
# Ogni volta che qualcuno manda /start al bot, il suo ID viene aggiunto qui
FILE_UTENTI = "/sdcard/rilevatore_corvi/utenti.txt"

# File dove salviamo l'ultimo aggiornamento Telegram letto
# Serve per non rileggere sempre gli stessi messaggi
FILE_OFFSET = "/sdcard/rilevatore_corvi/telegram_offset.txt"


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
# FUNZIONE: thread separato che esegue l'AI in background
# Mentre questo thread analizza, il loop principale scrive video senza interruzioni
# ============================================================
def thread_ai(rete_ai, stato):
    # stato è un dizionario condiviso tra il thread AI e il loop principale
    # Usiamo un dizionario perché in Python è il modo più semplice
    # per condividere dati tra thread in modo sicuro
    while stato['attivo']:
        # Prendiamo una copia del frame più recente in modo sicuro
        with stato['lock_frame']:
            if stato['frame'] is None:
                time.sleep(0.05)
                continue
            frame_da_analizzare = stato['frame'].copy()

        # Eseguiamo l'analisi AI sul frame copiato
        # Questo può richiedere 1-2 secondi sul tablet, ma non blocca la registrazione
        risultato = trova_uccelli(
            rete_ai,
            frame_da_analizzare,
            stato['larghezza'],
            stato['altezza']
        )

        # Salviamo il risultato in modo sicuro
        with stato['lock_uccelli']:
            stato['uccelli'] = risultato

        # Aspettiamo prima della prossima analisi
        time.sleep(1.0 / ANALISI_AL_SECONDO)


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
# FUNZIONE: legge la lista degli utenti salvati nel file utenti.txt
# ============================================================
def leggi_utenti():
    # Se il file non esiste ancora, restituiamo lista vuota
    if not os.path.exists(FILE_UTENTI):
        return []
    with open(FILE_UTENTI, 'r') as f:
        # Leggiamo ogni riga, rimuoviamo spazi vuoti, saltiamo righe vuote
        return [riga.strip() for riga in f.readlines() if riga.strip()]


# ============================================================
# FUNZIONE: controlla se ci sono nuovi utenti che hanno scritto /start al bot
# Va chiamata periodicamente nel ciclo principale
# ============================================================
def registra_nuovi_utenti():
    # Leggiamo l'offset — cioè l'ID dell'ultimo messaggio già elaborato
    # Serve per non rileggere sempre gli stessi messaggi vecchi
    offset = 0
    if os.path.exists(FILE_OFFSET):
        with open(FILE_OFFSET, 'r') as f:
            contenuto = f.read().strip()
            if contenuto:
                offset = int(contenuto)

    try:
        # Chiamiamo l'API Telegram per ricevere i nuovi messaggi
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        risposta = requests.get(
            url,
            params={'offset': offset, 'timeout': 2},
            timeout=5
        )

        if risposta.status_code != 200:
            return

        dati = risposta.json()
        aggiornamenti = dati.get('result', [])

        if not aggiornamenti:
            return

        # Carichiamo gli utenti già registrati per non aggiungere duplicati
        utenti_esistenti = set(leggi_utenti())
        nuovi_trovati = []

        for aggiornamento in aggiornamenti:
            # Aggiorniamo l'offset: usiamo l'ID di questo messaggio + 1
            # così la prossima volta partiamo dal messaggio successivo
            nuovo_offset = aggiornamento['update_id'] + 1
            if nuovo_offset > offset:
                offset = nuovo_offset

            # Controlliamo se c'è un messaggio di testo
            messaggio = aggiornamento.get('message', {})
            testo = messaggio.get('text', '')
            chat_id = str(messaggio.get('chat', {}).get('id', ''))

            # Registriamo chiunque scriva qualcosa al bot (non solo /start)
            # così anche il tuo amico viene aggiunto automaticamente
            if chat_id and chat_id not in utenti_esistenti:
                utenti_esistenti.add(chat_id)
                nuovi_trovati.append(chat_id)
                print(f"[TELEGRAM] Nuovo utente registrato: {chat_id}")

                # Mandiamo un messaggio di benvenuto al nuovo utente
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    data={
                        'chat_id': chat_id,
                        'text': "Ciao! Sei registrato. Riceverai i video quando viene rilevato un corvo."
                    },
                    timeout=10
                )

        # Salviamo i nuovi utenti nel file
        if nuovi_trovati:
            with open(FILE_UTENTI, 'a') as f:
                for uid in nuovi_trovati:
                    f.write(uid + '\n')

        # Salviamo il nuovo offset
        with open(FILE_OFFSET, 'w') as f:
            f.write(str(offset))

    except Exception:
        # Se c'è un errore di rete ignoriamo silenziosamente
        pass


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

    # Leggiamo la lista di tutti gli utenti registrati
    utenti = leggi_utenti()

    if not utenti:
        print("[TELEGRAM] Nessun utente registrato — nessuno scritto /start al bot")
        return

    print(f"[TELEGRAM] Invio a {len(utenti)} utente/i...")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo"

    dim_mb = os.path.getsize(percorso_da_inviare) / (1024 * 1024)
    didascalia = (
        f"Corvo rilevato!\n"
        f"Tempo visibile: {secondi_visibile:.0f} secondi\n"
        f"Dimensione: {dim_mb:.1f} MB"
    )

    try:
        # Apriamo il file una volta sola e lo mandiamo a tutti
        with open(percorso_da_inviare, 'rb') as file_video:
            contenuto_video = file_video.read()  # leggiamo il file in memoria

        for chat_id in utenti:
            try:
                risposta = requests.post(
                    url,
                    data={'chat_id': chat_id, 'caption': didascalia},
                    files={'video': ('video.mp4', contenuto_video, 'video/mp4')},
                    timeout=180
                )
                if risposta.status_code == 200:
                    print(f"[TELEGRAM] Inviato a {chat_id}")
                else:
                    print(f"[TELEGRAM] Errore per {chat_id}: {risposta.status_code}")
            except Exception as e:
                print(f"[TELEGRAM] Errore per {chat_id}: {e}")

    except requests.exceptions.ConnectionError:
        print("[TELEGRAM] Errore: nessuna connessione internet")
    except Exception as e:
        print(f"[TELEGRAM] Errore imprevisto: {e}")

    finally:
        # Cancelliamo il file compresso temporaneo dopo l'invio
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

    # --- STATO CONDIVISO CON IL THREAD AI ---
    # Dizionario condiviso tra il loop principale e il thread AI
    stato = {
        'attivo':       True,          # False = ferma il thread AI
        'frame':        None,          # frame più recente da analizzare
        'uccelli':      [],            # risultato dell'ultima analisi
        'larghezza':    larghezza,     # dimensioni del frame
        'altezza':      altezza,
        'lock_frame':   threading.Lock(),   # protezione accesso al frame
        'lock_uccelli': threading.Lock(),   # protezione accesso agli uccelli
    }

    # Avviamo il thread AI in background
    # daemon=True = il thread si ferma automaticamente quando il programma esce
    t_ai = threading.Thread(target=thread_ai, args=(rete_ai, stato), daemon=True)
    t_ai.start()
    print("Thread AI avviato in background")

    # --- VARIABILI DI STATO ---

    sta_registrando        = False  # True = stiamo registrando
    scrittore_video        = None   # oggetto che scrive il file video
    tempo_ultimo_corvo     = None   # quando abbiamo visto l'ultimo uccello
    nome_file_video        = None   # nome del file video corrente
    secondi_corvo_totali   = 0.0    # quanti secondi il corvo è stato visibile in questa registrazione
    ultimo_tick_corvo      = None   # timestamp dell'ultimo frame in cui il corvo era visibile

    print("\n" + "=" * 55)
    print("  Monitoraggio ATTIVO — Premi CTRL+C per uscire")
    print(f"  Timer di stop: {SECONDI_SENZA_CORVO} secondi")
    print("=" * 55 + "\n")

    # Timestamp dell'ultimo controllo nuovi utenti Telegram
    ultimo_controllo_utenti = 0

    # --- CICLO PRINCIPALE ---

    try:
        while True:

            # Leggiamo un fotogramma dalla fotocamera il più veloce possibile
            # Il thread AI lavora in parallelo e non blocca mai questo ciclo
            successo, frame = fotocamera.read()

            # Se il frame non è valido, saltiamo
            if not successo or frame is None:
                time.sleep(0.01)
                continue

            # Aggiorniamo il frame condiviso con il thread AI
            with stato['lock_frame']:
                stato['frame'] = frame

            momento_attuale = time.time()

            # Ogni 30 secondi controlliamo se ci sono nuovi utenti Telegram
            if momento_attuale - ultimo_controllo_utenti >= 30:
                registra_nuovi_utenti()
                ultimo_controllo_utenti = momento_attuale

            # Leggiamo il risultato più recente del thread AI (senza aspettarlo)
            with stato['lock_uccelli']:
                uccelli_correnti = stato['uccelli']

            # C'è almeno un uccello visibile?
            corvo_visibile = len(uccelli_correnti) > 0


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
        stato['attivo'] = False  # diciamo al thread AI di fermarsi

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
