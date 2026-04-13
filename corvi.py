import os
import sys

# Silenziamo stderr a livello di sistema operativo PRIMA di tutto
# Questo elimina gli errori "Expected boundary" di ffmpeg
devnull = open(os.devnull, 'w')
os.dup2(devnull.fileno(), sys.stderr.fileno())

import cv2
import numpy as np
import time
import requests
import subprocess
import threading
from config import TELEGRAM_TOKEN

# ============================================================
# IMPOSTAZIONI
# ============================================================

SECONDI_SENZA_CORVO   = 30
CARTELLA_VIDEO        = "/sdcard/rilevatore_corvi/"
MODELLO_AI            = "/sdcard/rilevatore_corvi/yolov8n.onnx"
SOGLIA_CONFIDENZA     = 0.15   # abbassata per non perdere rilevamenti
CLASSE_UCCELLO        = 14
DIMENSIONE_MODELLO    = 640
ANALIZZA_OGNI_N_FRAME = 10     # AI ogni 10 frame — buon compromesso
SECONDI_MINIMI_CORVO  = 5
RISOLUZIONE_SALVATAGGIO = (1280, 720)
FPS_SALVATAGGIO       = 30
RISOLUZIONE_TELEGRAM  = (854, 480)
FPS_TELEGRAM          = 30
TELEGRAM_CANALE       = "@crowwatcher"
FILE_UTENTI           = "/sdcard/rilevatore_corvi/utenti.txt"
FILE_OFFSET           = "/sdcard/rilevatore_corvi/telegram_offset.txt"
DEBUG_AI              = True   # stampa cosa vede l'AI — metti False quando funziona

# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================

def crea_cartella_output():
    if not os.path.exists(CARTELLA_VIDEO):
        os.makedirs(CARTELLA_VIDEO)
        print(f"Cartella creata: {CARTELLA_VIDEO}")


def trova_uccelli(rete_ai, frame, larghezza_frame, altezza_frame):
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0 / 255.0,
        size=(DIMENSIONE_MODELLO, DIMENSIONE_MODELLO),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )
    rete_ai.setInput(blob)
    output = rete_ai.forward()
    predizioni = np.squeeze(output).T

    scala_x = larghezza_frame / DIMENSIONE_MODELLO
    scala_y = altezza_frame / DIMENSIONE_MODELLO
    uccelli_trovati = []
    miglior = (0.0, -1)

    for r in predizioni:
        punteggi = r[4:]
        classe   = int(np.argmax(punteggi))
        conf     = float(punteggi[classe])
        if conf > miglior[0]:
            miglior = (conf, classe)
        if classe == CLASSE_UCCELLO and conf >= SOGLIA_CONFIDENZA:
            cx = r[0] * scala_x
            cy = r[1] * scala_y
            w  = r[2] * scala_x
            h  = r[3] * scala_y
            uccelli_trovati.append({
                'x1': int(cx - w/2), 'y1': int(cy - h/2),
                'x2': int(cx + w/2), 'y2': int(cy + h/2),
                'confidenza': conf
            })

    if DEBUG_AI and miglior[0] > 0.05:
        nomi = {0:'persona',14:'uccello',15:'gatto',16:'cane',2:'auto',63:'laptop'}
        nome = nomi.get(miglior[1], f'classe_{miglior[1]}')
        tag  = " <<< UCCELLO!" if miglior[1] == CLASSE_UCCELLO else ""
        print(f"[AI] {nome} {miglior[0]*100:.1f}%{tag}   ", end='\r')

    return uccelli_trovati


def leggi_utenti():
    if not os.path.exists(FILE_UTENTI):
        return []
    with open(FILE_UTENTI, 'r') as f:
        return [r.strip() for r in f if r.strip()]


def registra_nuovi_utenti():
    offset = 0
    if os.path.exists(FILE_OFFSET):
        with open(FILE_OFFSET, 'r') as f:
            c = f.read().strip()
            if c:
                offset = int(c)
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        r = requests.get(url, params={'offset': offset, 'timeout': 2}, timeout=5)
        if r.status_code != 200:
            return
        aggiornamenti = r.json().get('result', [])
        if not aggiornamenti:
            return
        esistenti = set(leggi_utenti())
        nuovi = []
        for a in aggiornamenti:
            nuovo_offset = a['update_id'] + 1
            if nuovo_offset > offset:
                offset = nuovo_offset
            msg     = a.get('message', {})
            chat_id = str(msg.get('chat', {}).get('id', ''))
            if chat_id and chat_id not in esistenti:
                esistenti.add(chat_id)
                nuovi.append(chat_id)
                print(f"\n[TELEGRAM] Nuovo utente: {chat_id}")
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    data={'chat_id': chat_id, 'text': "Sei registrato! Riceverai i video dei corvi."},
                    timeout=10
                )
        if nuovi:
            with open(FILE_UTENTI, 'a') as f:
                for uid in nuovi:
                    f.write(uid + '\n')
        with open(FILE_OFFSET, 'w') as f:
            f.write(str(offset))
    except Exception:
        pass


def comprimi_video(percorso_originale):
    base, ext = os.path.splitext(percorso_originale)
    percorso_compresso = base + "_tg" + ext
    lw, lh = RISOLUZIONE_TELEGRAM
    print("[FFMPEG] Compressione...")
    try:
        subprocess.run([
            "ffmpeg", "-i", percorso_originale,
            "-vf", f"scale={lw}:{lh}",
            "-c:v", "libx264", "-crf", "18",
            "-preset", "fast", "-r", str(FPS_TELEGRAM),
            "-an", "-y", percorso_compresso
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        if os.path.exists(percorso_compresso):
            mb_orig = os.path.getsize(percorso_originale) / 1024 / 1024
            mb_comp = os.path.getsize(percorso_compresso) / 1024 / 1024
            print(f"[FFMPEG] {mb_orig:.1f}MB → {mb_comp:.1f}MB")
            return percorso_compresso
    except Exception as e:
        print(f"[FFMPEG] Errore: {e}")
    return percorso_originale


def invia_video_telegram(percorso_video, secondi_visibile):
    percorso_da_inviare = comprimi_video(percorso_video)
    utenti      = leggi_utenti()
    destinatari = [TELEGRAM_CANALE] + utenti
    print(f"[TELEGRAM] Invio a {len(destinatari)} destinatari...")
    mb = os.path.getsize(percorso_da_inviare) / 1024 / 1024
    didascalia = f"Corvo rilevato!\nVisibile: {secondi_visibile:.0f}s\nDim: {mb:.1f}MB"
    try:
        with open(percorso_da_inviare, 'rb') as f:
            contenuto = f.read()
        for chat_id in destinatari:
            try:
                r = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo",
                    data={'chat_id': chat_id, 'caption': didascalia},
                    files={'video': ('video.mp4', contenuto, 'video/mp4')},
                    timeout=180
                )
                if r.status_code == 200:
                    print(f"[TELEGRAM] Inviato a {chat_id}")
                else:
                    print(f"[TELEGRAM] Errore {chat_id}: {r.status_code}")
            except Exception as e:
                print(f"[TELEGRAM] Errore {chat_id}: {e}")
    except Exception as e:
        print(f"[TELEGRAM] Errore: {e}")
    finally:
        if percorso_da_inviare != percorso_video and os.path.exists(percorso_da_inviare):
            os.remove(percorso_da_inviare)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 55)
    print("   RILEVATORE DI CORVI")
    print("=" * 55)

    crea_cartella_output()

    if not os.path.exists(MODELLO_AI):
        print(f"ERRORE: modello non trovato in {MODELLO_AI}")
        return

    print("Caricamento modello AI...")
    rete_ai = cv2.dnn.readNetFromONNX(MODELLO_AI)
    print("Modello caricato!")

    print("Connessione fotocamera...")
    fotocamera = cv2.VideoCapture("http://127.0.0.1:8080/video")

    if not fotocamera.isOpened():
        print("ERRORE: IP Webcam non raggiungibile. Avvia il server nell'app.")
        return

    larghezza = int(fotocamera.get(cv2.CAP_PROP_FRAME_WIDTH))
    altezza   = int(fotocamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps       = int(fotocamera.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = FPS_SALVATAGGIO
    print(f"Fotocamera: {larghezza}x{altezza} @ {fps}fps")

    sta_registrando      = False
    scrittore_video      = None
    tempo_ultimo_corvo   = None
    nome_file_video      = None
    contatore_frame      = 0
    uccelli_correnti     = []
    secondi_corvo_totali = 0.0
    ultimo_tick_corvo    = None
    ultimo_check_utenti  = 0

    print("\n" + "=" * 55)
    print("  ATTIVO — CTRL+C per uscire")
    print("=" * 55 + "\n")

    try:
        while True:
            ok, frame = fotocamera.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            contatore_frame += 1
            momento_attuale  = time.time()

            # Analisi AI ogni N frame
            if contatore_frame % ANALIZZA_OGNI_N_FRAME == 0:
                uccelli_correnti = trova_uccelli(rete_ai, frame, larghezza, altezza)

            # Check nuovi utenti Telegram ogni 30 secondi
            if momento_attuale - ultimo_check_utenti >= 30:
                threading.Thread(target=registra_nuovi_utenti, daemon=True).start()
                ultimo_check_utenti = momento_attuale

            corvo_visibile = len(uccelli_correnti) > 0

            if corvo_visibile:
                tempo_ultimo_corvo = momento_attuale
                if ultimo_tick_corvo is not None:
                    secondi_corvo_totali += momento_attuale - ultimo_tick_corvo
                ultimo_tick_corvo = momento_attuale

                if not sta_registrando:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    nome_file_video = os.path.join(CARTELLA_VIDEO, f"corvo_{ts}.mp4")
                    codec = cv2.VideoWriter_fourcc(*'mp4v')
                    scrittore_video = cv2.VideoWriter(
                        nome_file_video, codec, FPS_SALVATAGGIO, RISOLUZIONE_SALVATAGGIO
                    )
                    sta_registrando = True
                    print(f"\n[REC] Inizio → {nome_file_video}")

            else:
                ultimo_tick_corvo = None
                if sta_registrando and tempo_ultimo_corvo is not None:
                    secondi_assenza = momento_attuale - tempo_ultimo_corvo
                    secondi_rimasti = SECONDI_SENZA_CORVO - secondi_assenza
                    if secondi_rimasti > 0:
                        print(f"[TIMER] Stop tra {int(secondi_rimasti)}s | visibile {secondi_corvo_totali:.0f}s   ", end='\r')

                    if secondi_assenza >= SECONDI_SENZA_CORVO:
                        scrittore_video.release()
                        scrittore_video = None
                        sta_registrando = False
                        if secondi_corvo_totali >= SECONDI_MINIMI_CORVO:
                            print(f"\n[SALVATO] {secondi_corvo_totali:.0f}s → {nome_file_video}")
                            threading.Thread(
                                target=invia_video_telegram,
                                args=(nome_file_video, secondi_corvo_totali),
                                daemon=True
                            ).start()
                        else:
                            os.remove(nome_file_video)
                            print(f"\n[SCARTATO] solo {secondi_corvo_totali:.0f}s")
                        nome_file_video      = None
                        tempo_ultimo_corvo   = None
                        uccelli_correnti     = []
                        secondi_corvo_totali = 0.0

            if sta_registrando and scrittore_video is not None:
                scrittore_video.write(cv2.resize(frame, RISOLUZIONE_SALVATAGGIO))

            if corvo_visibile and sta_registrando:
                print(f"[REC] {len(uccelli_correnti)} uccello/i | {secondi_corvo_totali:.0f}s totali   ", end='\r')
            elif not corvo_visibile and not sta_registrando:
                print("[IN ASCOLTO]   ", end='\r')

    except KeyboardInterrupt:
        print("\n\nChiusura...")

    finally:
        if scrittore_video is not None:
            scrittore_video.release()
            if secondi_corvo_totali >= SECONDI_MINIMI_CORVO:
                print(f"[SALVATO] {nome_file_video}")
                invia_video_telegram(nome_file_video, secondi_corvo_totali)
            else:
                os.remove(nome_file_video)
        fotocamera.release()
        print("Terminato.")


if __name__ == "__main__":
    main()
