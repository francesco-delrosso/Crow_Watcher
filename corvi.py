import os
import sys

# Silenziamo stderr prima di tutto per eliminare gli errori ffmpeg
devnull = open(os.devnull, 'w')
os.dup2(devnull.fileno(), sys.stderr.fileno())

import cv2
import numpy as np
import time
import sqlite3
import random
import requests
import subprocess
import threading
import queue
from config import TELEGRAM_TOKEN

# ============================================================
# IMPOSTAZIONI
# ============================================================

SECONDI_SENZA_CORVO     = 30
CARTELLA_VIDEO          = "/sdcard/rilevatore_corvi/"
MODELLO_AI              = "/sdcard/rilevatore_corvi/yolov8n.onnx"
DATABASE                = "/sdcard/rilevatore_corvi/corvi.db"
SOGLIA_CONFIDENZA       = 0.05
CLASSE_UCCELLO          = 14
DIMENSIONE_MODELLO      = 640
ZONA_RILEVAMENTO        = 1.00   # 1.0 = analizza tutto il frame
ANALIZZA_OGNI_N_FRAME   = 10
SECONDI_MINIMI_CORVO    = 5
SECONDI_BUFFER_FINE     = 4    # secondi di buffer dopo che il corvo sparisce
RISOLUZIONE_SALVATAGGIO = (1920, 1080)
FPS_SALVATAGGIO         = 30
RISOLUZIONE_TELEGRAM    = (1280, 720)
FPS_TELEGRAM            = 30
TELEGRAM_CANALE         = "@crowwatcher"
FILE_UTENTI             = "/sdcard/rilevatore_corvi/utenti.txt"
FILE_OFFSET             = "/sdcard/rilevatore_corvi/telegram_offset.txt"
DEBUG_AI                = True

# Frasi casuali sui corvi per il messaggio Telegram
FRASI_CORVI = [
    # Intelligenza e QI
    "🧠 QI superiore alla media. Non tua, quella dei corvidi.",
    "🪞 Ha superato il test dello specchio. Molti influencer no.",
    "🎓 Ha un cervello grande rispetto al corpo quanto quello dei delfini. Anche lui non nuota.",
    "🧩 Risolve puzzle a più step. Ha scoperto le cause-effetto. Congratulazioni a lui.",
    "📦 Nasconde il cibo e ricorda centinaia di nascondigli. Funziona meglio di Google Maps.",
    "🔢 Conta fino a 8. Più di quante dita usi per scrivere i messaggi vocali.",
    "📐 Capisce i concetti di volume e peso. Ha fatto fisica senza laurearsi.",
    "🧪 Usa strumenti di pietra. Ha saltato il Neolitico direttamente al problem-solving.",
    "🎯 Pianifica le mosse in anticipo come un giocatore di scacchi. Con le ali.",
    "🤯 Il suo cervello in proporzione è tra i più grandi del regno animale. Non commentare.",
    "📚 Impara guardando gli altri sbagliare. Tecnica che molti umani non hanno ancora adottato.",
    "🔬 Capisce la causalità. Sa che se fa X succede Y. Nobel non ancora assegnato.",

    # Memoria e vendetta
    "👁️ Ti ha già visto. Catalogato. Ha già deciso cosa pensa di te.",
    "😤 Se lo offendi, aspetta. Mesi. Anni se necessario. Non ha fretta.",
    "🗂️ Tiene un archivio mentale dei nemici. Aggiornato in tempo reale.",
    "📸 Ricorda le facce meglio di te dopo tre Aperol Spritz.",
    "🎭 Non dimentica chi è stato gentile con lui. Lista corta, molto curata.",
    "🕰️ Ha aspettato 6 mesi per vendicarsi di uno studioso che lo aveva spaventato. Pazienza infinita.",
    "🧳 Ha una lista nera. Sei in una lista. Non sappiamo quale.",
    "💾 Memoria a lungo termine attiva. Considera cosa hai fatto negli ultimi anni.",

    # Comportamento sociale
    "💀 Un gruppo di corvi si chiama 'omicidio'. Il nome lo hanno scelto loro.",
    "🤝 Si aiutano a vicenda anche senza essere parenti. Concetto rivoluzionario.",
    "😴 Dorme in gruppi enormi per sicurezza. Introvertito di giorno, animale sociale di notte.",
    "⚖️ Ha un senso della giustizia. Punisce chi imbroglia. Anche senza toga.",
    "🗣️ Tiene discorsi agli altri corvi. Non sappiamo cosa dice. Probabilmente niente di buono.",
    "👥 Fa il funerale ai corvi morti. Raduno solenne, nessuna battuta.",
    "🎪 Gioca con altri animali per divertimento. Ha un concetto di svago più sviluppato di molti.",
    "📣 Avvisa gli altri del pericolo anche a costo suo. Solidarietà vera.",
    "🚨 Organizza mob collettivi contro i predatori. Sa fare rete.",
    "🎁 Porta regali alle persone che gli piacciono. Rifletti sul fatto che a te non ha portato niente.",

    # Abilità fisiche e tecniche
    "🚗 Usa le macchine per schiacciare le noci. Ha capito come sfruttare gli umani.",
    "🕵️ Usa ramoscelli come strumenti. Ha inventato il bastone prima di noi.",
    "🪝 Piega i fili per farne uncini. Brevetto non depositato.",
    "🌊 Sa usare l'acqua per ammorbidire il cibo duro. Chef non certificato.",
    "✂️ Taglia le foglie nella forma giusta per i suoi scopi. Artigiano autodidatta.",
    "🔑 Apre contenitori con coperchi a vite. Considera di cambiare le serrature.",
    "🪨 Lascia cadere sassi sugli intrusi dall'alto. Ingegneria difensiva.",
    "🎣 Usa il pane come esca per pescare. Ha inventato la pesca sportiva.",
    "🏗️ Costruisce nidi strutturati con strati diversi per isolamento. Geometra abusivo.",
    "🌡️ Sceglie materiali diversi in base alla temperatura. Edilizia sostenibile.",

    # Comunicazione
    "🗣️ Dialetti diversi tra corvi di zone diverse. Più cultura linguistica del turista medio.",
    "📡 Oltre 250 vocalizzi distinti. Ha un vocabolario più ricco di certi tiktoker.",
    "🤫 Comunicazione silenziosa con gli occhi tra partner. Coppia affiatata.",
    "📢 Sa imitare la voce umana. Probabilmente lo sta già facendo.",
    "🔊 Sa imitare altri animali per ingannarli. Doppiogiochista certificato.",
    "💬 Ha un linguaggio specifico per 'pericolo', 'cibo', 'vieni qui'. Tre parole essenziali.",
    "🎵 Canta per piacere, non solo per comunicare. Ha già un album in mente.",
    "📻 Trasmette informazioni culturali alle generazioni successive. Tradizione orale funzionante.",

    # Adattamento e sopravvivenza
    "🌍 Vive su ogni continente tranne l'Antartide. Ha fatto scelte migliori di molti.",
    "🏙️ Prospera nelle città meglio della maggior parte dei residenti.",
    "🗑️ Ha imparato a vivere di spazzatura umana. Economia circolare.",
    "🌧️ Attivo in qualsiasi condizione meteo. Zero giorni di malattia.",
    "🌿 Onnivoro totale. Nessuna intolleranza alimentare.",
    "🏃 Velocità di volo fino a 50 km/h. Arriva prima di te ovunque.",
    "🌱 Si adatta a qualsiasi habitat in poche generazioni. Startup del mondo animale.",
    "🔄 Cambia strategia se quella attuale non funziona. Agilità mentale reale.",
    "🛡️ Sfrutta gli umani come protezione involontaria. Parassitismo elegante.",
    "🌐 Colonizzato ogni nicchia disponibile sul pianeta. Espansionismo senza eserciti.",

    # Vita di coppia e famiglia
    "💍 Si accoppia per tutta la vita. Fedeltà: 100%. Commento: omesso.",
    "👶 Insegna ai piccoli cosa fare e cosa non fare. Sistema educativo: funzionante.",
    "🏠 Torna allo stesso nido per anni. Ha risolto il problema degli affitti.",
    "👫 Il maschio porta cibo alla femmina durante la cova. Romantico o strategico, non si sa.",
    "🧸 I piccoli restano con i genitori fino a 3 anni. Almeno loro se ne vanno.",
    "🎓 I giovani vanno a scuola dai vecchi. Sistema di apprendistato medievale ma funziona.",
    "❤️ Fa le coccole alla compagna. Sì, i corvi fanno le coccole.",
    "🪺 Decora il nido con oggetti brillanti. Interior design istintivo.",

    # Esistenziale e filosofico
    "🐦‍⬛ È venuto, ha visto, ha giudicato silenziosamente, se n'è andato.",
    "🎭 Può fare il morto per ingannare i predatori. Tecnica usata anche in molte riunioni.",
    "☃️ Gioca nella neve per puro divertimento. Ha più work-life balance di qualsiasi CEO.",
    "🔮 Associato a presagi in ogni cultura del mondo. Coincidenza? Lui non commenta.",
    "⚡ Nella mitologia nordica era il messaggero di Odino. Carriera pregressa notevole.",
    "🌑 Simbolo di morte in Occidente, fortuna in Oriente. Ha un'immagine ambivalente.",
    "🪄 Considerato magico per migliaia di anni. PR naturale senza ufficio stampa.",
    "📖 Citato in Edgar Allan Poe, Shakespeare, Esopo. Curriculum letterario invidiabile.",
    "🌀 Associato alla trasformazione in quasi tutte le culture. Coach motivazionale ante litteram.",
    "🏺 Raffigurato nelle pitture rupestri. Presenza mediatica da 40.000 anni.",

    # Meta e ironici
    "📷 Sa che lo stai guardando. Ti sta valutando.",
    "🤔 Non sta mangiando. Sta pensando. Differenza sottile ma importante.",
    "😑 Quella faccia che fa? Non è neutra. Sta giudicando.",
    "🎬 Ogni suo movimento è intenzionale. Non esiste il 'per caso' nel vocabolario corvino.",
    "🕶️ Aspetto minaccioso, comportamento diplomatico. Estetica dark, valori solidi.",
    "🌟 Non ha bisogno della tua approvazione. Non l'ha mai cercata.",
    "🧘 Non si stresa. Osserva. Aspetta. Agisce al momento giusto.",
    "💼 Se fosse umano, sarebbe il tipo silenzioso in ufficio che poi diventa il capo.",
    "🎲 Non lascia niente al caso. Ha già calcolato le probabilità.",
    "🚀 Esiste da 17 milioni di anni. Noi da 300.000. Chi è il nuovo arrivato?",
    "👑 Non chiede rispetto. Lo ottiene automaticamente.",
    "🔭 Osserva dall'alto. Letteralmente e metaforicamente.",
    "⏳ Ha tutto il tempo del mondo. Non ha un calendario da rispettare.",
    "🎩 Elegante, intelligente, imprevedibile. Il James Bond degli uccelli.",
    "🌙 Attivo anche di notte se necessario. Non conosce il concetto di orario.",
    "🏆 Nessun predatore naturale da adulto. Ha raggiunto la vetta della catena alimentare locale.",
    "💭 Cosa pensa? Non lo sappiamo. Lui sì.",
    "🎯 Ogni azione ha uno scopo. Lo spreco di energia non è nel suo vocabolario.",
    "🦾 Sopravvissuto a cinque estinzioni di massa. Hai presente la resilienza? Ecco.",
    "🌈 Piumaggio che riflette colori cangianti sotto la luce. Bello e non lo sa. O lo sa.",
    "🍷 Se fosse un vino sarebbe un Barolo del 2008. Complesso, austero, incompreso dai più.",
    "🎻 Nella cultura giapponese porta fortuna. In quella irlandese porta sfortuna. Lui se ne frega in entrambi i casi.",
    "🥷 Silenzioso quando vuole. Quando lo senti è perché ha deciso di farti sapere che c'è.",
    "🌊 Ha attraversato oceani in migrazioni da 3.000 km. Il viaggio in ritardo alla stazione ti sembra ancora un problema?",
]

# ============================================================
# CAMERA THREAD
# ============================================================

_camera_frame = None
_camera_lock  = threading.Lock()
_camera_attiva = True

def _camera_worker(fotocamera):
    """Legge frame dalla camera in continuo, senza bloccare il loop principale."""
    global _camera_frame
    while _camera_attiva:
        ok, frame = fotocamera.read()
        if ok and frame is not None:
            with _camera_lock:
                _camera_frame = frame

def leggi_frame():
    with _camera_lock:
        return _camera_frame


# ============================================================
# AI THREAD
# ============================================================

# Coda con maxsize=1: tiene solo l'ultimo frame, l'AI non accumula lavoro arretrato
_coda_ai     = queue.Queue(maxsize=1)
_uccelli_ai  = []
_lock_ai     = threading.Lock()

def _ai_worker(rete_ai):
    global _uccelli_ai
    while True:
        try:
            frame = _coda_ai.get(timeout=1)
            risultato = trova_uccelli(rete_ai, frame)
            with _lock_ai:
                _uccelli_ai = risultato
        except queue.Empty:
            continue

def leggi_uccelli_ai():
    with _lock_ai:
        return list(_uccelli_ai)

def invia_frame_ad_ai(frame):
    try:
        _coda_ai.put_nowait(frame.copy())
    except queue.Full:
        pass


# ============================================================
# DATABASE
# ============================================================

def inizializza_db():
    conn = sqlite3.connect(DATABASE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS avvistamenti (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_inizio TEXT NOT NULL,
            timestamp_fine   TEXT,
            durata_secondi   REAL,
            nome_video       TEXT
        )
    """)
    conn.commit()
    conn.close()


def salva_avvistamento(timestamp_inizio, timestamp_fine, durata, nome_video):
    try:
        conn = sqlite3.connect(DATABASE)
        conn.execute(
            "INSERT INTO avvistamenti (timestamp_inizio, timestamp_fine, durata_secondi, nome_video) VALUES (?,?,?,?)",
            (timestamp_inizio, timestamp_fine, durata, nome_video)
        )
        conn.commit()
        conn.close()
        print(f"[DB] Avvistamento salvato: {durata:.0f}s")
    except Exception as e:
        print(f"[DB] Errore: {e}")


def conta_avvistamenti_oggi():
    try:
        oggi = time.strftime("%Y-%m-%d")
        conn = sqlite3.connect(DATABASE)
        n = conn.execute(
            "SELECT COUNT(*) FROM avvistamenti WHERE timestamp_inizio LIKE ?",
            (f"{oggi}%",)
        ).fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


# ============================================================
# AI
# ============================================================

def trova_uccelli(rete_ai, frame):
    # Dimensioni reali dal frame, non da fotocamera.get() che può restituire 0
    altezza_frame, larghezza_frame = frame.shape[:2]

    # Ritaglia la parte bassa del frame (strada) prima di analizzare
    righe_analisi = int(altezza_frame * ZONA_RILEVAMENTO)
    frame_analisi = frame[:righe_analisi, :]

    blob = cv2.dnn.blobFromImage(
        frame_analisi,
        scalefactor=1.0 / 255.0,
        size=(DIMENSIONE_MODELLO, DIMENSIONE_MODELLO),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )
    rete_ai.setInput(blob)
    predizioni = np.squeeze(rete_ai.forward()).T

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

    if DEBUG_AI:
        nomi = {0:'persona', 1:'bici', 2:'auto', 14:'uccello', 15:'gatto', 16:'cane', 47:'bicchiere', 63:'laptop', 67:'telefono'}
        # Top 3 classi per capire cosa vede l'AI
        tutti = []
        for r in predizioni:
            ps = r[4:]
            c  = int(np.argmax(ps))
            v  = float(ps[c])
            if v > 0.04:
                tutti.append((v, c))
        tutti.sort(reverse=True)
        top3 = tutti[:3]
        parti = []
        for v, c in top3:
            n = nomi.get(c, f'cls{c}')
            tag = '🐦' if c == CLASSE_UCCELLO else ''
            parti.append(f"{n}{tag} {v*100:.0f}%")
        print(f"[AI] frame {larghezza_frame}x{righe_analisi} | {' | '.join(parti) if parti else 'niente'}   ", end='\r')

    return uccelli_trovati


# ============================================================
# TELEGRAM
# ============================================================

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
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={'offset': offset, 'timeout': 2}, timeout=5
        )
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
            chat_id = str(a.get('message', {}).get('chat', {}).get('id', ''))
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


def aggiungi_a_compilazione(percorso_video):
    """Appende il video alla compilazione giornaliera."""
    oggi = time.strftime("%Y%m%d")
    compilazione = os.path.join(CARTELLA_VIDEO, f"compilazione_{oggi}.mp4")
    lista_file   = os.path.join(CARTELLA_VIDEO, f"lista_{oggi}.txt")

    # Aggiunge il video alla lista
    with open(lista_file, 'a') as f:
        f.write(f"file '{percorso_video}'\n")

    # Ricrea la compilazione con tutti i video del giorno
    print("[COMPILAZIONE] Aggiorno compilazione giornaliera...")
    try:
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", lista_file,
            "-c", "copy", "-y", compilazione
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        if os.path.exists(compilazione):
            mb = os.path.getsize(compilazione) / 1024 / 1024
            n  = open(lista_file).read().count("file '")
            print(f"[COMPILAZIONE] {n} video → {mb:.1f}MB → {compilazione}")
    except Exception as e:
        print(f"[COMPILAZIONE] Errore: {e}")


def invia_e_elimina_clip(percorso_video, secondi, ts):
    """Invia la clip su Telegram, poi la elimina dal tablet."""
    invia_video_telegram(percorso_video, secondi, ts)
    try:
        os.remove(percorso_video)
        print(f"[CLIP] Eliminata dal tablet: {os.path.basename(percorso_video)}")
    except Exception as e:
        print(f"[CLIP] Errore eliminazione: {e}")


LIMITE_TELEGRAM_MB = 49

def comprimi_video(percorso_originale):
    base, ext = os.path.splitext(percorso_originale)
    out = base + "_tg" + ext
    lw = RISOLUZIONE_TELEGRAM[0]
    mb1 = os.path.getsize(percorso_originale) / 1024 / 1024

    # Prova CRF crescente finché il file sta sotto il limite Telegram
    # Partiamo da CRF 20 perché la sorgente è già H.264 (minor degrado rispetto a mp4v)
    for crf in [20, 25, 30, 35, 40]:
        print(f"[FFMPEG] CRF {crf}...")
        try:
            subprocess.run([
                "ffmpeg", "-i", percorso_originale,
                "-vf", f"scale={lw}:-2",
                "-c:v", "libx264", "-crf", str(crf),
                "-preset", "fast", "-r", str(FPS_TELEGRAM),
                "-an", "-y", out
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        except Exception as e:
            print(f"[FFMPEG] Errore: {e}")
            return percorso_originale

        if os.path.exists(out):
            mb2 = os.path.getsize(out) / 1024 / 1024
            print(f"[FFMPEG] {mb1:.1f}MB → {mb2:.1f}MB (CRF {crf})")
            if mb2 <= LIMITE_TELEGRAM_MB:
                return out

    print(f"[FFMPEG] Impossibile scendere sotto {LIMITE_TELEGRAM_MB}MB, invio comunque")
    return out


def invia_video_telegram(percorso_video, secondi_visibile, timestamp_inizio):
    percorso_da_inviare = comprimi_video(percorso_video)
    utenti      = leggi_utenti()
    destinatari = [TELEGRAM_CANALE]
    print(f"[TELEGRAM] Invio al canale {TELEGRAM_CANALE}...")

    # Costruiamo il messaggio divertente
    n_oggi    = conta_avvistamenti_oggi()
    ora       = time.strftime("%H:%M", time.localtime())
    mb        = os.path.getsize(percorso_da_inviare) / 1024 / 1024
    fatto      = random.choice(FRASI_CORVI)

    emoji_numero = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
    em = emoji_numero[min(n_oggi - 1, 9)] if n_oggi >= 1 else "🐦‍⬛"

    didascalia = (
        f"🐦‍⬛ CORVO AVVISTATO 🐦‍⬛\n"
        f"\n"
        f"🕐 Ore {ora}\n"
        f"⏱️ Visibile per {secondi_visibile:.0f} secondi\n"
        f"{em} Avvistamento numero {n_oggi} di oggi\n"
        f"\n"
        f"{fatto}"
    )

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

    if not os.path.exists(CARTELLA_VIDEO):
        os.makedirs(CARTELLA_VIDEO)

    inizializza_db()
    print("[DB] Database pronto")

    if not os.path.exists(MODELLO_AI):
        print(f"ERRORE: modello non trovato in {MODELLO_AI}")
        return

    print("Caricamento modello AI...")
    rete_ai = cv2.dnn.readNetFromONNX(MODELLO_AI)
    print("Modello caricato!")

    print("Connessione fotocamera...")
    fotocamera = cv2.VideoCapture("http://admin:admin@192.168.1.56:8081/video")
    if not fotocamera.isOpened():
        print("ERRORE: IP Webcam non raggiungibile.")
        return

    # Richiedi risoluzione nativa C920
    fotocamera.set(cv2.CAP_PROP_FRAME_WIDTH,  RISOLUZIONE_SALVATAGGIO[0])
    fotocamera.set(cv2.CAP_PROP_FRAME_HEIGHT, RISOLUZIONE_SALVATAGGIO[1])
    fotocamera.set(cv2.CAP_PROP_FPS, FPS_SALVATAGGIO)

    larghezza = int(fotocamera.get(cv2.CAP_PROP_FRAME_WIDTH))
    altezza   = int(fotocamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Fotocamera: {larghezza}x{altezza}")

    # Thread camera: legge frame in continuo senza bloccare il loop
    threading.Thread(target=_camera_worker, args=(fotocamera,), daemon=True).start()
    print("[CAM] Thread avviato")

    # Aspetta il primo frame
    while leggi_frame() is None:
        time.sleep(0.05)

    # Avvia il thread AI
    threading.Thread(target=_ai_worker, args=(rete_ai,), daemon=True).start()
    print("[AI] Thread avviato")

    # --- Stato ---
    sta_registrando      = False
    scrittore_video      = None
    tempo_ultimo_corvo   = None
    nome_file_video      = None
    timestamp_inizio_av  = None
    contatore_frame      = 0
    secondi_corvo_totali = 0.0
    ultimo_tick_corvo    = None
    ultimo_check_utenti  = 0

    print("\n" + "=" * 55)
    print("  ATTIVO — CTRL+C per uscire")
    print("=" * 55 + "\n")

    intervallo_frame = 1.0 / FPS_SALVATAGGIO
    prossimo_tick    = time.time()

    try:
        while True:
            # Ritmo fisso: scrivi esattamente FPS_SALVATAGGIO frame al secondo
            ora = time.time()
            attesa = prossimo_tick - ora
            if attesa > 0:
                time.sleep(attesa)
            prossimo_tick += intervallo_frame

            # Prende l'ultimo frame disponibile (mai bloccante)
            frame = leggi_frame()
            if frame is None:
                continue

            contatore_frame += 1
            momento_attuale  = time.time()

            # Ogni N frame manda il frame all'AI (non-bloccante)
            if contatore_frame % ANALIZZA_OGNI_N_FRAME == 0:
                invia_frame_ad_ai(frame)

            # Legge il risultato più recente dell'AI (mai bloccante)
            uccelli_correnti = leggi_uccelli_ai()

            # Check nuovi utenti ogni 30 secondi
            if momento_attuale - ultimo_check_utenti >= 30:
                threading.Thread(target=registra_nuovi_utenti, daemon=True).start()
                ultimo_check_utenti = momento_attuale

            corvo_visibile = len(uccelli_correnti) > 0

            # Scrivi il frame ogni volta che la registrazione è attiva
            if sta_registrando and scrittore_video is not None:
                frame_rid = cv2.resize(frame, RISOLUZIONE_SALVATAGGIO)
                try:
                    scrittore_video.stdin.write(frame_rid.tobytes())
                except Exception:
                    pass

            if corvo_visibile:
                tempo_ultimo_corvo = momento_attuale
                if ultimo_tick_corvo is not None:
                    secondi_corvo_totali += momento_attuale - ultimo_tick_corvo
                ultimo_tick_corvo = momento_attuale

                if not sta_registrando:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    timestamp_inizio_av = time.strftime("%Y-%m-%d %H:%M:%S")
                    nome_file_video = os.path.join(CARTELLA_VIDEO, f"corvo_{ts}.mp4")
                    lw, lh = RISOLUZIONE_SALVATAGGIO
                    scrittore_video = subprocess.Popen([
                        "ffmpeg", "-y",
                        "-f", "rawvideo", "-vcodec", "rawvideo",
                        "-s", f"{lw}x{lh}",
                        "-pix_fmt", "bgr24",
                        "-r", str(FPS_SALVATAGGIO),
                        "-i", "pipe:0",
                        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                        "-pix_fmt", "yuv420p",
                        nome_file_video
                    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    sta_registrando = True
                    print(f"\n[REC] Inizio → {nome_file_video}")

                print(f"[REC] {len(uccelli_correnti)} corvo/i | {secondi_corvo_totali:.0f}s   ", end='\r')

            else:
                ultimo_tick_corvo = None

                if sta_registrando and tempo_ultimo_corvo is not None:
                    secondi_assenza = momento_attuale - tempo_ultimo_corvo

                    secondi_rimasti = SECONDI_SENZA_CORVO - secondi_assenza
                    if secondi_rimasti > 0:
                        print(f"[TIMER] Stop tra {int(secondi_rimasti)}s | totale {secondi_corvo_totali:.0f}s   ", end='\r')

                    if secondi_assenza >= SECONDI_SENZA_CORVO:
                        scrittore_video.stdin.close()
                        scrittore_video.wait()
                        scrittore_video = None
                        sta_registrando = False

                        timestamp_fine = time.strftime("%Y-%m-%d %H:%M:%S")

                        if secondi_corvo_totali >= SECONDI_MINIMI_CORVO:
                            print(f"\n[SALVATO] {secondi_corvo_totali:.0f}s → {nome_file_video}")
                            # Salviamo nel DB
                            salva_avvistamento(
                                timestamp_inizio_av, timestamp_fine,
                                secondi_corvo_totali, nome_file_video
                            )
                            # Aggiunge alla compilazione giornaliera (sincrono, deve finire prima di eliminare)
                            aggiungi_a_compilazione(nome_file_video)
                            # Invia su Telegram e poi elimina la clip dal tablet
                            _ts = timestamp_inizio_av
                            _nf = nome_file_video
                            _s  = secondi_corvo_totali
                            threading.Thread(
                                target=invia_e_elimina_clip,
                                args=(_nf, _s, _ts),
                                daemon=True
                            ).start()
                        else:
                            os.remove(nome_file_video)
                            print(f"\n[SCARTATO] solo {secondi_corvo_totali:.0f}s di corvo")

                        nome_file_video      = None
                        tempo_ultimo_corvo   = None
                        uccelli_correnti     = []
                        secondi_corvo_totali = 0.0
                        timestamp_inizio_av  = None

                elif not sta_registrando:
                    print("[IN ASCOLTO]   ", end='\r')

    except KeyboardInterrupt:
        print("\n\nChiusura...")

    finally:
        if scrittore_video is not None:
            scrittore_video.stdin.close()
            scrittore_video.wait()
            if secondi_corvo_totali >= SECONDI_MINIMI_CORVO:
                ts_fine = time.strftime("%Y-%m-%d %H:%M:%S")
                salva_avvistamento(timestamp_inizio_av, ts_fine, secondi_corvo_totali, nome_file_video)
                aggiungi_a_compilazione(nome_file_video)
                print(f"[SALVATO] {nome_file_video}")
                invia_e_elimina_clip(nome_file_video, secondi_corvo_totali, timestamp_inizio_av)
            else:
                os.remove(nome_file_video)
        global _camera_attiva
        _camera_attiva = False
        fotocamera.release()
        print("Terminato.")


if __name__ == "__main__":
    main()
