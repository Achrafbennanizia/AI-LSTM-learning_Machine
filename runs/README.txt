TensorBoard reads only event files created by train.py.
Run training at least once before:
  tensorboard --logdir runs --samples_per_plugin=scalars=10000000
(Default TensorBoard downsamples scalars per tag; the flag raises the cap so curves use the full logged length.)

Event files live in: runs/<timestamp-or-run_name>/events.out.tfevents.*

Local:  cd <project> && tensorboard --logdir runs --samples_per_plugin=scalars=10000000
HPC login node:  cd <project-dir> && .venv/bin/tensorboard --logdir runs --samples_per_plugin=scalars=10000000

Falls „tensorboard: command not found“: venv aktivieren (source .venv/bin/activate) oder vollen Pfad:
  .venv/bin/tensorboard ...

Falls „could not bind to unsupported address family ::“ (IPv6 auf dem Knoten kaputt): NICHT --bind_all verwenden,
sondern IPv4 erzwingen, z.B. nur Login-Node + SSH-Tunnel:
  .venv/bin/tensorboard --logdir runs/ECHTER_RUN_ORDNER --port 6007 --host 127.0.0.1
  # Laptop: ssh -N -L 6007:127.0.0.1:6007 USER@LOGINNODE
Erreichbar von anderen Hosts aus (Firewall beachten):
  .venv/bin/tensorboard --logdir runs/ECHTER_RUN_ORDNER --port 6007 --host 0.0.0.0
(Liste echte Runs mit: ls runs/)

Nur einen Lauf ohne alte Graphen überlagern zu lassen:

  tensorboard --logdir runs/20260201_143022 --samples_per_plugin=scalars=10000000

(Den genauen Ordnernamen druckt train.py nach dem Training ebenfalls.)

Skalar-Schritte: Standard X-Achse = kumulierte Minibatches (loss/train_batch + Ende-Epoch-Metriken); mit train.py --no_tb_batch_scalars nur noch ~Anzahl Epochen.
Zwei Accuracy-Skalen in TensorBoard: accuracy/test_* = 0–1; epoch/acc_test_*_pct = Prozent (0–100).
