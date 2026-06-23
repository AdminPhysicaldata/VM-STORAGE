#!/bin/bash
# Monte en SFTP les deux dossiers exposés par le serveur vm-storage
# ("sessions" et "sessions_envoyees", service "sftp" de son docker-compose),
# puis démarre pipeline_service.py dessus. Conçu pour tourner sur une machine
# distincte de vm-storage, qui n'a accès qu'au réseau (SFTP_HOST:SFTP_PORT).
#
# Doit être lancé avec --cap-add SYS_ADMIN --device /dev/fuse (FUSE en
# conteneur). sshfs DOIT tourner avec -f (foreground) + un '&' shell pour le
# passer en arrière-plan : son fork interne (daemonize) est cassé dans Docker
# et échoue silencieusement (le point de montage s'enregistre mais toute
# opération filesystem renvoie une erreur I/O).
set -u

mkdir -p /data/sessions /data/session_envoye

mount_sftp() {
    local remote_dir="$1" local_dir="$2"
    echo "[mount] ${SFTP_USER}@${SFTP_HOST}:${SFTP_PORT}/${remote_dir} -> ${local_dir}"
    echo "$SFTP_PASS" | sshfs \
        -p "$SFTP_PORT" \
        -o password_stdin \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o reconnect -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -f \
        "${SFTP_USER}@${SFTP_HOST}:${remote_dir}" "$local_dir" &
    echo $!
}

# stat (pas ls) pour le test de montage : un ls sur un dossier SSHFS avec des
# dizaines de milliers de sessions peut dépasser le timeout, stat sur le point
# de montage lui-même est immédiat.
wait_for_mount() {
    local dir="$1"
    for _ in $(seq 1 30); do
        stat "$dir" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

sessions_pid=$(mount_sftp "sessions" /data/sessions)
sent_pid=$(mount_sftp "sessions_envoyees" /data/session_envoye)

if ! wait_for_mount /data/sessions || ! wait_for_mount /data/session_envoye; then
    echo "[mount] échec du montage SFTP (sessions ou sessions_envoyees) — abandon" >&2
    exit 1
fi
echo "[mount] montages SFTP opérationnels (sessions pid=${sessions_pid}, sessions_envoyees pid=${sent_pid})"

exec python3 -u pipeline_service.py
