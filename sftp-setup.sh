#!/bin/sh
chown root:root /home/exoria
chmod 755 /home/exoria
# exFAT ne supporte pas chown — les permissions viennent des options de montage (uid=1000,gid=1000,umask=022)
# Ce chmod est un fallback pour les filesystems qui l'ignorent ; sur exFAT c'est sans effet
chmod 775 /home/exoria/sessions || true
