#!/bin/sh
chown root:root /home/exoria
chmod 755 /home/exoria
chown 1000:1000 /home/exoria/sessions || true
chmod 775 /home/exoria/sessions || true
