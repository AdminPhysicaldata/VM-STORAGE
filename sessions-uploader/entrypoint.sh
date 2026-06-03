#!/bin/sh
# Rend les variables d'environnement Docker disponibles pour les jobs cron
printenv | grep -v '^_=' > /etc/environment
exec cron -f
